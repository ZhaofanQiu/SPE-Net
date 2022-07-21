import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import PIL.Image as Image
import torch
import itertools
from pytorchpoints.config import kfg, configurable
from pytorchpoints.utils.comm import all_gather, is_main_process, synchronize
from pytorchpoints.utils.file_io import PathManager
from .evaluator import DatasetEvaluator, AverageMeter, s3dis_metrics
from pytorchpoints.datasets.transforms import PointcloudScaleAndJitter, PointcloudRandomRotate
from pytorchpoints.functional import get_scene_seg_features
from pytorchpoints.utils import comm
from .build import EVALUATION_REGISTRY

__all__ = ["S3DISEvaluator"]

@EVALUATION_REGISTRY.register()
class S3DISEvaluator(DatasetEvaluator):
    @configurable
    def __init__(self):
        super(S3DISEvaluator, self).__init__()

    @classmethod
    def from_config(cls, cfg):
        return {}

    def transformer(self, cfg, data, TS, RT, vote_idx):
        if vote_idx > 0:
            for d in data:
                points = d[kfg.POINTS]
                points = RT(points)
                points = TS(points)
                d[kfg.POINTS] = points
        return data

    def evaluate(self, cfg, model, test_data_loader, epoch):
        logger = logging.getLogger(__name__)
        
        RT = PointcloudRandomRotate(cfg.AUG.X_ANGLE_RANGE, cfg.AUG.Y_ANGLE_RANGE, cfg.AUG.Z_ANGLE_RANGE)
        TS = PointcloudScaleAndJitter(scale_low=cfg.AUG.SCALE_LOW,
            scale_high=cfg.AUG.SCALE_HIGH,
            std=cfg.AUG.NOISE_STD,
            clip=cfg.AUG.NOISE_CLIP,
            augment_symmetries=cfg.AUG.AUGMENT_SYMMETRIES)

        vote_logits_sum = [np.zeros((cfg.MODEL.NUM_CLASSES, l.shape[0]), dtype=np.float32) for l in
                           test_data_loader.dataset._map_func._obj.sub_clouds_points_labels]
        vote_counts = [np.zeros((1, l.shape[0]), dtype=np.float32) + 1e-6 for l in
                       test_data_loader.dataset._map_func._obj.sub_clouds_points_labels]
        vote_logits = [np.zeros((cfg.MODEL.NUM_CLASSES, l.shape[0]), dtype=np.float32) for l in
                       test_data_loader.dataset._map_func._obj.sub_clouds_points_labels]
        validation_proj = test_data_loader.dataset._map_func._obj.projections
        validation_labels = test_data_loader.dataset._map_func._obj.clouds_points_labels

        val_proportions = np.zeros(cfg.MODEL.NUM_CLASSES, dtype=np.float32)
        for label_value in range(cfg.MODEL.NUM_CLASSES):
            val_proportions[label_value] = np.sum(
                [np.sum(labels == label_value) for labels in test_data_loader.dataset._map_func._obj.clouds_points_labels])
        
        eval_res = {}
        num_votes = cfg.AUG.NUM_VOTES if epoch >= cfg.SOLVER.VOTE_START else 1
        for v in range(num_votes):
            test_data_loader.dataset._map_func._obj.epoch = (epoch - 1 + v) % 20

            for idx, data in enumerate(test_data_loader):
                data = self.transformer(cfg, data, TS, RT, v)
                data = comm.unwrap_model(model).preprocess_batch(data)
                with torch.cuda.amp.autocast(enabled=cfg.ENGINE.FP16):
                    pred = model(data)
            
                bsz = pred.shape[0]
                mask = data[kfg.MASKS]
                cloud_label = data[kfg.CLOUD_INDEX]
                input_inds = data[kfg.INPUT_INDEX]
                for ib in range(bsz):
                    mask_i = mask[ib].cpu().numpy().astype(np.bool)
                    logits = pred[ib].cpu().numpy()[:, mask_i]
                    inds = input_inds[ib].cpu().numpy()[mask_i]
                    c_i = cloud_label[ib].item()
                    vote_logits_sum[c_i][:, inds] = vote_logits_sum[c_i][:, inds] + logits
                    vote_counts[c_i][:, inds] += 1
                    vote_logits[c_i] = vote_logits_sum[c_i] / vote_counts[c_i]

            if comm.get_world_size() > 1:
                comm.synchronize()
                all_vote_logits_sum = comm.all_gather(vote_logits_sum[0])
                all_vote_counts = comm.all_gather(vote_counts[0])
                
                reduce_sum = 0
                reduce_count = 0
                for vote_logit in all_vote_logits_sum:
                    reduce_sum += vote_logit
                for vote_count in all_vote_counts:
                    reduce_count += vote_count
                vote_logits = [reduce_sum / reduce_count]
                
            IoUs, mIoU = s3dis_metrics(cfg.MODEL.NUM_CLASSES, vote_logits, validation_proj, validation_labels)
            logger.info(f'E{epoch} V{v} * mIoU {mIoU:.3%}')
            logger.info(f'E{epoch} V{v}  * msIoU {IoUs}')

        eval_res['mIoU'] = mIoU
        return eval_res
