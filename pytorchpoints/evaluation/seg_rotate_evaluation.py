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
from .evaluator import DatasetEvaluator, AverageMeter, partnet_metrics
from pytorchpoints.datasets.transforms import PointcloudScaleAndJitter, PointcloudTestRotate
from pytorchpoints.functional import get_part_seg_features
from pytorchpoints.utils import comm
from .build import EVALUATION_REGISTRY

__all__ = ["SegRotateEvaluator"]

@EVALUATION_REGISTRY.register()
class SegRotateEvaluator(DatasetEvaluator):
    @configurable
    def __init__(self):
        super(SegRotateEvaluator, self).__init__()

    @classmethod
    def from_config(cls, cfg):
        return {}

    def transformer(self, cfg, data, TS, rotate, vote_idx):
        for d in data:
            points = d[kfg.POINTS]
            if vote_idx > 0:
                points = TS(points)
            if rotate is not None:
                points = rotate(points)
            pc = points[:, :3]
            features = get_part_seg_features(cfg.MODEL.INFEATS_DIM, pc)
            d[kfg.POINTS] = points
            d[kfg.FEATS] = features
        return data

    def one_vote(self, cfg, model, test_data_loader, TS, rotate, vote_idx):
        all_logits = []
        all_points_labels = []
        all_shape_labels = []

        for idx, data in enumerate(test_data_loader):
            data = self.transformer(cfg, data, TS, rotate, vote_idx)
            data = comm.unwrap_model(model).preprocess_batch(data)
            with torch.cuda.amp.autocast(enabled=cfg.ENGINE.FP16):
                pred = model(data)
            points_labels = data[kfg.POINTS_LABELS]
            shape_labels = data[kfg.LABELS]

            bsz = pred[0].shape[0]
            for ib in range(bsz):
                sl = shape_labels[ib]
                logits = pred[sl][ib]
                pl = points_labels[ib]
                all_logits.append(logits.cpu().numpy())
                all_points_labels.append(pl.cpu().numpy())
                all_shape_labels.append(sl.cpu().numpy())
        return all_logits, all_points_labels, all_shape_labels


    def evaluate(self, cfg, model, test_data_loader, epoch):
        logger = logging.getLogger(__name__)
        eval_res = {}
        TS = PointcloudScaleAndJitter(scale_low=cfg.AUG.SCALE_LOW,
                                      scale_high=cfg.AUG.SCALE_HIGH,
                                      std=cfg.AUG.NOISE_STD,
                                      clip=cfg.AUG.NOISE_CLIP)

        num_votes = cfg.AUG.NUM_VOTES if epoch >= cfg.SOLVER.VOTE_START else 1

        if cfg.AUG.TEST_ROTATE:
            if cfg.ENGINE.BEST_METRICS == ['SO3-mmpIoU']:
                r_types = ['SO3']
            elif cfg.ENGINE.BEST_METRICS == ['Z-mmpIoU']:
                r_types = ['Z', 'SO3']
            else:
                raise NotImplementedError(
                    f"BEST_METRICS {cfg.ENGINE.BEST_METRICS} in TEST_ROTATE not supported")
        else:
            r_types = ['None']
        eval_res = {}
        for r in r_types:
            vote_logits = None
            vote_points_labels = None
            vote_shape_labels = None

            rotate = PointcloudTestRotate(r)
            for v in range(num_votes):
                all_logits, all_points_labels, all_shape_labels = self.one_vote(cfg, model, test_data_loader, TS, rotate, v)

                if comm.get_world_size() > 1:
                    comm.synchronize()
                    all_logits = comm.all_gather(all_logits)
                    all_logits = list(itertools.chain(*all_logits))
                    if vote_logits is None:
                        all_points_labels = comm.all_gather(all_points_labels)
                        all_points_labels = list(itertools.chain(*all_points_labels))
                        all_shape_labels = comm.all_gather(all_shape_labels)
                        all_shape_labels = list(itertools.chain(*all_shape_labels))

                if vote_logits is None:
                    vote_logits = all_logits
                    vote_points_labels = all_points_labels
                    vote_shape_labels = all_shape_labels
                else:
                    for i in range(len(vote_logits)):
                        vote_logits[i] = vote_logits[i] + (all_logits[i] - vote_logits[i]) / (v + 1)

                msIoU, mpIoU, mmsIoU, mmpIoU = partnet_metrics(
                    cfg.MODEL.NUM_CLASSES,
                    cfg.DATALOADER.NUM_PARTS,
                    vote_shape_labels,
                    vote_logits,
                    vote_points_labels)

                logger.info(f' * Vote{v} mmsIoU {mmsIoU:.3%} in {all_points_labels[0].shape[0]} points')
                logger.info(f' * Vote{v} mmpIoU {mmpIoU:.3%} in {all_points_labels[0].shape[0]} points')
                logger.info(f' * Vote{v} msIoU {str(msIoU)} in {all_points_labels[0].shape[0]} points')
                logger.info(f' * Vote{v} mpIoU {str(mpIoU)} in {all_points_labels[0].shape[0]} points')

            if r == 'None':
                eval_res['mmsIoU'] = mmsIoU
                eval_res['mmpIoU'] = mmpIoU
            else:
                eval_res[r + '-mmsIoU'] = mmsIoU
                eval_res[r + '-mmpIoU'] = mmpIoU

        return eval_res
