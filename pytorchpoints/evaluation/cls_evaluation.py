import itertools
import json
import logging
import numpy as np
import os
from collections import OrderedDict
import PIL.Image as Image
import torch

from pytorchpoints.config import kfg, configurable
from pytorchpoints.utils.comm import all_gather, is_main_process, synchronize
from pytorchpoints.utils.file_io import PathManager
from .evaluator import DatasetEvaluator, AverageMeter, accuracy
from pytorchpoints.datasets.transforms import PointcloudScaleAndJitter, PointcloudTestRotate
from pytorchpoints.functional import get_cls_features
from pytorchpoints.utils import comm
from .build import EVALUATION_REGISTRY

__all__ = ["ClsEvaluator"]

@EVALUATION_REGISTRY.register()
class ClsEvaluator(DatasetEvaluator):
    @configurable
    def __init__(self):
        super(ClsEvaluator, self).__init__()

    @classmethod
    def from_config(cls, cfg):
        return {}

    def evaluate(self, cfg, model, test_data_loader, epoch):
        logger = logging.getLogger(__name__)
        vote_preds = None
        TS = PointcloudScaleAndJitter(scale_low=cfg.AUG.SCALE_LOW,
                                      scale_high=cfg.AUG.SCALE_HIGH,
                                      std=cfg.AUG.NOISE_STD,
                                      clip=cfg.AUG.NOISE_CLIP)
        if cfg.AUG.TEST_ROTATE:
            if cfg.ENGINE.BEST_METRICS == ['SO3-ACC@1']:
                r_types = ['SO3']
            elif cfg.ENGINE.BEST_METRICS == ['Z-ACC@1']:
                r_types = ['Z', 'SO3']
            else:
                raise NotImplementedError(
                    f"BEST_METRICS {cfg.ENGINE.BEST_METRICS} in TEST_ROTATE not supported")
        else:
            r_types = ['None']
        eval_res = {}
        for r in r_types:
            rotate = PointcloudTestRotate(r)
            num_votes = cfg.AUG.NUM_VOTES if epoch >= cfg.SOLVER.VOTE_START else 1
            for v in range(num_votes):
                preds = []
                targets = []
                for idx, data in enumerate(test_data_loader):
                    # augment for voting
                    if v > 0 or r != 'None':
                        if cfg.MODEL.INFEATS_DIM not in (3, 4):
                            raise NotImplementedError(
                                f"input_features_dim {cfg.MODEL.INFEATS_DIM} in voting not supported")
                        for d in data:
                            points = d[kfg.POINTS]
                            if v > 0:
                                points = TS(points)
                            if r != 'None':
                                points = rotate(points)
                            pc = points[:, :3]
                            normal = points[:, 3:]
                            features = get_cls_features(cfg.MODEL.INFEATS_DIM, pc, normal)
                            d[kfg.POINTS] = points
                            d[kfg.FEATS] = features
                    data = comm.unwrap_model(model).preprocess_batch(data)
                    # forward
                    with torch.cuda.amp.autocast(enabled=cfg.ENGINE.FP16):
                        pred = model(data)
                    target = data[kfg.LABELS].view(-1)
                    preds.append(pred)
                    targets.append(target)
                preds = torch.cat(preds, 0)
                targets = torch.cat(targets, 0)
                if comm.get_world_size() > 1:
                    comm.synchronize()
                    preds_list = comm.all_gather(preds.cpu())
                    targets_list = comm.all_gather(targets.cpu())
                    preds_list = [torch.from_numpy(ent.cpu().data.numpy()) for ent in preds_list]
                    targets_list = [torch.from_numpy(ent.cpu().data.numpy()) for ent in targets_list]
                    preds = torch.cat(preds_list, 0)
                    targets = torch.cat(targets_list, 0)
                if v == 0:
                    vote_preds = preds
                else:
                    vote_preds += preds
                vote_acc1 = accuracy(vote_preds, targets, topk=(1,))[0].item()
                logger.info(f' * Vote{v} Acc@1 {vote_acc1:.3%} in {preds.shape[0]} points')
            if r == 'None':
                eval_res['ACC@1'] = vote_acc1
            else:
                eval_res[r + '-ACC@1'] = vote_acc1

        return eval_res