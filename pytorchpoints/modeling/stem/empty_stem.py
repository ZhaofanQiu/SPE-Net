import torch
from torch import nn

from pytorchpoints.config import kfg
from pytorchpoints.config import configurable
from .build import STEM_REGISTRY
from ..local_aggregation import build_local_aggregation

__all__ = ["EmptyStem"]

@STEM_REGISTRY.register()
class EmptyStem(nn.Module):
    @configurable
    def __init__(self, in_dim, width, la):
        super(EmptyStem, self).__init__()
        self.la = la
        self.width = width

    @classmethod
    def from_config(cls, cfg):
        return {
            "in_dim": cfg.MODEL.INFEATS_DIM,
            "width": cfg.MODEL.STEM.WIDTH,

            "la": build_local_aggregation(
                cfg,
                cfg.MODEL.LA_TYPE.NAME,
                cfg.MODEL.STEM.WIDTH,
                cfg.MODEL.STEM.WIDTH,
                cfg.MODEL.BACKBONE.RADIUS,
                cfg.MODEL.BACKBONE.NSAMPLES[0],
                cfg.MODEL.DEVICE
            ) if len(cfg.MODEL.LA_TYPE.NAME) > 0 else None
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs):
        B, C, N = batched_inputs[kfg.FEATS].shape
        features = torch.zeros([B, self.width, N]).cuda()
        if self.la is not None:
            xyz = batched_inputs[kfg.POINTS]
            mask = batched_inputs[kfg.MASKS]
            features = self.la(xyz, xyz, mask, mask, features)
        return {kfg.FEATS: features}