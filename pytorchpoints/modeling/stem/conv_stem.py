import torch
from torch import nn

from pytorchpoints.config import kfg
from pytorchpoints.config import configurable
from .build import STEM_REGISTRY
from ..local_aggregation import build_local_aggregation

__all__ = ["ConvStem"]

@STEM_REGISTRY.register()
class ConvStem(nn.Module):
    @configurable
    def __init__(self, in_dim, width, bnact, la):
        super(ConvStem, self).__init__()
        if bnact:
            self.conv1 = nn.Sequential(
                nn.Conv1d(in_dim, width, kernel_size=1, bias=False),
                nn.BatchNorm1d(width),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv1 = nn.Conv1d(in_dim, width, kernel_size=1, bias=False)
        self.la = la

    @classmethod
    def from_config(cls, cfg):  
        return {
            "in_dim": cfg.MODEL.INFEATS_DIM,
            "width": cfg.MODEL.STEM.WIDTH,
            "bnact": cfg.MODEL.STEM.USE_BNACT,
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
        features = batched_inputs[kfg.FEATS]
        features = self.conv1(features)
        if self.la is not None:
            xyz = batched_inputs[kfg.POINTS]
            mask = batched_inputs[kfg.MASKS]
            features = self.la(xyz, xyz, mask, mask, features)
        return { kfg.FEATS: features }