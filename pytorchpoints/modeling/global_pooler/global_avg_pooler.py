import torch
from torch import nn
from pytorchpoints.config import kfg
from pytorchpoints.config import configurable
from .build import GLOBAL_POOLER_REGISTRY

__all__ = ["GlobalAvgPooler"]

@GLOBAL_POOLER_REGISTRY.register()
class GlobalAvgPooler(nn.Module):
    @configurable
    def __init__(self):
        super(GlobalAvgPooler, self).__init__()

    @classmethod
    def from_config(cls, cfg):
        return {}

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs):
        feats = batched_inputs[kfg.FEATS]
        if kfg.MASKS in batched_inputs:
            out = feats.sum(-1)
            masks = batched_inputs[kfg.MASKS]
            pcl_num = masks.sum(-1, keepdim=True)
            out = out / pcl_num
        else:
            out = feats.mean(-1)
        return { kfg.FEATS: out }