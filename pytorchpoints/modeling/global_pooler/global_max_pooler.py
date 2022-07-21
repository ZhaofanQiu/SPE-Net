import torch
from torch import nn
from pytorchpoints.config import kfg
from pytorchpoints.config import configurable
from .build import GLOBAL_POOLER_REGISTRY

__all__ = ["GlobalMaxPooler"]

@GLOBAL_POOLER_REGISTRY.register()
class GlobalMaxPooler(nn.Module):
    @configurable
    def __init__(self):
        super(GlobalMaxPooler, self).__init__()

    @classmethod
    def from_config(cls, cfg):
        return {}

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs):
        feats = batched_inputs[kfg.FEATS]
        out = torch.nn.functional.adaptive_max_pool1d(feats, 1).squeeze(dim=-1)
        return { kfg.FEATS: out }