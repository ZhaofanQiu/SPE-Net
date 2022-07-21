import torch
from torch import nn

from .base_net import BaseNet
from pytorchpoints.config import configurable
from .build import META_ARCH_REGISTRY

from ..stem import build_stem
from ..backbones import build_backbone
from ..global_pooler import build_pooler
from ..heads import build_head

__all__ = ["ClsNet"]

@META_ARCH_REGISTRY.register()
class ClsNet(BaseNet):
    @configurable
    def __init__(self, stem, backbone, pooler, head):
        super(ClsNet, self).__init__()
        self.stem = stem
        self.backbone = backbone
        self.pooler = pooler
        self.head = head

    @classmethod
    def from_config(cls, cfg):
        return {
            'stem': build_stem(cfg),
            'backbone': build_backbone(cfg),
            'pooler': build_pooler(cfg),
            'head': build_head(cfg)
        }

    @classmethod
    def add_config(cls, cfg, tmp_cfg):
        super().add_config(cfg, tmp_cfg)

    def forward(self, batched_inputs):
        out = self.stem(batched_inputs)
        batched_inputs.update(out)

        out = self.backbone(batched_inputs)
        batched_inputs.update(out)

        out = self.pooler(batched_inputs)
        batched_inputs.update(out)

        out = self.head(batched_inputs)
        return out