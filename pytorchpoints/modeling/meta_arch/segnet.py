import torch
from torch import nn

from .base_net import BaseNet
from pytorchpoints.config import configurable
from .build import META_ARCH_REGISTRY

from ..stem import build_stem
from ..backbones import build_backbone
from ..heads import build_head

__all__ = ["SegNet"]

@META_ARCH_REGISTRY.register()
class SegNet(BaseNet):
    @configurable
    def __init__(self, stem, backbone, head):
        super(SegNet, self).__init__()
        self.stem = stem
        self.backbone = backbone
        self.head = head

    @classmethod
    def from_config(cls, cfg):
        return {
            'stem': build_stem(cfg),
            'backbone': build_backbone(cfg),
            'head': build_head(cfg)
        }

    @classmethod
    def add_config(cls, cfg, tmp_cfg):
        super().add_config(cfg, tmp_cfg)

    def forward(self, batched_inputs):
        out = self.stem(batched_inputs)
        batched_inputs.update(out)

        out = self.backbone(batched_inputs, seg=True)
        batched_inputs.update(out)

        out = self.head(batched_inputs)
        return out