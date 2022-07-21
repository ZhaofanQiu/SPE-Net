import torch
from pytorchpoints.utils.registry import Registry

BACKBONES_REGISTRY = Registry("BACKBONES")  # noqa F401 isort:skip
BACKBONES_REGISTRY.__doc__ = """
Registry for backbones
"""

def build_backbone(cfg):
    backbone = BACKBONES_REGISTRY.get(cfg.MODEL.BACKBONE.NAME)(cfg)
    backbone.to(torch.device(cfg.MODEL.DEVICE))
    return backbone

def add_backbone_config(cfg, tmp_cfg):
    BACKBONES_REGISTRY.get(tmp_cfg.MODEL.BACKBONE.NAME).add_config(cfg)