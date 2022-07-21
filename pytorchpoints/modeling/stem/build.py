import torch
from pytorchpoints.utils.registry import Registry

STEM_REGISTRY = Registry("STEM")
STEM_REGISTRY.__doc__ = """
Registry for stem
"""

def build_stem(cfg):
    stem = STEM_REGISTRY.get(cfg.MODEL.STEM.NAME)(cfg)
    stem.to(torch.device(cfg.MODEL.DEVICE))
    return stem

def add_stem_config(cfg, tmp_cfg):
    STEM_REGISTRY.get(tmp_cfg.MODEL.STEM.NAME).add_config(cfg)