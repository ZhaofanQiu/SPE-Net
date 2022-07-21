import torch
from pytorchpoints.utils.registry import Registry

HEADS_REGISTRY = Registry("HEADS")  # noqa F401 isort:skip
HEADS_REGISTRY.__doc__ = """
Registry for heads
"""

def build_head(cfg):
    head = HEADS_REGISTRY.get(cfg.MODEL.HEAD.NAME)(cfg)
    head.to(torch.device(cfg.MODEL.DEVICE))
    return head

def add_head_config(cfg, tmp_cfg):
    HEADS_REGISTRY.get(tmp_cfg.MODEL.HEAD.NAME).add_config(cfg)