import torch
from pytorchpoints.utils.registry import Registry

GLOBAL_POOLER_REGISTRY = Registry("GLOBAL_POOLER")  # noqa F401 isort:skip
GLOBAL_POOLER_REGISTRY.__doc__ = """
Registry for global-pooler
"""

def build_pooler(cfg):
    pooler = GLOBAL_POOLER_REGISTRY.get(cfg.MODEL.GLOBAL_POOLER.NAME)(cfg)
    pooler.to(torch.device(cfg.MODEL.DEVICE))
    return pooler

def add_pooler_config(cfg, tmp_cfg):
    GLOBAL_POOLER_REGISTRY.get(tmp_cfg.MODEL.GLOBAL_POOLER.NAME).add_config(cfg)