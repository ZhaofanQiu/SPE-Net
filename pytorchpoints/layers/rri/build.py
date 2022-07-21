import torch
from pytorchpoints.utils.registry import Registry

RRI_REGISTRY = Registry("RRI")  # noqa F401 isort:skip
RRI_REGISTRY.__doc__ = """
Registry for RRI
"""

def build_rri(cfg, nsample):
    rri = RRI_REGISTRY.get(cfg.MODEL.RRI.NAME)(cfg, nsample)
    rri.to(torch.device(cfg.MODEL.DEVICE))
    return rri

def add_rri_config(cfg, tmp_cfg):
    RRI_REGISTRY.get(tmp_cfg.MODEL.RRI.NAME).add_config(cfg)