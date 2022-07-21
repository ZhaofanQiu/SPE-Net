import torch
from pytorchpoints.utils.registry import Registry

META_ARCH_REGISTRY = Registry("META_ARCH")  # noqa F401 isort:skip
META_ARCH_REGISTRY.__doc__ = """
Registry for meta-architectures, i.e. the whole model
"""

def build_model(cfg):
    model = META_ARCH_REGISTRY.get(cfg.MODEL.META_ARCHITECTURE)(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    return model

def add_model_config(cfg, tmp_cfg):
    META_ARCH_REGISTRY.get(tmp_cfg.MODEL.META_ARCHITECTURE).add_config(cfg, tmp_cfg)