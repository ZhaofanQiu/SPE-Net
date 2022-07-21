import torch
from pytorchpoints.utils.registry import Registry

DOWN_SAMPLER_REGISTRY = Registry("DOWN_SAMPLER")  # noqa F401 isort:skip
DOWN_SAMPLER_REGISTRY.__doc__ = """
Registry for down_sampler
"""

def build_downsampler(name, npoint, radius, nsample, sampleDl, device):
    downsampler = DOWN_SAMPLER_REGISTRY.get(name)(npoint, radius, nsample, sampleDl)
    downsampler.to(torch.device(device))
    return downsampler

def add_downsampler_config(cfg, name):
    DOWN_SAMPLER_REGISTRY.get(name).add_config(cfg)