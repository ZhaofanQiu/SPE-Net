import torch
from pytorchpoints.utils.registry import Registry

LOCAL_AGGREGATION_REGISTRY = Registry("LOCAL_AGGREGATION")  # noqa F401 isort:skip
LOCAL_AGGREGATION_REGISTRY.__doc__ = """
Registry for local aggregation
"""

def build_local_aggregation(cfg, name, in_channels, out_channels, radius, nsample, device):
    local_aggregation = LOCAL_AGGREGATION_REGISTRY.get(name)(cfg, in_channels, out_channels, radius, nsample)
    local_aggregation.to(torch.device(device))
    return local_aggregation

def add_local_aggregation_config(cfg, name):
    LOCAL_AGGREGATION_REGISTRY.get(name).add_config(cfg)