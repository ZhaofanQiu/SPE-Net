import torch
from torch import nn
from pytorchpoints.config import kfg
from pytorchpoints.config import configurable
from .build import DOWN_SAMPLER_REGISTRY
from pytorchpoints.functional.pt_custom_ops.pt_utils import MaskedMaxPool

__all__ = ["MaxPoolDS"]

@DOWN_SAMPLER_REGISTRY.register()
class MaxPoolDS(nn.Module):
    @configurable
    def __init__(self, npoint, radius, nsample, sampleDl):
        super(MaxPoolDS, self).__init__()
        self.maxpool = MaskedMaxPool(npoint, radius, nsample, sampleDl)

    @classmethod
    def from_config(cls, cfg, npoint, radius, nsample, sampleDl):
        return {
            "npoint": npoint,
            "radius": radius,
            "nsample": nsample,
            "sampleDl": sampleDl,
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, xyz, mask, features):
        sub_xyz, sub_mask, sub_features = self.maxpool(xyz, mask, features)
        return sub_xyz, sub_mask, sub_features