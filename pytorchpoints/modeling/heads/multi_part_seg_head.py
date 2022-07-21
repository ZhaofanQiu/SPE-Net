import torch
from torch import nn

from pytorchpoints.config import kfg
from pytorchpoints.config import configurable
from pytorchpoints.functional.pt_custom_ops.pt_utils import MaskedUpsample
from .build import HEADS_REGISTRY

__all__ = ["MultiPartSegHead"]

@HEADS_REGISTRY.register()
class MultiPartSegHead(nn.Module):
    @configurable
    def __init__(self, num_classes, width, base_radius, nsamples, num_parts):
        super(MultiPartSegHead, self).__init__()
        self.num_classes = num_classes
        self.base_radius = base_radius
        self.nsamples = nsamples
        self.num_parts = num_parts
        self.up0 = MaskedUpsample(radius=8 * base_radius, nsample=nsamples[3], mode='nearest')
        self.up1 = MaskedUpsample(radius=4 * base_radius, nsample=nsamples[2], mode='nearest')
        self.up2 = MaskedUpsample(radius=2 * base_radius, nsample=nsamples[1], mode='nearest')
        self.up3 = MaskedUpsample(radius=base_radius, nsample=nsamples[0], mode='nearest')

        self.up_conv0 = nn.Sequential(nn.Conv1d(24 * width, 4 * width, kernel_size=1, bias=False),
                                      nn.BatchNorm1d(4 * width),
                                      nn.ReLU(inplace=True))
        self.up_conv1 = nn.Sequential(nn.Conv1d(8 * width, 2 * width, kernel_size=1, bias=False),
                                      nn.BatchNorm1d(2 * width),
                                      nn.ReLU(inplace=True))
        self.up_conv2 = nn.Sequential(nn.Conv1d(4 * width, width, kernel_size=1, bias=False),
                                      nn.BatchNorm1d(width),
                                      nn.ReLU(inplace=True))
        self.up_conv3 = nn.Sequential(nn.Conv1d(2 * width, width // 2, kernel_size=1, bias=False),
                                      nn.BatchNorm1d(width // 2),
                                      nn.ReLU(inplace=True))
        self.multi_shape_heads = nn.ModuleList()
        for i in range(num_classes):
            self.multi_shape_heads.append(
                nn.Sequential(nn.Conv1d(width // 2, width // 2, kernel_size=1, bias=False),
                              nn.BatchNorm1d(width // 2),
                              nn.ReLU(inplace=True),
                              nn.Conv1d(width // 2, num_parts[i], kernel_size=1, bias=True)))

    @classmethod
    def from_config(cls, cfg):
        return {
            "num_classes": cfg.MODEL.NUM_CLASSES,
            "width": cfg.MODEL.BACKBONE.DIMS[0],
            "base_radius": cfg.MODEL.BACKBONE.RADIUS,
            "nsamples": cfg.MODEL.BACKBONE.NSAMPLES,
            "num_parts": cfg.DATALOADER.NUM_PARTS
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs):
        features = self.up0(batched_inputs['4_' + kfg.POINTS], batched_inputs['5_' + kfg.POINTS],
                            batched_inputs['4_' + kfg.MASKS], batched_inputs['5_' + kfg.MASKS], batched_inputs['5_' + kfg.FEATS])
        features = torch.cat([features, batched_inputs['4_' + kfg.FEATS]], 1)
        features = self.up_conv0(features)

        features = self.up1(batched_inputs['3_' + kfg.POINTS], batched_inputs['4_' + kfg.POINTS],
                            batched_inputs['3_' + kfg.MASKS], batched_inputs['4_' + kfg.MASKS], features)
        features = torch.cat([features, batched_inputs['3_' + kfg.FEATS]], 1)
        features = self.up_conv1(features)

        features = self.up2(batched_inputs['2_' + kfg.POINTS], batched_inputs['3_' + kfg.POINTS],
                            batched_inputs['2_' + kfg.MASKS], batched_inputs['3_' + kfg.MASKS], features)
        features = torch.cat([features, batched_inputs['2_' + kfg.FEATS]], 1)
        features = self.up_conv2(features)

        features = self.up3(batched_inputs['1_' + kfg.POINTS], batched_inputs['2_' + kfg.POINTS],
                            batched_inputs['1_' + kfg.MASKS], batched_inputs['2_' + kfg.MASKS], features)
        features = torch.cat([features, batched_inputs['1_' + kfg.FEATS]], 1)
        features = self.up_conv3(features)

        logits_all_shapes = []
        for i in range(self.num_classes):
            logits = self.multi_shape_heads[i](features)
            if not self.training:
                logits = logits.softmax(dim=1)
            logits_all_shapes.append(logits)

        return logits_all_shapes