import torch
from torch import nn

from pytorchpoints.config import kfg
from pytorchpoints.config import configurable
from .build import HEADS_REGISTRY

__all__ = ["ClsHead"]

@HEADS_REGISTRY.register()
class ClsHead(nn.Module):
    @configurable
    def __init__(self, num_classes, head_in_dim):
        super(ClsHead, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(head_in_dim, head_in_dim // 2),
            nn.BatchNorm1d(head_in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(head_in_dim // 2, head_in_dim // 4),
            nn.BatchNorm1d(head_in_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(head_in_dim // 4, head_in_dim // 8),
            nn.BatchNorm1d(head_in_dim // 8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(head_in_dim // 8, num_classes))

    @classmethod
    def from_config(cls, cfg):
        return {
            "num_classes": cfg.MODEL.NUM_CLASSES,
            "head_in_dim": cfg.MODEL.BACKBONE.DIMS[-1]
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs):
        feats = batched_inputs[kfg.FEATS]
        logits = self.classifier(feats)
        if self.training:
            return logits
        else:
            porbs = logits.softmax(dim=-1)
            return porbs