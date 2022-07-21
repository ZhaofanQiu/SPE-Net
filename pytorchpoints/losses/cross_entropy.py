import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorchpoints.config import configurable
from pytorchpoints.config import kfg
from pytorchpoints.config import CfgNode as CN
from .build import LOSSES_REGISTRY


@LOSSES_REGISTRY.register()
class CrossEntropy(nn.Module):
    @configurable
    def __init__(self):
        super(CrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    @classmethod
    def from_config(cls, cfg):
        return {}

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs):
        logits = batched_inputs[kfg.LOGITS]
        target = batched_inputs[kfg.LABELS]
        return self._forward(logits, target)

    def _forward(self, pred, target):
        loss = self.criterion(pred, target)
        return { "cross_entropy_loss": loss }