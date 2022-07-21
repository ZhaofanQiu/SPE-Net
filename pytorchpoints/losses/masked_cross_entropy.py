import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorchpoints.config import configurable
from pytorchpoints.config import kfg
from pytorchpoints.config import CfgNode as CN
from .build import LOSSES_REGISTRY

@LOSSES_REGISTRY.register()
class MaskedCrossEntropy(nn.Module):
    @configurable
    def __init__(self):
        super(MaskedCrossEntropy, self).__init__()

    @classmethod
    def from_config(cls, cfg):
        return {}

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs):
        logits = batched_inputs[kfg.LOGITS]
        mask = batched_inputs[kfg.MASKS]
        points_labels = batched_inputs[kfg.POINTS_LABELS]
        return self._forward(logits, points_labels, mask)

    def _forward(self, logit, target, mask):
        loss = F.cross_entropy(logit, target, reduction='none')
        loss *= mask
        loss = loss.sum() / mask.sum()
        return { "masked_cross_entropy_loss": loss }