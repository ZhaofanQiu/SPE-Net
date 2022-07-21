import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorchpoints.config import configurable
from pytorchpoints.config import kfg
from pytorchpoints.config import CfgNode as CN
from .build import LOSSES_REGISTRY

@LOSSES_REGISTRY.register()
class MultiShapeCrossEntropy(nn.Module):
    @configurable
    def __init__(self, num_classes):
        super(MultiShapeCrossEntropy, self).__init__()
        self.num_classes = num_classes

    @classmethod
    def from_config(cls, cfg):
        return { "num_classes": cfg.MODEL.NUM_CLASSES }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs):
        logits = batched_inputs[kfg.LOGITS]
        shape_labels = batched_inputs[kfg.LABELS]
        points_labels = batched_inputs[kfg.POINTS_LABELS]
        return self._forward(logits, points_labels, shape_labels)

    def _forward(self, logits_all_shapes, points_labels, shape_labels):
        batch_size = shape_labels.shape[0]
        losses = 0
        for i in range(batch_size):
            sl = shape_labels[i]
            logits = torch.unsqueeze(logits_all_shapes[sl][i], 0)
            pl = torch.unsqueeze(points_labels[i], 0)
            loss = F.cross_entropy(logits, pl)
            losses += loss
            for isl in range(self.num_classes):
                if isl == sl:
                    continue
                losses += 0.0 * logits_all_shapes[isl][i].sum()
        loss = losses / batch_size
        return { "multi_shape_cross_entropy_loss": loss }