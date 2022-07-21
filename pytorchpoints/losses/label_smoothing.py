import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorchpoints.config import configurable
from pytorchpoints.config import kfg
from .build import LOSSES_REGISTRY

@LOSSES_REGISTRY.register()
class LabelSmoothing(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        smoothing_ratio
    ):
        super(LabelSmoothing, self).__init__()
        self.smoothing_ratio = smoothing_ratio

    @classmethod
    def from_config(cls, cfg):
        return {
            "smoothing_ratio": cfg.LOSSES.LABELSMOOTHING
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs):
        logits = batched_inputs[kfg.LOGITS]
        target = batched_inputs[kfg.LABELS]
        return self._forward(logits, target)

    def _forward(self, pred, target):
        eps = self.smoothing_ratio
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
        return { "label_smoothing_loss": loss }