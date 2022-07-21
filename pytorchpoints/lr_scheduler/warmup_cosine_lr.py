import torch
from pytorchpoints.config import configurable
from .build import LR_SCHEDULER_REGISTRY

# noinspection PyProtectedMember
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR


# noinspection PyAttributeOutsideInit
class _WarmupCosineLR(_LRScheduler):

    def __init__(self, optimizer, epoch, warmup_epoch, multiplier=100, last_epoch=-1):
        self.multiplier = multiplier
        if self.multiplier <= 1.:
            raise ValueError('multiplier should be greater than 1.')
        self.epoch = epoch
        self.warmup_epoch = warmup_epoch
        self.after_scheduler = CosineAnnealingLR(
            optimizer=optimizer,
            eta_min=0.000001,
            T_max=epoch - warmup_epoch)
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch > self.warmup_epoch:
            return self.after_scheduler.get_lr()
        else:
            return [base_lr / self.multiplier * ((self.multiplier - 1.) * self.last_epoch / self.warmup_epoch + 1.)
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch
        self.last_epoch = epoch
        if epoch > self.warmup_epoch:
            self.after_scheduler.step()
        else:
            super(_WarmupCosineLR, self).step()

    def state_dict(self):
        state = {key: value for key, value in self.__dict__.items() if key != 'optimizer' and key != 'after_scheduler'}
        state['after_scheduler'] = self.after_scheduler.state_dict()
        return state

    def load_state_dict(self, state_dict):
        after_scheduler_state = state_dict.pop('after_scheduler')
        self.__dict__.update(state_dict)
        self.after_scheduler.load_state_dict(after_scheduler_state)


@LR_SCHEDULER_REGISTRY.register()
class WarmupCosineLR(_WarmupCosineLR):
    @configurable
    def __init__(
        self, 
        *,
        optimizer, 
        epoch,
        warmup_epoch
    ):
        super(WarmupCosineLR, self).__init__(
            optimizer=optimizer,
            epoch=epoch,
            warmup_epoch=warmup_epoch,
        )

    @classmethod
    def from_config(cls, cfg, optimizer, data_size):
        return {
            "optimizer": optimizer,
            "epoch": cfg.SOLVER.EPOCH * data_size,
            "warmup_epoch": cfg.LR_SCHEDULER.WARMUP_EPOCH * data_size
        }
