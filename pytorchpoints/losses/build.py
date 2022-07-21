from pytorchpoints.utils.registry import Registry

LOSSES_REGISTRY = Registry("LOSSES")
LOSSES_REGISTRY.__doc__ = """
Registry for losses
"""

def build_losses(cfg):
    losses = []
    for name in cfg.LOSSES.NAMES:
        loss = LOSSES_REGISTRY.get(name)(cfg)
        losses.append(loss)
    return losses

def add_loss_config(cfg, tmp_cfg):
    for name in tmp_cfg.LOSSES.NAMES:
        LOSSES_REGISTRY.get(name).add_config(cfg)