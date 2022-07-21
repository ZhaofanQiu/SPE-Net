import torch

from pytorchpoints.utils.registry import Registry


EVALUATION_REGISTRY = Registry("EVALUATION")
EVALUATION_REGISTRY.__doc__ = """
Registry for evaluation
"""

def build_evaluation(cfg):
    evaluation = EVALUATION_REGISTRY.get(cfg.INFERENCE.NAME)(cfg) if len(cfg.INFERENCE.NAME) > 0 else None
    return evaluation