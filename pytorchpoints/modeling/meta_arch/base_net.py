import torch
from torch import nn
from abc import ABCMeta, abstractmethod

from ..stem import add_stem_config
from ..backbones import add_backbone_config
from ..global_pooler import add_pooler_config
from ..heads import add_head_config
from ..local_aggregation import add_local_aggregation_config
from ..down_sampler import add_downsampler_config
from pytorchpoints.config import kfg
from pytorchpoints.functional import dict_to_cuda

class BaseNet(nn.Module, metaclass=ABCMeta):
    def __init__(self):
        super(BaseNet, self).__init__()

    @classmethod
    def add_config(cls, cfg, tmp_cfg):
        add_stem_config(cfg, tmp_cfg)
        add_backbone_config(cfg, tmp_cfg)
        add_pooler_config(cfg, tmp_cfg)
        add_head_config(cfg, tmp_cfg)
        add_local_aggregation_config(cfg, tmp_cfg.MODEL.LA_TYPE.NAME)
        add_downsampler_config(cfg, tmp_cfg.MODEL.DOWN_SAMPLER.NAME)

    @abstractmethod
    def forward(self, batched_inputs):
        pass

    def preprocess_batch(self, batched_inputs):
        ret = {}
        keys = [kfg.POINTS, kfg.NORMALS, kfg.LABELS, kfg.MASKS, kfg.FEATS, kfg.POINTS_LABELS, kfg.CLOUD_INDEX, kfg.INPUT_INDEX]
        if 'aug1' in batched_inputs[0]: # batch augmentation
            for k in keys:
                if k in batched_inputs[0]['aug1']:
                    ent1 = [x['aug1'][k] for x in batched_inputs]
                    ent2 = [x['aug2'][k] for x in batched_inputs]
                    ent = torch.stack(ent1 + ent2, dim=0)
                    ret.update({k: ent})
        else:
            for k in keys:
                if k in batched_inputs[0]:
                    ent = [x[k] for x in batched_inputs]
                    ent = torch.stack(ent, dim=0)
                    ret.update({k: ent})

        dict_to_cuda(ret)
        return ret
