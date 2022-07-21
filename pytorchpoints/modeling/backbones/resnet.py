import torch
from torch import nn

from pytorchpoints.config import kfg
from pytorchpoints.config import configurable
from pytorchpoints.layers import DropPath
from .build import BACKBONES_REGISTRY
from ..local_aggregation import build_local_aggregation
from ..down_sampler import build_downsampler
from functools import partial

__all__ = ["ResNet"]

class Bottleneck(nn.Module):
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        bottleneck_ratio,
        la,
        downsample,
        drop_path
    ):
        super(Bottleneck, self).__init__()
        self.downsample = downsample
        self.local_aggregation = la
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels // bottleneck_ratio, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels // bottleneck_ratio),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channels // bottleneck_ratio, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels))
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)) if in_channels != out_channels else nn.Identity()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, xyz, mask, features):
        if self.downsample is not None:
            sub_xyz, sub_mask, sub_features = self.downsample(xyz, mask, features)
            query_xyz = sub_xyz
            query_mask = sub_mask
            identity = sub_features
        else:
            query_xyz = xyz
            query_mask = mask
            identity = features

        output = self.conv1(features)
        output = self.local_aggregation(query_xyz, xyz, query_mask, mask, output)
        output = self.conv2(output)
        output = self.drop_path(output)
        output += self.shortcut(identity)
        output = self.relu(output)
        return query_xyz, query_mask, output

@BACKBONES_REGISTRY.register()
class ResNet(nn.Module):
    @configurable
    def __init__(self, 
        stem_width,
        dims,
        btnk_ratios,
        layers,
        npoints,
        nsamples,
        radius, 
        sampleDl, 
        la_fn,        
        down_sampler,
        drop_path,
        device
    ):
        super(ResNet, self).__init__()
        stages = []
        last_dim = stem_width

        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(layers))]
        dpr_idx = 0
        for stage_idx, (dim, btnk_ratio, num_blocks, num_pts) in enumerate(zip(dims, btnk_ratios, layers, npoints)):
            stage_name = f'layer{stage_idx + 1}'
            blocks = []
            for block_idx in range(num_blocks):
                downsample = None
                if block_idx == 0:
                    nsample = nsamples[stage_idx-1] if stage_idx > 0 else nsamples[0]
                    if num_pts > 0:
                        sampleDl *= 2
                        downsample = build_downsampler(down_sampler, num_pts, radius, nsample, sampleDl, device)
                else:
                    nsample = nsamples[stage_idx]
                    if num_pts > 0:
                        radius *= 2
                
                la = la_fn(dim // btnk_ratio, dim // btnk_ratio, radius, nsample, device)
                blocks.append(Bottleneck(last_dim, dim, btnk_ratio, la, downsample, dpr[dpr_idx]))
                last_dim = dim
                dpr_idx += 1
            stages.append((stage_name, nn.ModuleList(blocks)))

        for stage in stages:
            self.add_module(*stage)
        self.num_stages = len(layers)

    @classmethod
    def from_config(cls, cfg):
        la_fn = partial(build_local_aggregation, cfg, cfg.MODEL.LA_TYPE.NAME)
        return {
            "stem_width": cfg.MODEL.STEM.WIDTH,
            "dims": cfg.MODEL.BACKBONE.DIMS,
            "btnk_ratios": cfg.MODEL.BACKBONE.BTNK_RATIOS,
            "layers": cfg.MODEL.BACKBONE.LAYERS,
            "npoints": cfg.MODEL.BACKBONE.NPOINTS,
            "nsamples": cfg.MODEL.BACKBONE.NSAMPLES,
            "radius": cfg.MODEL.BACKBONE.RADIUS,
            "sampleDl": cfg.MODEL.BACKBONE.SAMPLEDL,
            "la_fn": la_fn,     
            "down_sampler": cfg.MODEL.DOWN_SAMPLER.NAME,
            "drop_path": cfg.MODEL.DROP_PATH,
            "device": cfg.MODEL.DEVICE
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, batched_inputs, seg=False):
        xyz = batched_inputs[kfg.POINTS]
        mask = batched_inputs[kfg.MASKS]
        features = batched_inputs[kfg.FEATS]

        ret = {}
        for stage_idx in range(self.num_stages):
            blocks = getattr(self, f"layer{stage_idx + 1}")
            for block in blocks:
                xyz, mask, features = block(xyz, mask, features)
            if seg:
                ret.update({
                    "{}_{}".format(stage_idx+1, kfg.POINTS): xyz,
                    "{}_{}".format(stage_idx+1, kfg.MASKS): mask,
                    "{}_{}".format(stage_idx+1, kfg.FEATS): features,
                })

        if not seg:
            ret = { kfg.POINTS: xyz, kfg.MASKS: mask, kfg.FEATS: features }
        return ret