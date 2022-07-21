import torch
import torch.nn.functional as F
from torch import nn


from pytorchpoints.config import kfg
from pytorchpoints.config import configurable
from .build import RRI_REGISTRY
from pytorchpoints.functional.pt_custom_ops.pt_utils import MaskedQueryAndGroup

__all__ = ["DistNetRRI"]

@RRI_REGISTRY.register()
class DistNetRRI(nn.Module):
    @configurable
    def __init__(self, nsample, dims, norm_last, act_last, reduction):
        super(DistNetRRI, self).__init__()
        self.grouper = MaskedQueryAndGroup(9999, nsample, use_xyz=False, ret_grouped_xyz=False, normalize_xyz=False)
        self.mask_val = 10000
        self.rri_init_dim = 6

        rri_mlps = []
        last_dim = self.rri_init_dim
        for i, dim in enumerate(dims):
            rri_mlps.append(nn.Conv2d(last_dim, dim, kernel_size=1, padding=0, bias=False))
            if (i != len(dims) - 1) or norm_last:
                rri_mlps.append(nn.BatchNorm2d(dim))
            if (i != len(dims) - 1) or act_last:
                rri_mlps.append(nn.ReLU(inplace=True))
            last_dim = dim
        self.rri_mlps = nn.Sequential(*rri_mlps)
        self.reduction = reduction

        if self.reduction == 'att':
            self.gate = nn.Conv2d(last_dim, 1, kernel_size=1, padding=0)

    @classmethod
    def from_config(cls, cfg, nsample):
        return {
            "nsample": nsample,
            "dims": cfg.MODEL.RRI.DIMS,
            "norm_last": cfg.MODEL.RRI.NORM_LAST,
            "act_last": cfg.MODEL.RRI.ACT_LAST,
            "reduction": cfg.MODEL.RRI.REDUCTION
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def xyz_to_rri(self, xyz, knn_mask, eps=1e-8):
        B, N, K = xyz.shape[:-1]
        xyz_mean = xyz.mean(-2, keepdim=True)
        xyz = torch.cat([xyz_mean, xyz], dim=2)

        P_all = xyz.reshape(B*N, K+1, 3)   # B*N, K+1, 3
        r_all = torch.norm(P_all, dim=-1) # B*N, K+1

        P = P_all[:, 1:2].contiguous()
        Pm = P_all[:, 0:1].contiguous()
        Pi = P_all[:, 2:].contiguous()

        dist_P_Pi = torch.norm(P - Pi, dim=-1)
        dist_Pm_Pi = torch.norm(Pi - Pm, dim=-1)
        dist_P_Pm = torch.norm(P - Pm, dim=-1)
        dist_P_Pm = dist_P_Pm.expand(B*N, K-1)

        r = r_all[:, 1:2].expand(B*N, K-1)     # B*N, K-1
        rm = r_all[:, 0:1].expand(B*N, K-1)     # B*N, K-1
        ri = r_all[:, 2:]
        
        rri = torch.stack([r, rm, ri, dist_P_Pi, dist_Pm_Pi, dist_P_Pm], dim=-1)
        rri = rri.reshape(B, N, K-1, self.rri_init_dim).permute(0, 3, 1, 2).contiguous()
        return rri 
    
    def forward(self, xyz, mask):
        knn_xyz, knn_mask = self.grouper(xyz, xyz, mask, mask, xyz.transpose(-1,-2).contiguous())
        knn_xyz = knn_xyz.permute(0, 2, 3, 1)

        with torch.no_grad():
            rri = self.xyz_to_rri(knn_xyz, knn_mask)
        rri = self.rri_mlps(rri)

        if self.reduction == 'max':
            #ignore = knn_mask[:, :, 1:].sum(-1) == 0
            #mask = knn_mask[:, :, 1:] + ignore.unsqueeze(-1)
            #rri = rri + (1- mask.unsqueeze(1)) * self.mask_val * -1.0
            rri = F.max_pool2d(rri, kernel_size=[1, rri.shape[-1]]).squeeze(-1)
        elif self.reduction == 'att':
            attn = self.gate(rri)
            attn = attn.softmax(-1)
            rri = (rri * attn).sum(-1)
        else:
            raise NotImplementedError(f'Reduction {self.reduction} not implemented in ClusterNetRRI')
        return rri