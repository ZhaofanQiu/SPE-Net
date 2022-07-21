import torch
import torch.nn.functional as F
from torch import nn


from pytorchpoints.config import kfg
from pytorchpoints.config import configurable
from .build import RRI_REGISTRY
from pytorchpoints.functional.pt_custom_ops.pt_utils import MaskedQueryAndGroup

__all__ = ["ClusterNetRRI"]

@RRI_REGISTRY.register()
class ClusterNetRRI(nn.Module):
    @configurable
    def __init__(self, nsample, dims, norm_last, act_last, reduction):
        super(ClusterNetRRI, self).__init__()
        self.grouper = MaskedQueryAndGroup(9999, nsample, use_xyz=False, ret_grouped_xyz=False, normalize_xyz=False)
        self.mask_val = 10000
        self.rri_init_dim = 4

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
        # xyz: (B, C, npoint, knn_size)
        B, N, K = xyz.shape[:-1]
        P_all = xyz.reshape(B*N, K, 3)   # B*N, K, 3
        r_all = torch.norm(P_all, dim=-1) # B*N, K
        P_all = P_all / (r_all.unsqueeze(-1) + eps)
        
        P = P_all[:, :1].contiguous()
        Pi = P_all[:, 1:].contiguous()
    
        cos_theta = (P * Pi).sum(-1) # B*N, K-1
        cos_theta = torch.clip(cos_theta, min=-1, max=1)
        theta = torch.arccos(cos_theta) # B*N, K-1
        
        T_pi = Pi - cos_theta.unsqueeze(-1) * P  # B*N, K-1, 3 (check = (P * T_pi).sum(-1))
        T_pi = F.normalize(T_pi, dim=-1)

        cross_T_pik_T_pij = torch.cross( \
            T_pi.unsqueeze(-2).expand(B*N, K-1, K-1, 3), \
            T_pi.unsqueeze(1).expand(B*N, K-1, K-1, 3), dim=-1)
        sin_phi = (cross_T_pik_T_pij * P.unsqueeze(1)).sum(-1)     # B*N, K-1, K-1
        cos_phi = (T_pi.unsqueeze(-2) * T_pi.unsqueeze(1)).sum(-1)  # B*N, K-1, K-1
    
        sin_phi = torch.clip(sin_phi, min=-1, max=1)
        cos_phi = torch.clip(cos_phi, min=-1, max=1)

        phi = torch.atan2(sin_phi, cos_phi)   # B*N, K-1, K-1
        I = torch.eye(K-1, device=phi.device)
        phi = phi + I.unsqueeze(0) * self.mask_val

        #mask = knn_mask.view(B*N, K)[:, 1:].contiguous()
        #ignore = mask.sum(-1) == 0
        #mask = mask + ignore.unsqueeze(-1)
        #phi = phi + (1-mask).unsqueeze(1) * self.mask_val
        phi = torch.min(phi, dim=-1)[0]       # B*N, K-1
    
        r = r_all[:, :1].expand(B*N, K-1)     # B*N, K-1
        rri = torch.stack([r, r_all[:, 1:], theta, phi], dim=-1)
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