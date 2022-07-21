import torch
import torch.nn.functional as F
from torch import nn


from pytorchpoints.config import kfg
from pytorchpoints.config import configurable
from .build import RRI_REGISTRY
from pytorchpoints.functional.pt_custom_ops.pt_utils import MaskedQueryAndGroup

__all__ = ["RotInvRRI"]

def get_plane_equation(p1, p2, p3, eps):
    # p1 (B, N, K, 3)
    # return: (B, N, K, 4)
    p1p2 = p2-p1
    p2p3 = p3-p2
    normal = torch.cross(p1p2, p2p3)
    normal_length = torch.norm(normal, dim=-1, keepdim=True)  # [B, N, K, 1]
    normal = normal / (normal_length + eps)
    plane_a = normal[:, :, :, 0]
    plane_b = normal[:, :, :, 1]
    plane_c = normal[:, :, :, 2]

    x1 = p1[:, :, :, 0]
    y1 = p1[:, :, :, 1]
    z1 = p1[:, :, :, 2]
    plane_d = -1.0*(plane_a*x1+plane_b*y1+plane_c*z1)

    plane_params = torch.concat([
        plane_a.unsqueeze(-1),
        plane_b.unsqueeze(-1),
        plane_c.unsqueeze(-1),
        plane_d.unsqueeze(-1),
    ], dim=-1)
    return plane_params

@RRI_REGISTRY.register()
class RotInvRRI(nn.Module):
    @configurable
    def __init__(self, nsample, radius, dims, norm_last, act_last, reduction):
        super(RotInvRRI, self).__init__()
        self.grouper = MaskedQueryAndGroup(radius, nsample, use_xyz=False, ret_grouped_xyz=False, normalize_xyz=False)
        self.mask_val = 10000
        self.rri_init_dim = 12
        self.radius = radius

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
            "reduction": cfg.MODEL.RRI.REDUCTION,
            "radius": cfg.MODEL.BACKBONE.RADIUS,
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def xyz_to_rri_v0(self, xyz, knn_mask, radius=0.05, eps=1e-8, point_start = 1): # B, N, K, 3
        B, N, K = xyz.shape[:-1]
        diff = xyz.unsqueeze(-2) - xyz.unsqueeze(-3)
        dist = (diff * diff).sum(-1)
        dist = (dist * knn_mask.unsqueeze(-2)).sum(-1) # # B, N, K
        dist = dist + (1-knn_mask)*self.mask_val

        center_idx = torch.min(dist[:,:,point_start:], dim=-1)[1] + point_start
        center_idx = center_idx.unsqueeze(-1).expand(B, N, 3).unsqueeze(-2)
        
        centroid_xyz = torch.gather(xyz, 2, center_idx).view(B, N, 3)
        new_xyz = xyz[:, :, 0].contiguous()
        grouped_xyz = xyz[:,:,1:].contiguous()

        reference_vector_norm = torch.norm(new_xyz, dim=-1, keepdim=True)
        reference_vector_unit = new_xyz / (reference_vector_norm + eps)
        inter_xyz = radius * reference_vector_unit + new_xyz

        centroid_reference_vector = new_xyz - centroid_xyz
        centroid_reference_dist = torch.norm(centroid_reference_vector, dim=-1, keepdim=True)

        centroid_inter_vector = inter_xyz - centroid_xyz
        centroid_inter_dist = torch.norm(centroid_inter_vector, dim=-1, keepdim=True)

        inter_reference_vector = new_xyz - inter_xyz
        inter_reference_dist = torch.norm(inter_reference_vector, dim=-1, keepdim=True)

        center_point_features = torch.concat([
            reference_vector_norm, 
            centroid_reference_dist, 
            centroid_inter_dist,
            inter_reference_dist
        ], dim=-1)  # [B, N, 4]
        center_point_features = center_point_features.unsqueeze(-2).expand(B, N, K-1, 4)

        centroid_xyz_tile = centroid_xyz.unsqueeze(-2).expand(B, N, K-1, 3)
        neighbor_centroid_vector = centroid_xyz_tile - grouped_xyz
        reference_vector_tile = new_xyz.unsqueeze(-2).expand(B, N, K-1, 3)
        neighbor_reference_vector = reference_vector_tile - grouped_xyz
        inter_pts = inter_xyz.unsqueeze(-2).expand(B, N, K-1, 3)
        neighbor_inter_vector = inter_pts - grouped_xyz

        neighbor_centroid_dist = torch.norm(neighbor_centroid_vector, dim=-1, keepdim=True)
        neighbor_reference_dist = torch.norm(neighbor_reference_vector, dim=-1, keepdim=True)
        neighbor_inter_dist = torch.norm(neighbor_inter_vector, dim=-1, keepdim=True)

        neighbor_centroid_vector_norm = neighbor_centroid_vector / (neighbor_centroid_dist+eps)
        neighbor_reference_vector_norm = neighbor_reference_vector / (neighbor_reference_dist+eps)
        neighbor_inter_vector_norm = neighbor_inter_vector / (neighbor_inter_dist+eps)

        centroid_neighbor_reference_angle = (neighbor_centroid_vector_norm * neighbor_reference_vector_norm).sum(-1)
        reference_neighbor_inter_angle  = (neighbor_reference_vector_norm * neighbor_inter_vector_norm).sum(-1)
        inter_neighbor_centroid_angle = (neighbor_inter_vector_norm * neighbor_centroid_vector_norm).sum(-1)
        grouped_xyz_norm = torch.norm(grouped_xyz, dim=-1, keepdim=True)

        neighbor_point_features = torch.concat([
            neighbor_centroid_dist, 
            neighbor_reference_dist, 
            neighbor_inter_dist,
            centroid_neighbor_reference_angle.unsqueeze(-1), 
            reference_neighbor_inter_angle.unsqueeze(-1),
            inter_neighbor_centroid_angle.unsqueeze(-1), 
            grouped_xyz_norm], dim=-1)
        
        rri = torch.cat([center_point_features, neighbor_point_features], dim=-1)
        rri = rri.permute(0, 3, 1, 2).contiguous()
        return rri 

    def xyz_to_rri_v1(self, xyz, knn_mask, radius=0.05, eps=1e-8, point_start = 1): # B, N, K, 3
        B, N, K = xyz.shape[:-1]
        diff = xyz.unsqueeze(-2) - xyz.unsqueeze(-3)
        dist = (diff * diff).sum(-1)
        dist = (dist * knn_mask.unsqueeze(-2)).sum(-1) # # B, N, K
        dist = dist + (1-knn_mask)*self.mask_val

        center_idx = torch.min(dist[:,:,point_start:], dim=-1)[1] + point_start
        center_idx = center_idx.unsqueeze(-1).expand(B, N, 3).unsqueeze(-2)
        
        centroid_xyz = torch.gather(xyz, 2, center_idx).view(B, N, 3)
        new_xyz = xyz[:, :, 0].contiguous()
        grouped_xyz = xyz[:,:,1:].contiguous()

        reference_vector_norm = torch.norm(new_xyz, dim=-1, keepdim=True)
        reference_vector_unit = new_xyz / (reference_vector_norm + eps)
        inter_xyz = radius * reference_vector_unit + new_xyz

        centroid_reference_vector = new_xyz - centroid_xyz
        centroid_reference_dist = torch.norm(centroid_reference_vector, dim=-1, keepdim=True)
        centroid_reference_vector_norm = centroid_reference_vector / (centroid_reference_dist + eps)

        centroid_inter_vector = inter_xyz - centroid_xyz
        centroid_inter_dist = torch.norm(centroid_inter_vector, dim=-1, keepdim=True)
        centroid_inter_vector_norm = centroid_inter_vector / (centroid_inter_dist + eps)
        reference_centroid_inter_angle = (centroid_reference_vector_norm * centroid_inter_vector_norm).sum(-1)

        inter_reference_vector = new_xyz - inter_xyz
        inter_reference_dist = torch.norm(inter_reference_vector, dim=-1, keepdim=True)
        inter_reference_vector_norm = inter_reference_vector / (inter_reference_dist + eps)

        inter_centroid_vector = centroid_xyz - inter_xyz
        inter_centroid_dist = torch.norm(inter_centroid_vector, dim=-1, keepdim=True)
        inter_centroid_vector_norm = inter_centroid_vector / (inter_centroid_dist + eps)

        reference_inter_centroid_angle = (inter_reference_vector_norm * inter_centroid_vector_norm).sum(-1)

        center_point_features = torch.concat([
            reference_vector_norm, 
            centroid_reference_dist, 
            centroid_inter_dist,
            reference_centroid_inter_angle.unsqueeze(-1),
            reference_inter_centroid_angle.unsqueeze(-1)
        ], dim=-1)  # [B, N, 5]
        center_point_features = center_point_features.unsqueeze(-2).expand(B, N, K-1, 5)

        centroid_xyz_tile = centroid_xyz.unsqueeze(-2).expand(B, N, K-1, 3)
        neighbor_centroid_vector = centroid_xyz_tile - grouped_xyz
        reference_vector_tile = new_xyz.unsqueeze(-2).expand(B, N, K-1, 3)
        neighbor_reference_vector = reference_vector_tile - grouped_xyz
        inter_pts = inter_xyz.unsqueeze(-2).expand(B, N, K-1, 3)
        neighbor_inter_vector = inter_pts - grouped_xyz

        neighbor_centroid_dist = torch.norm(neighbor_centroid_vector, dim=-1, keepdim=True)
        neighbor_reference_dist = torch.norm(neighbor_reference_vector, dim=-1, keepdim=True)
        neighbor_inter_dist = torch.norm(neighbor_inter_vector, dim=-1, keepdim=True)

        neighbor_centroid_vector_norm = neighbor_centroid_vector / (neighbor_centroid_dist+eps)
        neighbor_reference_vector_norm = neighbor_reference_vector / (neighbor_reference_dist+eps)
        neighbor_inter_vector_norm = neighbor_inter_vector / (neighbor_inter_dist+eps)

        centroid_neighbor_reference_angle = (neighbor_centroid_vector_norm * neighbor_reference_vector_norm).sum(-1)
        reference_neighbor_inter_angle  = (neighbor_reference_vector_norm * neighbor_inter_vector_norm).sum(-1)
        inter_neighbor_centroid_angle = (neighbor_inter_vector_norm * neighbor_centroid_vector_norm).sum(-1)
        #grouped_xyz_norm = torch.norm(grouped_xyz, dim=-1, keepdim=True)

        #################### calculate angle ####################
        reference_plane_params = get_plane_equation(inter_pts, reference_vector_tile, centroid_xyz_tile, eps)  # [B, N, K, 4]
        reference_normal_vector = reference_plane_params[:, :, :, 0:3]
        neighbor_plane_params = get_plane_equation(inter_pts, reference_vector_tile, grouped_xyz, eps)
        neighbor_normal_vector = neighbor_plane_params[:, :, :, 0:3]
        dot_product = (reference_normal_vector * neighbor_normal_vector).sum(-1)
        #cos_plane_angle = dot_product
        cos_plane_angle = torch.clip(dot_product, min=-1, max=1)

        plane_angle = torch.acos(cos_plane_angle)  # [B, N, K, 1]
        pos_state = (reference_normal_vector * -neighbor_reference_vector).sum(-1)
        pos_state = torch.sign(pos_state)
        plane_angle_direction = plane_angle * pos_state
        angle = torch.cos(0.25*plane_angle_direction) - torch.sin(0.25*plane_angle_direction) - 0.75

        neighbor_point_features = torch.concat([
            neighbor_centroid_dist, 
            neighbor_reference_dist, 
            neighbor_inter_dist,
            centroid_neighbor_reference_angle.unsqueeze(-1), 
            reference_neighbor_inter_angle.unsqueeze(-1),
            inter_neighbor_centroid_angle.unsqueeze(-1), 
            angle.unsqueeze(-1)], dim=-1)
        
        rri = torch.cat([center_point_features, neighbor_point_features], dim=-1)
        rri = rri.permute(0, 3, 1, 2).contiguous()
        return rri 
    
    def forward(self, xyz, mask):
        knn_xyz, knn_mask = self.grouper(xyz, xyz, mask, mask, xyz.transpose(-1,-2).contiguous())
        knn_xyz = knn_xyz.permute(0, 2, 3, 1)

        with torch.no_grad():
            rri = self.xyz_to_rri_v1(knn_xyz, knn_mask, self.radius)
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