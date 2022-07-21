import torch
from torch import nn
import torch.nn.functional as F
from pytorchpoints.config import kfg
from pytorchpoints.config import configurable
from .build import LOCAL_AGGREGATION_REGISTRY
from pytorchpoints.functional.pt_custom_ops.pt_utils import MaskedQueryAndGroup

from .pointwise_mlp_rri import rel_pos_to_rel_rot_v11

__all__ = ["SPEMLP"]


@LOCAL_AGGREGATION_REGISTRY.register()
class SPEMLP(nn.Module):
    @configurable
    def __init__(
        self,
        in_channels,
        out_channels,
        radius,
        nsample,
        feat_type,
        num_mlps,
        reduction,
        init_gamma1,
        init_gamma2
    ):
        super(SPEMLP, self).__init__()
        self.num_mlps = num_mlps
        self.reduction = reduction
        ss = feat_type.split(',')
        if len(ss) >= 1:
            self.attn_type = ss[0]
        else:
            self.attn_type = 'n'
        if len(ss) >= 2:
            self.mul_type = ss[1]
        else:
            self.mul_type = 'n'
        if len(ss) >= 3:
            self.gc_type = ss[2]
        else:
            self.gc_type = 'n'

        last_dim1 = in_channels // 3 + 3
        last_dim2 = in_channels // 3 + 4
        last_dim3 = (in_channels - in_channels // 3 * 2) + 12

        assert self.num_mlps == 1
        self.conv_fi1 = nn.Conv2d(in_channels // 3, out_channels // 3, kernel_size=1, bias=False)
        self.conv_fi2 = nn.Conv2d(in_channels // 3, out_channels // 3, kernel_size=1, bias=False)
        self.conv_fi3 = nn.Conv2d((in_channels - in_channels // 3 * 2), (out_channels - out_channels // 3 * 2), kernel_size=1, bias=False)

        self.conv_df_pos1 = nn.Conv2d(last_dim1, out_channels // 3, kernel_size=1, bias=False)
        self.conv_df_pos2 = nn.Conv2d(last_dim2, out_channels // 3, kernel_size=1, bias=False)
        self.conv_df_pos3 = nn.Conv2d(last_dim3, (out_channels - out_channels // 3 * 2), kernel_size=1, bias=False)

        self.act = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.feats_bn = nn.BatchNorm2d(19)

        if self.attn_type != 'n':
            self.attn = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

        if self.mul_type == 'dyn_conv':
            self.w1 = nn.Sequential(
                nn.Conv2d(3, out_channels // 3, kernel_size=1, bias=True),
                nn.Sigmoid()
            )
            self.w2 = nn.Sequential(
                nn.Conv2d(4, out_channels // 3, kernel_size=1, bias=True),
                nn.Sigmoid()
            )
            self.w3 = nn.Sequential(
                nn.Conv2d(12, (out_channels - out_channels // 3 * 2), kernel_size=1, bias=True),
                nn.Sigmoid()
            )
        elif self.mul_type == 'ada_weight':
            self.w1 = nn.Sequential(
                nn.Conv2d(3, out_channels // 3, kernel_size=1, bias=True)
            )
            self.w2 = nn.Sequential(
                nn.Conv2d(4, out_channels // 3, kernel_size=1, bias=True)
            )
            self.w3 = nn.Sequential(
                nn.Conv2d(12, (out_channels - out_channels // 3 * 2), kernel_size=1, bias=True)
            )
        elif self.mul_type == 'ada_weight_tanh':
            self.w1 = nn.Sequential(
                nn.Conv2d(3, out_channels // 3, kernel_size=1, bias=True),
                nn.Tanh()
            )
            self.w2 = nn.Sequential(
                nn.Conv2d(4, out_channels // 3, kernel_size=1, bias=True),
                nn.Tanh()
            )
            self.w3 = nn.Sequential(
                nn.Conv2d(12, (out_channels - out_channels // 3 * 2), kernel_size=1, bias=True),
                nn.Tanh()
            )
        elif self.mul_type == 'dyn_conv_cat':
            self.w1 = nn.Sequential(
                nn.Conv2d(in_channels + 3, out_channels // 3, kernel_size=1, bias=True),
                nn.Sigmoid()
            )
            self.w2 = nn.Sequential(
                nn.Conv2d(in_channels + 4, out_channels // 3, kernel_size=1, bias=True),
                nn.Sigmoid()
            )
            self.w3 = nn.Sequential(
                nn.Conv2d(in_channels + 12, (out_channels - out_channels // 3 * 2), kernel_size=1, bias=True),
                nn.Sigmoid()
            )
        elif self.mul_type == 'dyn_conv_cat_c1':
            self.w1 = nn.Sequential(
                nn.Conv2d(in_channels + 3, 1, kernel_size=1, bias=True),
                nn.Sigmoid()
            )
            self.w2 = nn.Sequential(
                nn.Conv2d(in_channels + 4, 1, kernel_size=1, bias=True),
                nn.Sigmoid()
            )
            self.w3 = nn.Sequential(
                nn.Conv2d(in_channels + 12, 1, kernel_size=1, bias=True),
                nn.Sigmoid()
            )

        if self.gc_type == 'gc_linear':
            self.gc = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=True)
        elif self.gc_type == 'gc_mlp':
            self.gc = nn.Sequential(
                nn.Conv1d(out_channels, out_channels // 4, kernel_size=1, bias=True),
                nn.ReLU(),
                nn.Conv1d(out_channels // 4, out_channels, kernel_size=1, bias=True),
            )
        elif self.gc_type == 'gc_gate':
            self.gc = nn.Sequential(
                nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=True),
                nn.Sigmoid()
            )

        self.radius = radius
        self.grouper = MaskedQueryAndGroup(radius, nsample, use_xyz=False, ret_grouped_xyz=True,
                                           normalize_xyz=False)
        self._gamma1 = init_gamma1
        self._gamma2 = init_gamma2

    @classmethod
    def from_config(cls, cfg, in_channels, out_channels, radius, nsample):
        return {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "radius": radius,
            "nsample": nsample,
            "feat_type": cfg.MODEL.LA_TYPE.FEAT_TYPE,
            "num_mlps": cfg.MODEL.LA_TYPE.NUM_MLPS,
            "reduction": cfg.MODEL.LA_TYPE.REDUCTION,
            "init_gamma1": cfg.MODEL.INIT_GAMMA1,
            "init_gamma2": cfg.MODEL.INIT_GAMMA2,
        }

    @classmethod
    def add_config(cls, cfg):
        pass

    def forward(self, q_xyz, sup_xyz, q_mask, sup_mask, sup_feats):
        nbhd_feats, rel_pos, nbhd_mask = self.grouper(q_xyz, sup_xyz, q_mask, sup_mask, sup_feats)

        B, C, N, M = nbhd_feats.shape

        feats = rel_pos_to_rel_rot_v11(rel_pos, q_xyz, nbhd_mask, self.radius)
        feats = self.feats_bn(feats)

        # 3, 4, 12
        feats1 = feats[:, 0:3] * self._gamma1
        feats2 = feats[:, 3:7] * self._gamma2
        feats3 = feats[:, 7:]

        nbhd_feats1 = nbhd_feats[:, 0:C // 3]
        nbhd_feats2 = nbhd_feats[:, C // 3:C // 3 * 2]
        nbhd_feats3 = nbhd_feats[:, C // 3 * 2:]

        center_feats1 = nbhd_feats1[..., 0].unsqueeze(-1)
        rel_feats1 = nbhd_feats1 - center_feats1
        center_feats2 = nbhd_feats2[..., 0].unsqueeze(-1)
        rel_feats2 = nbhd_feats2 - center_feats2
        center_feats3 = nbhd_feats3[..., 0].unsqueeze(-1)
        rel_feats3 = nbhd_feats3 - center_feats3

        output1 = self.conv_fi1(center_feats1) + self.conv_df_pos1(torch.cat([rel_feats1, feats1], dim=1))
        output2 = self.conv_fi2(center_feats2) + self.conv_df_pos2(torch.cat([rel_feats2, feats2], dim=1))
        output3 = self.conv_fi3(center_feats3) + self.conv_df_pos3(torch.cat([rel_feats3, feats3], dim=1))

        if self.mul_type != 'n':
            if self.mul_type == 'dyn_conv_cat' or self.mul_type == 'dyn_conv_cat_c1':
                center_feats = nbhd_feats[..., 0].unsqueeze(-1).expand(B, C, N, M)
                output1 = output1 * self.w1(torch.cat((center_feats, feats1), dim=1))
                output2 = output2 * self.w2(torch.cat((center_feats, feats2), dim=1))
                output3 = output3 * self.w3(torch.cat((center_feats, feats3), dim=1))
            else:
                output1 = output1 * self.w1(feats1)
                output2 = output2 * self.w2(feats2)
                output3 = output3 * self.w3(feats3)

        output = torch.cat([output1, output2, output3], dim=1)
        output = self.act(output)

        if self.attn_type != 'n':
            if self.attn_type == 'pool':
                attn = self.attn(nbhd_feats.mean(-1, True))
            elif self.attn_type == 'query':
                attn = self.attn(nbhd_feats[..., 0].unsqueeze(-1))
            elif self.attn_type == 'global':
                attn = self.attn(nbhd_feats.mean(-1, True).mean(-2, True))
            else:
                raise NotImplementedError(f'Attention {self.attn_type} not implemented in PointWiseMLP')
            attn = attn.sigmoid()

            output = output * attn

        if self.reduction == 'max':
            out_feats = F.max_pool2d(output, kernel_size=[1, M]).squeeze(-1)
        elif self.reduction == 'top3':
            topk = 3
            out_feats = torch.topk(output, k=topk, dim=-1, sorted=False)[0]
            out_feats = F.avg_pool2d(out_feats, kernel_size=[1, topk]).squeeze(-1)
        elif self.reduction == 'top5':
            topk = 5
            out_feats = torch.topk(output, k=topk, dim=-1, sorted=False)[0]
            out_feats = F.avg_pool2d(out_feats, kernel_size=[1, topk]).squeeze(-1)
        elif self.reduction == 'top7':
            topk = 7
            out_feats = torch.topk(output, k=topk, dim=-1, sorted=False)[0]
            out_feats = F.avg_pool2d(out_feats, kernel_size=[1, topk]).squeeze(-1)
        elif self.reduction == 'top9':
            topk = 9
            out_feats = torch.topk(output, k=topk, dim=-1, sorted=False)[0]
            out_feats = F.avg_pool2d(out_feats, kernel_size=[1, topk]).squeeze(-1)
        elif self.reduction == 'avg' or self.reduction == 'mean':
            feats_mask = nbhd_mask + (1 - q_mask[:, :, None])
            feats_mask = feats_mask[:, None, :, :]
            out_feats = (output * feats_mask).sum(-1) / feats_mask.sum(-1)
        elif self.reduction == 'sum':
            feats_mask = nbhd_mask + (1 - q_mask[:, :, None])
            feats_mask = feats_mask[:, None, :, :]
            out_feats = (output * feats_mask).sum(-1)
        else:
            raise NotImplementedError(f'Reduction {self.reduction} not implemented in PointWiseMLP')

        if self.gc_type != 'n':
            if self.gc_type == 'gc_gate':
                out_feats = out_feats * self.gc(out_feats.mean(-1, True))
            else:
                out_feats = out_feats + self.gc(out_feats.mean(-1, True))
        return out_feats
