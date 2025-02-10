import math
import copy
from functools import partial
import sys
import logging
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from timm.layers import create_norm_layer, create_act_layer, to_2tuple
from .ms_deform_attn import MSDeformAttn


logger = logging.getLogger(__file__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel("DEBUG")


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, scale=None
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")

        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros(
                (x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool
            )
        not_mask = (~mask).to(x.dtype)
        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(
            self.num_pos_feats, dtype=torch.int64, device=x.device
        ).type_as(x)
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats
        )

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)


class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        ffn_dim=1024,
        dropout=0.1,
        norm_layer="layernorm",
        act_layer="relu",
        num_levels=4,
        num_points=4,
    ):
        super().__init__()
        self.self_attn = MSDeformAttn(
            d_model=embed_dim,
            n_levels=num_levels,
            n_heads=num_heads,
            n_points=num_points,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = create_norm_layer(norm_layer, embed_dim)
        # ffn
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.activation = create_act_layer(act_layer)
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = create_norm_layer(norm_layer, embed_dim)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.fc2(self.dropout2(self.activation(self.fc1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(
        self,
        src,
        pos,
        reference_points,
        spatial_shapes,
        level_start_index,
        padding_mask=None,
    ):
        # self attention
        # print("before src shape is",src.shape)
        # print("before src is", src)
        # breakpoint()
        query = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            query=query,
            reference_points=reference_points,
            input_flatten=src,
            input_spatial_shapes=spatial_shapes,
            input_level_start_index=level_start_index,
            input_padding_mask=padding_mask,
        )
        # print("src2 shape", src2.shape)
        # print("after src2 is", src2)
        src = src + self.dropout(src2)
        # print("[MSDeformAttnTransformerEncoderLayer::forward] before norm src is",src)
        src = self.norm(src)
        # print("[MSDeformAttnTransformerEncoderLayer::forward] before norm src is",src)
        src = self.forward_ffn(src)
        # print("[MSDeformAttnTransformerEncoderLayer::forward] after mlp src is",src)

        if self.training:
            if torch.isinf(src).any() or torch.isnan(src).any():
                clamp_value = torch.finfo(src.dtype).max - 1000
                # print(clamp_value)
                src = torch.clamp(src, min=-clamp_value, max=clamp_value)

        return src


class MSDeformAttnTransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)]
        )
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes_list, valid_ratios, device):
        """
        Get reference points for each feature map. Used in decoder.

        Args:
            spatial_shapes_list (`list` of `tuple`):
                Spatial shapes of the backbone feature maps as a list of tuples.
            valid_ratios (`torch.FloatTensor`):
                Valid ratios of each feature map, has shape of `(batch_size, num_feature_levels, 2)`.
            device (`torch.device`):

                Device on which to create the tensors.
        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_queries, num_feature_levels, 2)`
        """
        reference_points_list = []
        for lvl, (height, width) in enumerate(spatial_shapes_list):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, height - 0.5, height, dtype=valid_ratios.dtype, device=device
                ),
                torch.linspace(
                    0.5, width - 0.5, width, dtype=valid_ratios.dtype, device=device
                ),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * width)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)

        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]

        return reference_points

    def forward(
        self,
        src,
        spatial_shapes,
        level_start_index,
        valid_ratios,
        pos=None,
        padding_mask=None,
    ):
        output = src
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=src.device
        )
        for i, layer in enumerate(self.layers):
            # print(f"layer {i} start>>>>>>>>>>>>")
            output = layer(
                src=output,
                pos=pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                padding_mask=padding_mask,
            )
        return output


class MSDeformAttnTransformer(nn.Module):
    def __init__(
        self,
        embed_dim=256,
        num_heads=8,
        num_layers=6,
        ffn_dim=1024,
        dropout=0.1,
        act_layer="relu",
        num_levels=4,
        num_points=4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.encoder = MSDeformAttnTransformerEncoder(
            MSDeformAttnTransformerEncoderLayer(
                embed_dim,
                num_heads,
                ffn_dim,
                dropout,
                act_layer=act_layer,
                num_points=num_points,
            ),
            num_layers=num_layers,
        )
        self.level_embed = nn.Parameter(torch.Tensor(num_levels, embed_dim))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask, dtype=torch.float32):
        """Get the valid ratio of all feature maps."""
        _, height, width = mask.shape
        valid_height = torch.sum(~mask[:, :, 0], 1)
        valid_width = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_heigth = valid_height.to(dtype) / height
        valid_ratio_width = valid_width.to(dtype) / width
        valid_ratio = torch.stack([valid_ratio_width, valid_ratio_heigth], -1)
        return valid_ratio

    def forward(self, srcs, pos_embeds):
        masks = [
            torch.zeros(
                (x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool
            )
            for x in srcs
        ]
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []

        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            B, C, H, W = src.shape
            spatial_shape = (H, W)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)  # B,C,H,W - > B,L,C
            mask = mask.flatten(1)  # B,H,W -> B,L
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m, dtype=src_flatten.dtype) for m in masks], 1
        )

        memory = self.encoder(
            src=src_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            pos=lvl_pos_embed_flatten,
            padding_mask=mask_flatten,
        )

        return memory, spatial_shapes, level_start_index


class MSDeformAttnPixelDecoder(nn.Module):

    def __init__(
        self,
        in_channels,
        embed_dim,
        mask_dim,
        num_outs=3,
        transformer_dropout=0.0,
        transformer_num_heads=4,
        transformer_ffn_dim=1024,
        transformer_num_layers=3,
        norm_layer="layernorm2d",
        act_layer="relu",
    ):
        super().__init__()
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        input_projs = []

        for in_chans in in_channels[::-1]:
            input_projs.append(
                nn.Sequential(
                    nn.Conv2d(in_chans, embed_dim, kernel_size=1),
                    nn.GroupNorm(32, embed_dim),
                )
            )

        self.input_projs = nn.ModuleList(input_projs)

        self.transformer = MSDeformAttnTransformer(
            embed_dim=embed_dim,
            dropout=transformer_dropout,
            num_heads=transformer_num_heads,
            ffn_dim=transformer_ffn_dim,
            num_layers=transformer_num_layers,
            num_levels=self.num_ins,
        )

        N_steps = embed_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        self.mask_features = nn.Conv2d(
            embed_dim, mask_dim, kernel_size=1, stride=1, padding=0
        )

        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for in_chans in in_channels[::-1]:
            lateral_conv = nn.Sequential(
                nn.Conv2d(in_chans, embed_dim, kernel_size=1),
                create_norm_layer(norm_layer, embed_dim),
                create_act_layer(act_layer),
            )

            output_conv = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
                create_norm_layer(norm_layer, embed_dim),
                create_act_layer(act_layer),
            )
            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)

    def forward(self, features):
        features = features[: self.num_ins]
        srcs = []
        pos = []

        # 从小到大计算
        for i, f in enumerate(features[::-1]):
            x = f.float()
            srcs.append(self.input_projs[i](x))
            pos.append(self.pe_layer(x))

        # transformer 会将多尺度特征合并在一起进行计算，返回合并后的特征y，每一层特征的spatial_shape,和每一层特征的起始index
        y, spatial_shapes, level_start_index = self.transformer(srcs, pos)

        # 将多尺度特征分离开来
        bs = y.shape[0]
        split_size_or_sections = [None] * self.num_ins
        for i in range(self.num_ins):
            if i < self.num_ins - 1:
                split_size_or_sections[i] = (
                    level_start_index[i + 1] - level_start_index[i]
                )
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = torch.split(y, split_size_or_sections, dim=1)

        outs = []
        multi_scale_features = []
        num_cur_levels = 0

        for i, z in enumerate(y):
            outs.append(
                z.transpose(1, 2).view(
                    bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]
                )
            )

        for i, f in enumerate(features[::-1]):
            cur_feat = self.lateral_convs[i](f)
            y = cur_feat + F.interpolate(
                outs[-1], size=cur_feat.shape[-2:], mode="bilinear", align_corners=False
            )
            y = self.output_convs[i](y)
            outs.append(y)

        for o in outs:
            if num_cur_levels < self.num_outs:
                multi_scale_features.append(o)
                num_cur_levels += 1
        return self.mask_features(outs[-1]), multi_scale_features
