import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import create_norm_layer, create_act_layer
from .ms_deform_attn import MSDeformAttn
from ..utils import PositionEmbeddingSine


class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
        norm_layer="layernorm",
    ):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = create_norm_layer(norm_layer, d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = create_act_layer(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = create_norm_layer(norm_layer, d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
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
        src2 = self.self_attn(
            self.with_pos_embed(src, pos),
            reference_points,
            src,
            spatial_shapes,
            level_start_index,
            padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)
        return src


class MSDeformAttnTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for i in range(num_layers)]
        )
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H_ - 0.5, H_, dtype=valid_ratios.dtype, device=device
                ),
                torch.linspace(
                    0.5, W_ - 0.5, W_, dtype=valid_ratios.dtype, device=device
                ),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)  # [1, H_ * W_, 2]
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
        for _, layer in enumerate(self.layers):
            output = layer(
                output,
                pos,
                reference_points,
                spatial_shapes,
                level_start_index,
                padding_mask,
            )

        return output


class MSDeformAttnTransformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        num_feature_levels=4,
        enc_n_points=4,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = MSDeformAttnTransformerEncoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            enc_n_points,
        )
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer, num_encoder_layers)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self._init_weight()

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._init_weight()
        nn.init.normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, pos_embeds):
        masks = [
            torch.zeros(
                (x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool
            )
            for x in srcs
        ]
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
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
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(
            src_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            lvl_pos_embed_flatten,
            mask_flatten,
        )

        return memory, spatial_shapes, level_start_index


class MSDeformAttnPixelDecoder(nn.Module):
    def __init__(
        self,
        in_channels=[256, 256, 256, 256],
        in_strides=[4, 8, 16, 32],
        transformer_dropout=0.1,
        transformer_nheads=8,
        transformer_dim_feedforward=1024,
        transformer_enc_layers=6,
        conv_dim=256,
        mask_dim=256,
        # 前面backbone和neck中提取到多尺度特征需要传入 pixel decoder 中 transformer 模块的索引
        transformer_in_features=[1, 2, 3],
        common_stride=4,
    ):
        super().__init__()
        # transformer_input_shape = {k: v for k, v in input_shape.items() if k in transformer_in_features}
        self.in_features_indexes = range(len(in_channels))
        # this is the input shape of pixel decoder
        # self.in_features = [k for k, v in input_shape.items()]  # starting from "res3" to "res5"
        self.feature_channels = in_channels  # [v.channel for k, v in input_shape.items()] # eg. [16, 64, 128, 256]
        # this is the input shape of transformer encoder (could use less features than pixel decoder
        self.transformer_in_features = transformer_in_features  # [k for k, v in transformer_input_shape.items()]  # starting from "res3" to "res5"
        transformer_in_channels = [
            k for i, k in enumerate(in_channels) if i in transformer_in_features
        ]  # eg. [64, 128, 256]
        self.transformer_feature_strides = [
            k for i, k in enumerate(in_strides) if i in transformer_in_features
        ]  # to decide extra FPN layers

        self.transformer_num_feature_levels = len(self.transformer_in_features)
        if self.transformer_num_feature_levels > 1:
            input_proj_list = []
            # from low resolution to high resolution (res5 -> res3)
            for in_channels in transformer_in_channels[::-1]:
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, conv_dim, kernel_size=1),
                        nn.GroupNorm(32, conv_dim),
                    )
                )
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(transformer_in_channels[-1], conv_dim, kernel_size=1),
                        nn.GroupNorm(32, conv_dim),
                    )
                ]
            )

        # for proj in self.input_proj:
        #     nn.init.xavier_uniform_(proj[0].weight, gain=1)
        #     nn.init.constant_(proj[0].bias, 0)

        self.transformer = MSDeformAttnTransformer(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            num_feature_levels=self.transformer_num_feature_levels,
        )
        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        self.mask_dim = mask_dim
        # use 1x1 conv instead
        self.mask_features = nn.Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # nn.init.kaiming_uniform_(self.mask_features.weight, a=1)
        # nn.init.constant_(self.mask_features.bias, 0)

        self.maskformer_num_feature_levels = 3  # always use 3 scales
        self.common_stride = common_stride

        # extra fpn levels
        stride = min(self.transformer_feature_strides)
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))
        lateral_convs = []
        output_convs = []

        for idx, in_channels in enumerate(
            self.feature_channels[: self.num_fpn_levels]
        ):  # res2 -> fpn
            lateral_conv = nn.Sequential(
                nn.Conv2d(in_channels, conv_dim, kernel_size=1),
                nn.GroupNorm(32, conv_dim),
            )

            output_conv = nn.Sequential(
                nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(32, conv_dim),
                nn.ReLU(inplace=True),
            )

            # nn.init.kaiming_uniform_(lateral_conv[0].weight, a=1)
            # nn.init.constant_(lateral_conv[0].bias, 0)

            # nn.init.kaiming_uniform_(output_conv[0].weight, a=1)
            # nn.init.constant_(output_conv[0].bias, 0)

            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            self.add_module("layer_{}".format(idx + 1), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        self._init_weight()
        # print(
        #     f"param of pixel decoder: in_channels:{in_channels}, in_strides:{in_strides}, transformer_in_features:{transformer_in_features}, transformer_in_channels:{transformer_in_channels}, transformer_in_strides: {self.transformer_feature_strides}"
        # )
        # print(
        #     f"param of fpn: num_fpn_levels:{self.num_fpn_levels}, common_stride:{self.common_stride}"
        # )
        # print(
        #     f"len of lateral_convs:{len(self.lateral_convs)}, len of output_convs:{len(self.output_convs)}"
        # )

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            
            if isinstance(m, MSDeformAttnTransformer):
                m._init_weight()

    def forward(self, features):
        return self.forward_features(features)

    def forward_features(self, features):
        srcs = []
        pos = []
        # Reverse feature maps into top-down order (from low to high resolution), 'res5' -> 'res3'
        for idx, f in enumerate(self.transformer_in_features[::-1]):
            x = features[f].float()  # deformable detr does not support half precision
            # print(f"transformer pixel decoder {idx} input", x.shape)
            srcs.append(self.input_proj[idx](x))
            pos.append(self.pe_layer(x))

        y, spatial_shapes, level_start_index = self.transformer(srcs, pos)
        bs = y.shape[0]

        split_size_or_sections = [None] * self.transformer_num_feature_levels
        for i in range(self.transformer_num_feature_levels):
            if i < self.transformer_num_feature_levels - 1:
                split_size_or_sections[i] = (
                    level_start_index[i + 1] - level_start_index[i]
                )
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = torch.split(y, split_size_or_sections, dim=1)

        out = []
        multi_scale_features = []
        num_cur_levels = 0
        for i, z in enumerate(y):
            out.append(
                z.transpose(1, 2).view(
                    bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]
                )
            )

        # append `out` with extra FPN levels
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features_indexes[: self.num_fpn_levels][::-1]):
            x = features[f].float()
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cur_fpn = lateral_conv(x)
            # Following FPN implementation, we use nearest upsampling here
            y = cur_fpn + F.interpolate(
                out[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False
            )
            y = output_conv(y)
            # print(f"pixel decoder {idx} out:", y.shape)
            out.append(y)

        for o in out:
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1

        return self.mask_features(out[-1]), multi_scale_features
