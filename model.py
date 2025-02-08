from typing import List, Tuple
import math
import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from timm.models.vision_transformer_sam import VisionTransformerSAM
from timm.layers import create_norm_layer
from msdeform_pixel_decoder import (
    MSDeformAttnPixelDecoder
)  # noqa

logger = logging.getLogger(__file__)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.setLevel("INFO")


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(
        self, num_pos_feats= 64, temperature = 10000, normalize = False, scale = None
    ):
        super().__init__()
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, x, mask= None) -> Tensor:
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = (~mask).to(x.dtype)
        y_embed = not_mask.cumsum(1)
        x_embed = not_mask.cumsum(2)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.int64, device=x.device).type_as(x)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class VisionTransformer(VisionTransformerSAM):
    def forward(self, x):
        return super().forward_features(x)


class SimpleFPN(nn.Module):
    """Simple Feature Pyramid Network for ViTDet."""

    def __init__(
        self,
        backbone_channel: int,
        in_channels: list[int],
        out_channels: int,
        num_outs: int,
        norm_layer="layernorm2d",
    ) -> None:
        super().__init__()
        assert isinstance(in_channels, list)
        self.backbone_channel = backbone_channel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(self.backbone_channel, self.backbone_channel // 2, 2, 2),
            create_norm_layer(norm_layer, self.backbone_channel // 2),
            nn.ReLU(),
            nn.ConvTranspose2d(
                self.backbone_channel // 2, self.backbone_channel // 4, 2, 2
            ),
        )
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.backbone_channel, self.backbone_channel // 2, 2, 2)
        )
        self.fpn3 = nn.Sequential(nn.Identity())
        self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.num_ins):
            l_conv = nn.Conv2d(
                in_channels[i],
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True if norm_layer == "" else False,
            )

            fpn_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, x) -> tuple:
        """Forward function.

        Args:
            inputs (Tensor): Features from the upstream network, 4D-tensor
        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        print("simple fpn input", x)
        # build FPN
        inputs = []
        inputs.append(self.fpn1(x))
        inputs.append(self.fpn2(x))
        inputs.append(self.fpn3(x))
        inputs.append(self.fpn4(x))

        # build laterals
        laterals = [
            lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build outputs
        # part 1: from original levels
        outs = [self.fpn_convs[i](laterals[i]) for i in range(self.num_ins)]

        # part 2: add extra levels
        if self.num_outs > len(outs):
            for i in range(self.num_outs - self.num_ins):
                outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        return tuple(outs)


class MultiScalePixelDecoder(nn.Module):
    """Pixel decoder with a structure like fpn.

    Args:
        in_channels (list[int] | tuple[int]): Number of channels in the
            input feature maps.
        feat_channels (int): Number channels for feature.
        out_channels (int): Number channels for output.
        norm_cfg (:obj:`ConfigDict` or dict): Config for normalization.
            Defaults to dict(type='GN', num_groups=32).
        act_cfg (:obj:`ConfigDict` or dict): Config for activation.
            Defaults to dict(type='ReLU').
        encoder (:obj:`ConfigDict` or dict): Config for transorformer
            encoder.Defaults to None.
        positional_encoding (:obj:`ConfigDict` or dict): Config for
            transformer encoder position encoding. Defaults to
            dict(type='SinePositionalEncoding', num_feats=128,
            normalize=True).
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict. Defaults to None.
    """

    def __init__(
        self, in_channels, hidden_dim: int, out_channels: int, norm_layer="layernorm2d"
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.num_inputs = len(in_channels)
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()

        for i in range(self.num_inputs - 1, -1, -1):
            lateral_conv = nn.Sequential(
                nn.Conv2d(in_channels[i], hidden_dim, kernel_size=1),
                nn.GroupNorm(8, hidden_dim),
                nn.ReLU(),
            )
            output_conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(8, hidden_dim),
                nn.ReLU(),
            )
            self.lateral_convs.append(lateral_conv)
            self.output_convs.append(output_conv)

        self.last_feat_conv = nn.Sequential(
            nn.Conv2d(in_channels[1], hidden_dim, kernel_size=3, padding=1, stride=1),
            create_norm_layer(norm_layer, hidden_dim),
        )

        self.mask_feature = nn.Conv2d(
            hidden_dim, out_channels, kernel_size=3, stride=1, padding=1
        )

    def init_weights(self) -> None:
        """Initialize weights."""
        for i in range(0, self.num_inputs - 2):
            nn.init.xavier_normal(self.lateral_convs[i].conv, bias=0)
            nn.init.xavier_normal(self.output_convs[i].conv, bias=0)

        nn.init.xavier_normal(self.mask_feature, bias=0)
        nn.init.xavier_normal(self.last_feat_conv, bias=0)

    def forward(self, feats):
        """
        Args:
            feats (list[Tensor]): Feature maps of each level. Each has
                shape of (batch_size, c, h, w).
            batch_img_metas (list[dict]): List of image information.
                Pass in for creating more accurate padding mask. Not
                used here.

        Returns:
            tuple[Tensor, Tensor]: a tuple containing the following:

                - mask_feature (Tensor): Shape (batch_size, c, h, w).
                - memory (Tensor): Output of last stage of backbone.\
                        Shape (batch_size, c, h, w).
        """
        # [ 256 , 128 , 64 , 32 ]

        y = self.last_feat_conv(feats[-1])
        outs = []
        for i in range(self.num_inputs - 2, -1, -1):
            x = feats[i]
            cur_feat = self.lateral_convs[i](x)
            y = cur_feat + F.interpolate(y, size=cur_feat.shape[-2:], mode="nearest")
            y = self.output_convs[i](y)
            outs.append(y)

        mask_feature = self.mask_feature(y)
        multi_scale_memories = outs
        return mask_feature, multi_scale_memories


class CrossAttentionLayer(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
    ):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, tgt, memory, attn_mask, key_padding_mask, pos=None, query_pos=None
    ):
        # self,
        # query: Tensor,
        # key: Tensor,
        # value: Tensor,
        # key_padding_mask: Optional[Tensor] = None,
        # need_weights: bool = True,
        # attn_mask: Optional[Tensor] = None,
        # average_attn_weights: bool = True,
        # is_causal: bool = False,
        logger.debug(
            f"[CrossAttentionLayer::forward] tgt shape: {tgt.shape}, memory shape: {memory.shape}"
        )
        logger.debug(
            f"[CrossAttentionLayer::forward] pos shape: {pos.shape if pos is not None else None}"
        )

        q = tgt + query_pos if query_pos is not None else tgt
        k = memory + pos if pos is not None else memory
        logger.debug(
            f"[CrossAttentionLayer::forward] q shape: {q.shape}, k shape: {k.shape}, v shape: {memory.shape}"
        )
        logger.debug(
            f"[CrossAttentionLayer::forward] attn_mask shape: {attn_mask.shape}, key_padding_mask shape: {key_padding_mask.shape if key_padding_mask is not None else None}"
        )
        attn_output, _ = self.cross_attn(
            query=q,
            key=k,
            value=memory,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        return self.norm(attn_output)


class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt, attn_mask=None, key_padding_mask=None, query_pos=None):
        q = k = tgt + query_pos if query_pos is not None else tgt
        attn_output, _ = self.self_attn(
            query=q,
            key=k,
            value=tgt,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        return attn_output


class FFNLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim=2048, dropout=0.0):
        super().__init__()
        self.lin1 = nn.Linear(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.lin2 = nn.Linear(hidden_dim, embed_dim)

        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.ReLU()

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        shortcut = x
        x = self.lin1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.lin2(x)
        return x + shortcut


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_layers=3,
        num_heads=8,
        embed_dim=256,
        num_queries=200,
        num_classes=100,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.num_queries = num_queries
        self.num_heads = num_heads

        self.input_projects = nn.ModuleList(
            [
                (
                    nn.Conv2d(in_channels, embed_dim, 1)
                    if in_channels != embed_dim
                    else nn.Identity()
                )
                for _ in range(num_layers)
            ]
        )

        self.transformer_cross_attention_layers = nn.ModuleList(
            [CrossAttentionLayer(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.transformer_self_attention_layers = nn.ModuleList(
            [SelfAttentionLayer(embed_dim, num_heads) for _ in range(num_layers)]
        )
        self.transformer_ffn_layers = nn.ModuleList(
            [FFNLayer(embed_dim) for _ in range(num_layers)]
        )

        self.query_feat = nn.Embedding(num_queries, embed_dim)
        self.query_embed = nn.Embedding(num_queries, embed_dim)

        self.level_embed = nn.Embedding(num_layers, embed_dim)
        self.pos_embed = PositionEmbeddingSine(embed_dim // 2, normalize=True)
        self.cls_embed = nn.Linear(embed_dim, num_classes + 1)
        self.mask_embed = MLP(embed_dim, embed_dim, out_channels, 3)
        self.norm = nn.LayerNorm(embed_dim)

    def forward_head(
        self,
        output: Tensor,
        feature: Tensor,
        attn_mask_size: Tuple[int, int],
        threshold=0.5,
    ):
        logger.debug(
            f"[TransformerDecoder::forward_head] input `output` shape: {output.shape}"
        )
        decoder_out = self.norm(output)
        logger.debug(
            f"[TransformerDecoder::forward_head] `output` shape: {output.shape}"
        )
        # decoder_out = decoder_out.transpose(0,1)
        cls_pred = self.cls_embed(decoder_out)
        mask_embed = self.mask_embed(decoder_out)
        mask_pred = torch.einsum("bqc,bchw->bqhw", mask_embed, feature)
        attn_mask = F.interpolate(
            mask_pred, attn_mask_size, mode="bilinear", align_corners=False
        )
        # b,q,h,w -> b,q,(h*w) -> b,1,q,l -> b,n_head,q,l -> b*n_head,q,l
        attn_mask = (
            attn_mask.flatten(2)
            .unsqueeze(1)
            .repeat((1, self.num_heads, 1, 1))
            .flatten(0, 1)
            .bool()
        )
        attn_mask = attn_mask.sigmoid() < threshold
        attn_mask = attn_mask.detach()

        return cls_pred, mask_pred, attn_mask

    def forward(self, x: Tensor, memories: List[Tensor]):
        B = x.shape[0]
        logger.debug(f"[TransformerDecoder::forward] input `x` shape: {x.shape}")
        # pixel decoder
        pixel_decoder_outs = []
        transformer_positions = []
        attn_mask_sizes = []

        for i in range(self.num_layers):
            # generate pos embed for input x, output pos shape is B,C,H,W
            pos = self.pos_embed(memories[i])
            # b,c,h*w -> b,h*w,c
            pos = pos.flatten(2).permute(0, 2, 1)
            logger.debug(f"[TransformerDecoder::forward] pos shape: {pos.shape}")
            transformer_positions.append(pos)
            decoder_out = self.input_projects[i](memories[i])
            # b,c,h,w -> b,c,h*w + 1,c,1
            decoder_out = (
                decoder_out.flatten(2) + self.level_embed.weight[i][None, :, None]
            )
            # b,c,l -> b,l,c
            pixel_decoder_outs.append(decoder_out.permute(0, 2, 1))
            logger.debug(
                f"[TransformerDecoder::forward], decoder_out shape: {pixel_decoder_outs[-1].shape}"
            )
            attn_mask_sizes.append(memories[i].shape[-2:])

        # Q,C -> 1,Q,C -> B,Q,C
        output = self.query_feat.weight.unsqueeze(0).repeat((B, 1, 1))
        query_embed = self.query_embed.weight.unsqueeze(0).repeat((B, 1, 1))
        predict_classes = []
        predict_masks = []

        cls_pred, mask_pred, attn_mask = self.forward_head(
            output, x, attn_mask_size=attn_mask_sizes[0]
        )
        for i in range(self.num_layers):
            level_index = i % self.num_layers
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # mask attention stage
            cross_attn_layer: CrossAttentionLayer = (
                self.transformer_cross_attention_layers[level_index]
            )
            output = cross_attn_layer(
                output,
                pixel_decoder_outs[level_index],
                attn_mask=attn_mask,
                key_padding_mask=None,
                pos=transformer_positions[level_index],
                query_pos=query_embed,
            )

            # self attention stage
            self_attn_layer: SelfAttentionLayer = (
                self.transformer_self_attention_layers[level_index]
            )
            output = self_attn_layer(output, query_pos=query_embed)

            # ffn stage
            ffn_layer: FFNLayer = self.transformer_ffn_layers[level_index]
            output = ffn_layer(output)

            cls_pred, mask_pred, attn_mask = self.forward_head(
                output, x, attn_mask_sizes[(i + 1) % self.num_layers]
            )
            predict_classes.append(cls_pred)
            predict_masks.append(mask_pred)

        return predict_classes[-1], predict_masks[-1]


class Mask2Former(nn.Module):

    def __init__(self, backbone, neck, pixel_decoder, transformer_decoder):
        super().__init__()
        self.backbone = backbone
        self.neck = neck if neck is not None else nn.Identity()
        self.pixel_decoder = pixel_decoder
        self.transformer_decoder = transformer_decoder

    def forward(self, x):
        x = self.backbone(x)
        mlvl_features = self.neck(x)
        for i, f in enumerate(mlvl_features):
            print(f"lvl {i}", f.shape)
        x, memories = self.pixel_decoder(mlvl_features)
        pred_logits, pred_masks = self.transformer_decoder(x, memories)
        return pred_logits, pred_masks
