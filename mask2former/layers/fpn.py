import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import create_norm_layer

class SimpleFPN(nn.Module):
    """Simple Feature Pyramid Network for ViTDet."""

    def __init__(
        self,
        backbone_channel: int,
        in_channels: list[int],
        out_channels: int,
        num_outs: int=4,
        norm_layer="layernorm2d",
    ) -> None:
        super().__init__()
        # assert isinstance(in_channels, list), f"in_channels params type is not list but {type(in_channels)}"
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

