import torch
import torch.nn as nn
from yolox.models.color_attension import ColorAttentionBlock

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv


class YOLOPAFPNColor(nn.Module):
    def __init__(self, depth=1.0, width=1.0, in_features=("dark3", "dark4", "dark5"), in_channels=[256, 512, 1024],
                 depthwise=False, act="silu"):
        super().__init__()
        Conv = DWConv if depthwise else BaseConv
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.lateral_conv0 = BaseConv(int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act)

        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.reduce_conv1 = BaseConv(int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act)

        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.bu_conv2 = Conv(int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act)

        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.bu_conv1 = Conv(int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act)

        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.att_1 = ColorAttentionBlock(int(in_channels[0] * width))  # 对应dark3输出的256维度通道
        self.att_2 = ColorAttentionBlock(int(in_channels[1] * width))  # 对应dark4输出的512维度通道
        self.att_3 = ColorAttentionBlock(int(in_channels[2] * width))  # 对应dark5输出的1024维度通道

    def forward(self, input):
        out_features = self.backbone.forward(input)
        [feat1, feat2, feat3] = [out_features[f] for f in self.in_features]

        feat1 = self.att_1(feat1)
        feat2 = self.att_2(feat2)
        feat3 = self.att_3(feat3)

        p5 = self.lateral_conv0(feat3)
        p5_upsample = self.upsample(p5)
        p5_upsample = torch.cat([p5_upsample, feat2], 1)
        p5_upsample = self.C3_p4(p5_upsample)

        p4 = self.reduce_conv1(p5_upsample)
        p4_upsample = self.upsample(p4)
        p4_upsample = torch.cat([p4_upsample, feat1], 1)

        p3_out = self.C3_p3(p4_upsample)
        p3_downsample = self.bu_conv2(p3_out)
        p3_downsample = torch.cat([p3_downsample, p4], 1)

        p4_out = self.C3_n3(p3_downsample)
        p4_downsample = self.bu_conv1(p4_out)
        p4_downsample = torch.cat([p4_downsample, p5], 1)

        p5_out = self.C3_n4(p4_downsample)

        return p3_out, p4_out, p5_out
