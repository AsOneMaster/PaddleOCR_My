# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Constant

from ppdet.core.workspace import register, serializable
from ppdet.modeling.layers import ConvNormLayer
from ..shape_spec import ShapeSpec

__all__ = ['BiFPN']


class SeparableConvLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels=None,
                 kernel_size=3,
                 norm_type='bn',
                 norm_groups=32,
                 act='swish'):
        super(SeparableConvLayer, self).__init__()
        assert norm_type in ['bn', 'sync_bn', 'gn', None]
        assert act in ['swish', 'relu', None]

        self.in_channels = in_channels
        if out_channels is None:
            self.out_channels = self.in_channels
        self.norm_type = norm_type
        self.norm_groups = norm_groups
        self.depthwise_conv = nn.Conv2D(
            in_channels,
            in_channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=in_channels,
            bias_attr=False)
        self.pointwise_conv = nn.Conv2D(in_channels, self.out_channels, 1)

        # norm type
        if self.norm_type in ['bn', 'sync_bn']:
            self.norm = nn.BatchNorm2D(self.out_channels)
        elif self.norm_type == 'gn':
            self.norm = nn.GroupNorm(
                num_groups=self.norm_groups, num_channels=self.out_channels)

        # activation
        if act == 'swish':
            self.act = nn.Swish()
        elif act == 'relu':
            self.act = nn.ReLU()

    def forward(self, x):
        if self.act is not None:
            x = self.act(x)
        out = self.depthwise_conv(x)
        out = self.pointwise_conv(out)
        if self.norm_type is not None:
            out = self.norm(out)
        return out


class BiFPNCell(nn.Layer):
    def __init__(self,
                 channels=256,
                 num_levels=5,
                 eps=1e-5,
                 use_weighted_fusion=True,
                 kernel_size=3,
                 norm_type='bn',
                 norm_groups=32,
                 act='swish'):
        super(BiFPNCell, self).__init__()
        self.channels = channels
        self.num_levels = num_levels
        self.eps = eps
        self.use_weighted_fusion = use_weighted_fusion

        # up
        self.conv_up = nn.LayerList([
            SeparableConvLayer(
                self.channels,
                kernel_size=kernel_size,
                norm_type=norm_type,
                norm_groups=norm_groups,
                act=act) for _ in range(self.num_levels - 1)
        ])
        # down
        self.conv_down = nn.LayerList([
            SeparableConvLayer(
                self.channels,
                kernel_size=kernel_size,
                norm_type=norm_type,
                norm_groups=norm_groups,
                act=act) for _ in range(self.num_levels - 1)
        ])

        if self.use_weighted_fusion:
            self.up_weights = self.create_parameter(
                shape=[self.num_levels - 1, 2],
                attr=ParamAttr(initializer=Constant(1.)))
            self.down_weights = self.create_parameter(
                shape=[self.num_levels - 1, 3],
                attr=ParamAttr(initializer=Constant(1.)))

    def _feature_fusion_cell(self,
                             conv_layer,
                             lateral_feat,
                             sampling_feat,
                             route_feat=None,
                             weights=None):
        if self.use_weighted_fusion:
            weights = F.relu(weights)
            weights = weights / (weights.sum() + self.eps)
            if route_feat is not None:
                out_feat = weights[0] * lateral_feat + \
                           weights[1] * sampling_feat + \
                           weights[2] * route_feat
            else:
                out_feat = weights[0] * lateral_feat + \
                           weights[1] * sampling_feat
        else:
            if route_feat is not None:
                out_feat = lateral_feat + sampling_feat + route_feat
            else:
                out_feat = lateral_feat + sampling_feat

        out_feat = conv_layer(out_feat)
        return out_feat

    def forward(self, feats):
        # feats: [P3 - P7]
        lateral_feats = []

        # up
        up_feature = feats[-1]
        for i, feature in enumerate(feats[::-1]):
            if i == 0:
                lateral_feats.append(feature)
            else:
                shape = paddle.shape(feature)
                up_feature = F.interpolate(
                    up_feature, size=[shape[2], shape[3]])
                lateral_feature = self._feature_fusion_cell(
                    self.conv_up[i - 1],
                    feature,
                    up_feature,
                    weights=self.up_weights[i - 1]
                    if self.use_weighted_fusion else None)
                lateral_feats.append(lateral_feature)
                up_feature = lateral_feature

        out_feats = []
        # down
        down_feature = lateral_feats[-1]
        for i, (lateral_feature,
                route_feature) in enumerate(zip(lateral_feats[::-1], feats)):
            if i == 0:
                out_feats.append(lateral_feature)
            else:
                down_feature = F.max_pool2d(down_feature, 3, 2, 1)
                if i == len(feats) - 1:
                    route_feature = None
                    weights = self.down_weights[
                        i - 1][:2] if self.use_weighted_fusion else None
                else:
                    weights = self.down_weights[
                        i - 1] if self.use_weighted_fusion else None
                out_feature = self._feature_fusion_cell(
                    self.conv_down[i - 1],
                    lateral_feature,
                    down_feature,
                    route_feature,
                    weights=weights)
                out_feats.append(out_feature)
                down_feature = out_feature

        return out_feats


@register
@serializable
class BiFPN(nn.Layer):
    """
    Bidirectional Feature Pyramid Network, see https://arxiv.org/abs/1911.09070

    Args:
        in_channels (list[int]): input channels of each level which can be
            derived from the output shape of backbone by from_config.
        out_channel (int): output channel of each level.
        num_extra_levels (int): the number of extra stages added to the last level.
            default: 2
        fpn_strides (List): The stride of each level.
        num_stacks (int): the number of stacks for BiFPN, default: 1.
        use_weighted_fusion (bool): use weighted feature fusion in BiFPN, default: True.
        norm_type (string|None): the normalization type in BiFPN module. If
            norm_type is None, norm will not be used after conv and if
            norm_type is string, bn, gn, sync_bn are available. default: bn.
        norm_groups (int): if you use gn, set this param.
        act (string|None): the activation function of BiFPN.
    """

    def __init__(self,
                 in_channels=(512, 1024, 2048),
                 out_channel=256,
                 num_extra_levels=2,
                 fpn_strides=[8, 16, 32, 64, 128],
                 num_stacks=1,
                 use_weighted_fusion=True,
                 norm_type='bn',
                 norm_groups=32,
                 act='swish'):
        super(BiFPN, self).__init__()
        assert num_stacks > 0, "The number of stacks of BiFPN is at least 1."
        assert norm_type in ['bn', 'sync_bn', 'gn', None]
        assert act in ['swish', 'relu', None]
        assert num_extra_levels >= 0, \
            "The `num_extra_levels` must be non negative(>=0)."

        self.in_channels = in_channels
        self.out_channel = out_channel
        self.num_extra_levels = num_extra_levels
        self.num_stacks = num_stacks
        self.use_weighted_fusion = use_weighted_fusion
        self.norm_type = norm_type
        self.norm_groups = norm_groups
        self.act = act
        self.num_levels = len(self.in_channels) + self.num_extra_levels
        if len(fpn_strides) != self.num_levels:
            for i in range(self.num_extra_levels):
                fpn_strides += [fpn_strides[-1] * 2]
        self.fpn_strides = fpn_strides

        self.lateral_convs = nn.LayerList()
        for in_c in in_channels:
            self.lateral_convs.append(
                ConvNormLayer(in_c, self.out_channel, 1, 1))
        if self.num_extra_levels > 0:
            self.extra_convs = nn.LayerList()
            for i in range(self.num_extra_levels):
                if i == 0:
                    self.extra_convs.append(
                        ConvNormLayer(self.in_channels[-1], self.out_channel, 3,
                                      2))
                else:
                    self.extra_convs.append(nn.MaxPool2D(3, 2, 1))

        self.bifpn_cells = nn.LayerList()
        for i in range(self.num_stacks):
            self.bifpn_cells.append(
                BiFPNCell(
                    self.out_channel,
                    self.num_levels,
                    use_weighted_fusion=self.use_weighted_fusion,
                    norm_type=self.norm_type,
                    norm_groups=self.norm_groups,
                    act=self.act))

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            'in_channels': [i.channels for i in input_shape],
            'fpn_strides': [i.stride for i in input_shape]
        }

    @property
    def out_shape(self):
        return [
            ShapeSpec(
                channels=self.out_channel, stride=s) for s in self.fpn_strides
        ]

    def forward(self, feats):
        assert len(feats) == len(self.in_channels)
        fpn_feats = []
        for conv_layer, feature in zip(self.lateral_convs, feats):
            fpn_feats.append(conv_layer(feature))
        if self.num_extra_levels > 0:
            feat = feats[-1]
            for conv_layer in self.extra_convs:
                feat = conv_layer(feat)
                fpn_feats.append(feat)

        for bifpn_cell in self.bifpn_cells:
            fpn_feats = bifpn_cell(fpn_feats)
        return fpn_feats

# #学生：何迅
# #创建时间：2022/5/9 15:06
# class BiFPN(nn.Module):
#     def __init__(self, num_channels, conv_channels, first_time=False, epsilon=1e-4, onnx_export=False, attention=True):
#         super(BiFPN, self).__init__()
#         self.epsilon = epsilon
#         self.conv6_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
#         self.conv5_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
#         self.conv4_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
#         self.conv3_up = SeparableConvBlock(num_channels, onnx_export=onnx_export)
#
#         self.conv4_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
#         self.conv5_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
#         self.conv6_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
#         self.conv7_down = SeparableConvBlock(num_channels, onnx_export=onnx_export)
#
#         self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
#         self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
#         self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
#         self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')
#
#         self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
#         self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)
#         self.p6_downsample = MaxPool2dStaticSamePadding(3, 2)
#         self.p7_downsample = MaxPool2dStaticSamePadding(3, 2)
#
#         self.swish = MemoryEfficientSwish() if not onnx_export else Swish()
#
#         self.first_time = first_time
#         if self.first_time:
#             # 获取到了efficientnet的最后三层，对其进行通道的下压缩
#             self.p5_down_channel = nn.Sequential(
#                 Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
#                 nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
#             )
#             self.p4_down_channel = nn.Sequential(
#                 Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
#                 nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
#             )
#             self.p3_down_channel = nn.Sequential(
#                 Conv2dStaticSamePadding(conv_channels[0], num_channels, 1),
#                 nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
#             )
#
#             # 对输入进来的p5进行宽高的下采样
#             self.p5_to_p6 = nn.Sequential(
#                 Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
#                 nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
#                 MaxPool2dStaticSamePadding(3, 2)
#             )
#             self.p6_to_p7 = nn.Sequential(
#                 MaxPool2dStaticSamePadding(3, 2)
#             )
#
#             # BIFPN第一轮的时候，跳线那里并不是同一个in
#             self.p4_down_channel_2 = nn.Sequential(
#                 Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
#                 nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
#             )
#             self.p5_down_channel_2 = nn.Sequential(
#                 Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
#                 nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
#             )
#
#         # 简易注意力机制的weights
#         self.p6_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
#         self.p6_w1_relu = nn.ReLU()
#         self.p5_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
#         self.p5_w1_relu = nn.ReLU()
#         self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
#         self.p4_w1_relu = nn.ReLU()
#         self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
#         self.p3_w1_relu = nn.ReLU()
#
#         self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
#         self.p4_w2_relu = nn.ReLU()
#         self.p5_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
#         self.p5_w2_relu = nn.ReLU()
#         self.p6_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
#         self.p6_w2_relu = nn.ReLU()
#         self.p7_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
#         self.p7_w2_relu = nn.ReLU()
#
#         self.attention = attention
#
#     def forward(self, inputs):
#         """ bifpn模块结构示意图
#             P7_0 -------------------------> P7_2 -------->
#                |-------------|                ↑
#                              ↓                |
#             P6_0 ---------> P6_1 ---------> P6_2 -------->
#                |-------------|--------------↑ ↑
#                              ↓                |
#             P5_0 ---------> P5_1 ---------> P5_2 -------->
#                |-------------|--------------↑ ↑
#                              ↓                |
#             P4_0 ---------> P4_1 ---------> P4_2 -------->
#                |-------------|--------------↑ ↑
#                              |--------------↓ |
#             P3_0 -------------------------> P3_2 -------->
#         """
#         if self.attention:
#             p3_out, p4_out, p5_out, p6_out, p7_out = self._forward_fast_attention(inputs)
#         else:
#             p3_out, p4_out, p5_out, p6_out, p7_out = self._forward(inputs)
#
#         return p3_out, p4_out, p5_out, p6_out, p7_out
#
#     def _forward_fast_attention(self, inputs):
#         # ------------------------------------------------#
#         #   当phi=1、2、3、4、5的时候使用fast_attention
#         #   获得三个shape的有效特征层
#         #   分别是C3  64, 64, 40
#         #         C4  32, 32, 112
#         #         C5  16, 16, 320
#         # ------------------------------------------------#
#         if self.first_time:
#             # ------------------------------------------------------------------------#
#             #   第一次BIFPN需要 下采样 与 调整通道 获得 p3_in p4_in p5_in p6_in p7_in
#             # ------------------------------------------------------------------------#
#             p3, p4, p5 = inputs
#             # -------------------------------------------#
#             #   首先对通道数进行调整
#             #   C3 64, 64, 40 -> 64, 64, 64
#             # -------------------------------------------#
#             p3_in = self.p3_down_channel(p3)
#
#             # -------------------------------------------#
#             #   首先对通道数进行调整
#             #   C4 32, 32, 112 -> 32, 32, 64
#             #                  -> 32, 32, 64
#             # -------------------------------------------#
#             p4_in_1 = self.p4_down_channel(p4)
#             p4_in_2 = self.p4_down_channel_2(p4)
#
#             # -------------------------------------------#
#             #   首先对通道数进行调整
#             #   C5 16, 16, 320 -> 16, 16, 64
#             #                  -> 16, 16, 64
#             # -------------------------------------------#
#             p5_in_1 = self.p5_down_channel(p5)
#             p5_in_2 = self.p5_down_channel_2(p5)
#
#             # -------------------------------------------#
#             #   对C5进行下采样，调整通道数与宽高
#             #   C5 16, 16, 320 -> 8, 8, 64
#             # -------------------------------------------#
#             p6_in = self.p5_to_p6(p5)
#             # -------------------------------------------#
#             #   对P6_in进行下采样，调整宽高
#             #   P6_in 8, 8, 64 -> 4, 4, 64
#             # -------------------------------------------#
#             p7_in = self.p6_to_p7(p6_in)
#
#             # 简单的注意力机制，用于确定更关注p7_in还是p6_in
#             p6_w1 = self.p6_w1_relu(self.p6_w1)
#             weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
#             p6_td = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))
#
#             # 简单的注意力机制，用于确定更关注p6_up还是p5_in
#             p5_w1 = self.p5_w1_relu(self.p5_w1)
#             weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
#             p5_td = self.conv5_up(self.swish(weight[0] * p5_in_1 + weight[1] * self.p5_upsample(p6_td)))
#
#             # 简单的注意力机制，用于确定更关注p5_up还是p4_in
#             p4_w1 = self.p4_w1_relu(self.p4_w1)
#             weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
#             p4_td = self.conv4_up(self.swish(weight[0] * p4_in_1 + weight[1] * self.p4_upsample(p5_td)))
#
#             # 简单的注意力机制，用于确定更关注p4_up还是p3_in
#             p3_w1 = self.p3_w1_relu(self.p3_w1)
#             weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
#             p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_td)))
#
#             # 简单的注意力机制，用于确定更关注p4_in_2还是p4_up还是p3_out
#             p4_w2 = self.p4_w2_relu(self.p4_w2)
#             weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
#             p4_out = self.conv4_down(
#                 self.swish(weight[0] * p4_in_2 + weight[1] * p4_td + weight[2] * self.p4_downsample(p3_out)))
#
#             # 简单的注意力机制，用于确定更关注p5_in_2还是p5_up还是p4_out
#             p5_w2 = self.p5_w2_relu(self.p5_w2)
#             weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
#             p5_out = self.conv5_down(
#                 self.swish(weight[0] * p5_in_2 + weight[1] * p5_td + weight[2] * self.p5_downsample(p4_out)))
#
#             # 简单的注意力机制，用于确定更关注p6_in还是p6_up还是p5_out
#             p6_w2 = self.p6_w2_relu(self.p6_w2)
#             weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
#             p6_out = self.conv6_down(
#                 self.swish(weight[0] * p6_in + weight[1] * p6_td + weight[2] * self.p6_downsample(p5_out)))
#
#             # 简单的注意力机制，用于确定更关注p7_in还是p7_up还是p6_out
#             p7_w2 = self.p7_w2_relu(self.p7_w2)
#             weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
#             p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))
#         else:
#             p3_in, p4_in, p5_in, p6_in, p7_in = inputs
#
#             # 简单的注意力机制，用于确定更关注p7_in还是p6_in
#             p6_w1 = self.p6_w1_relu(self.p6_w1)
#             weight = p6_w1 / (torch.sum(p6_w1, dim=0) + self.epsilon)
#             p6_td = self.conv6_up(self.swish(weight[0] * p6_in + weight[1] * self.p6_upsample(p7_in)))
#
#             # 简单的注意力机制，用于确定更关注p6_up还是p5_in
#             p5_w1 = self.p5_w1_relu(self.p5_w1)
#             weight = p5_w1 / (torch.sum(p5_w1, dim=0) + self.epsilon)
#             p5_td = self.conv5_up(self.swish(weight[0] * p5_in + weight[1] * self.p5_upsample(p6_td)))
#
#             # 简单的注意力机制，用于确定更关注p5_up还是p4_in
#             p4_w1 = self.p4_w1_relu(self.p4_w1)
#             weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
#             p4_td = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_td)))
#
#             # 简单的注意力机制，用于确定更关注p4_up还是p3_in
#             p3_w1 = self.p3_w1_relu(self.p3_w1)
#             weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
#             p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_td)))
#
#             # 简单的注意力机制，用于确定更关注p4_in还是p4_up还是p3_out
#             p4_w2 = self.p4_w2_relu(self.p4_w2)
#             weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
#             p4_out = self.conv4_down(
#                 self.swish(weight[0] * p4_in + weight[1] * p4_td + weight[2] * self.p4_downsample(p3_out)))
#
#             # 简单的注意力机制，用于确定更关注p5_in还是p5_up还是p4_out
#             p5_w2 = self.p5_w2_relu(self.p5_w2)
#             weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
#             p5_out = self.conv5_down(
#                 self.swish(weight[0] * p5_in + weight[1] * p5_td + weight[2] * self.p5_downsample(p4_out)))
#
#             # 简单的注意力机制，用于确定更关注p6_in还是p6_up还是p5_out
#             p6_w2 = self.p6_w2_relu(self.p6_w2)
#             weight = p6_w2 / (torch.sum(p6_w2, dim=0) + self.epsilon)
#             p6_out = self.conv6_down(
#                 self.swish(weight[0] * p6_in + weight[1] * p6_td + weight[2] * self.p6_downsample(p5_out)))
#
#             # 简单的注意力机制，用于确定更关注p7_in还是p7_up还是p6_out
#             p7_w2 = self.p7_w2_relu(self.p7_w2)
#             weight = p7_w2 / (torch.sum(p7_w2, dim=0) + self.epsilon)
#             p7_out = self.conv7_down(self.swish(weight[0] * p7_in + weight[1] * self.p7_downsample(p6_out)))
#
#         return p3_out, p4_out, p5_out, p6_out, p7_out
#
#     def _forward(self, inputs):
#         # 当phi=6、7的时候使用_forward
#         if self.first_time:
#             # 第一次BIFPN需要下采样与降通道获得
#             # p3_in p4_in p5_in p6_in p7_in
#             p3, p4, p5 = inputs
#             p3_in = self.p3_down_channel(p3)
#             p4_in_1 = self.p4_down_channel(p4)
#             p4_in_2 = self.p4_down_channel_2(p4)
#             p5_in_1 = self.p5_down_channel(p5)
#             p5_in_2 = self.p5_down_channel_2(p5)
#             p6_in = self.p5_to_p6(p5)
#             p7_in = self.p6_to_p7(p6_in)
#
#             p6_td = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_in)))
#
#             p5_td = self.conv5_up(self.swish(p5_in_1 + self.p5_upsample(p6_td)))
#
#             p4_td = self.conv4_up(self.swish(p4_in_1 + self.p4_upsample(p5_td)))
#
#             p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_td)))
#
#             p4_out = self.conv4_down(
#                 self.swish(p4_in_2 + p4_td + self.p4_downsample(p3_out)))
#
#             p5_out = self.conv5_down(
#                 self.swish(p5_in_2 + p5_td + self.p5_downsample(p4_out)))
#
#             p6_out = self.conv6_down(
#                 self.swish(p6_in + p6_td + self.p6_downsample(p5_out)))
#
#             p7_out = self.conv7_down(self.swish(p7_in + self.p7_downsample(p6_out)))
#
#         else:
#             p3_in, p4_in, p5_in, p6_in, p7_in = inputs
#
#             p6_td = self.conv6_up(self.swish(p6_in + self.p6_upsample(p7_in)))
#
#             p5_td = self.conv5_up(self.swish(p5_in + self.p5_upsample(p6_td)))
#
#             p4_td = self.conv4_up(self.swish(p4_in + self.p4_upsample(p5_td)))
#
#             p3_out = self.conv3_up(self.swish(p3_in + self.p3_upsample(p4_td)))
#
#             p4_out = self.conv4_down(
#                 self.swish(p4_in + p4_td + self.p4_downsample(p3_out)))
#
#             p5_out = self.conv5_down(
#                 self.swish(p5_in + p5_td + self.p5_downsample(p4_out)))
#
#             p6_out = self.conv6_down(
#                 self.swish(p6_in + p6_td + self.p6_downsample(p5_out)))
#
#             p7_out = self.conv7_down(self.swish(p7_in + self.p7_downsample(p6_out)))
#
#         return p3_out, p4_out, p5_out, p6_out, p7_out