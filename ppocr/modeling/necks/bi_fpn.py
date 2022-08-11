import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
import numpy as np
import math
# class ConvBlock(nn.Layer):
#     def __init__(self, in_channels):
#         super(ConvBlock, self).__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2D(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels),
#             nn.Conv2D(in_channels, in_channels, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2D(num_features=in_channels, momentum=0.9997, epsilon=4e-5), nn.ReLU())
#
#     def forward(self, input):
#         return self.conv(input)
class MaxPool2dStaticSamePadding(nn.Layer):
    """
    created by Zylo117
    The real keras/tensorflow MaxPool2D with same padding
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2D(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.ksize

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]

        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.pool(x)
        return x
class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 if_act=True):
        super(ConvBNLayer, self).__init__()
        weight_attr = paddle.nn.initializer.KaimingUniform()
        self.if_act = if_act
        self.act = "swish"

        self.conv1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.conv2 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)

        self.bn = nn.BatchNorm2D(num_features=in_channels, momentum=0.99, epsilon=1e-3)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        if self.if_act:
            if self.act == "swish":
                x = F.swish(x)
            elif self.act == "hardswish":
                x = F.hardswish(x)
            else:
                print("The activation function({}) is selected incorrectly.".
                      format(self.act))
                exit()
        return x


class BiFPN_N(nn.Layer):

    def __init__(self, in_channels, out_channels, epsilon=1e-4):
        super(BiFPN_N, self).__init__()
        self.epsilon = epsilon
        self.out_channels = out_channels
        weight_attr = paddle.nn.initializer.KaimingUniform()
        # Conv layers
        print(in_channels)
        self.conv5_up = ConvBNLayer(in_channels=in_channels[3], out_channels=self.out_channels)   # 1/32
        self.conv4_up = ConvBNLayer(in_channels=in_channels[2], out_channels=self.out_channels)   # 1/16
        self.conv3_up = ConvBNLayer(in_channels=in_channels[1], out_channels=self.out_channels)   # 1/8
        self.conv2_up = ConvBNLayer(in_channels=in_channels[0], out_channels=self.out_channels)   # 1/4
        # self.conv3_down = ConvBNLayer(in_channels=in_channels[1], out_channels=self.out_channels)
        # self.conv4_down = ConvBNLayer(in_channels=in_channels[2], out_channels=self.out_channels)
        # self.conv5_down = ConvBNLayer(in_channels=in_channels[3], out_channels=self.out_channels)
        # self.conv7_down = ConvBlock(num_channels)
        self.in2_conv = nn.Conv2D(
            in_channels=in_channels[0],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.in3_conv = nn.Conv2D(
            in_channels=in_channels[1],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.in4_conv = nn.Conv2D(
            in_channels=in_channels[2],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.in5_conv = nn.Conv2D(
            in_channels=in_channels[3],
            out_channels=self.out_channels,
            kernel_size=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)

        # self.conv3_down = nn.Conv2D(
        #     in_channels=in_channels[1],
        #     out_channels=self.out_channels,
        #     kernel_size=1,
        #     weight_attr=ParamAttr(initializer=weight_attr),
        #     bias_attr=False)
        # self.conv4_down = nn.Conv2D(
        #     in_channels=in_channels[2],
        #     out_channels=self.out_channels,
        #     kernel_size=1,
        #     weight_attr=ParamAttr(initializer=weight_attr),
        #     bias_attr=False)
        # self.conv5_down = nn.Conv2D(
        #     in_channels=in_channels[3],
        #     out_channels=self.out_channels,
        #     kernel_size=1,
        #     weight_attr=ParamAttr(initializer=weight_attr),
        #     bias_attr=False)

        self.p5_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.p4_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.p3_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)
        self.p2_conv = nn.Conv2D(
            in_channels=self.out_channels,
            out_channels=self.out_channels // 4,
            kernel_size=3,
            padding=1,
            weight_attr=ParamAttr(initializer=weight_attr),
            bias_attr=False)

        # Feature scaling layers
        self.p2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # self.p4_downsample = nn.MaxPool2D(kernel_size=2)
        #11111111111111111111111111111111111111111111111111111111111111111111111
        self.p3_downsample = nn.MaxPool2D(kernel_size=2)
        self.p4_downsample = nn.MaxPool2D(kernel_size=2)
        self.p5_downsample = nn.MaxPool2D(kernel_size=2)
        # self.p3_downsample = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        # self.p4_downsample = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        # self.p5_downsample = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        # self.p3_downsample = MaxPool2dStaticSamePadding(3,2)
        # self.p4_downsample = MaxPool2dStaticSamePadding(3,2)
        # self.p5_downsample = MaxPool2dStaticSamePadding(3,2)
        # Weight
        # self.p2_w1 = paddle.create_parameter(shape=[2], dtype='float32', default_initializer=paddle.nn.initializer.Constant(value=1.0))
        # self.p2_w1_relu = nn.ReLU()
        # self.p3_w1 = paddle.create_parameter(shape=[2], dtype='float32', default_initializer=paddle.nn.initializer.Constant(value=1.0))
        # self.p3_w1_relu = nn.ReLU()
        # self.p4_w1 = paddle.create_parameter(shape=[2], dtype='float32', default_initializer=paddle.nn.initializer.Constant(value=1.0))
        # self.p4_w1_relu = nn.ReLU()
        # # self.p3_w1 = paddle.create_parameter(paddle.ones([2, 2], dtype="float32"))
        # # self.p3_w1_relu = nn.ReLU()
        # # self.p4_w2 = paddle.create_parameter(paddle.ones([3, 3], dtype="float32"))
        # # self.p4_w2_relu = nn.ReLU()
        # self.p3_w2 = paddle.create_parameter(shape=[3], dtype='float32', default_initializer=paddle.nn.initializer.Constant(value=1.0))
        # self.p3_w2_relu = nn.ReLU()
        # self.p4_w2 = paddle.create_parameter(shape=[3], dtype='float32', default_initializer=paddle.nn.initializer.Constant(value=1.0))
        # self.p4_w2_relu = nn.ReLU()
        # self.p5_w2 = paddle.create_parameter(shape=[2], dtype='float32', default_initializer=paddle.nn.initializer.Constant(value=1.0))
        # self.p5_w2_relu = nn.ReLU()

    def forward(self, inputs):
        """
            P5_0 -------------------------- P2_2 -------->
               |-------------|                ↑
                             ↓                |
            P4_0 ---------- P4_1 ---------- P4_2 -------->
               |-------------|--------------↑ ↑
                             ↓                |
            P3_0 ---------- P3_1 ---------- P3_2 -------->
               |-------------|---------------↑ ↑
                             |---------------↓ |
            P2_0 --------------- ---------- P2_2 -------->

        """

        # P2_1/4, P5_1/8, P6_1/16, P7_1/32 2 3 4 5
        c2, c3, c4, c5 = inputs
        p5 = self.conv5_up(c5)
        p4 = self.conv4_up(c4)
        p3 = self.conv3_up(c3)
        p2 = self.conv2_up(c2)

        p5_in = self.in5_conv(p5)
        p4_in = self.in4_conv(p4)
        p3_in = self.in3_conv(p3)
        p2_in = self.in2_conv(p2)
        # P7_0 to P7_2
        # Weights for P6_0 and P7_0 to P6_1
        # p4_w1 = self.p4_w1_relu(self.p4_w1)
        # weight = p4_w1 / (paddle.sum(p4_w1, axis=0) + self.epsilon)
        # Connections for P6_0 and P7_0 to P6_1 respectively
        # print(p5_in.shape)
        # print(self.p4_upsample(p5_in).shape)
        # print(p4_in.shape)

        # print(paddle.sum(p4_w1, axis=0) + self.epsilon)
        p4_up = p4_in + self.p4_upsample(p5_in)
        # print(p4_up.shape)
        # Weights for P5_0 and P6_0 to P5_1
        # p3_w1 = self.p3_w1_relu(self.p3_w1)
        # weight = p3_w1 / (paddle.sum(p3_w1,  axis=0) + self.epsilon)
        # Connections for P5_0 and P6_0 to P5_1 respectively
        p3_up = p3_in + self.p3_upsample(p4_up)

        # Weights for P4_0, P4_1 and P3_2 to P4_2
        # p2_w1 = self.p2_w1_relu(self.p2_w1)
        # weight = p2_w1 / (paddle.sum(p2_w1,  axis=0) + self.epsilon)
        # Connections for P4_0, P4_1 and P3_2 to P4_2 respectively
        p2_out = p2_in + self.p2_upsample(p3_up)
        # Weights for P5_0, P5_1 and P4_2 to P5_2
        # p3_w2 = self.p3_w2_relu(self.p3_w2)
        # weight = p3_w2 / (paddle.sum(p3_w2,  axis=0) + self.epsilon)
        # Connections for P5_0, P5_1 and P4_2 to P5_2 respectively
        p3_out = p3_in + p3_up + self.p3_downsample(p2_out)
        # Weights for P6_0, P6_1 and P5_2 to P6_2
        # p4_w2 = self.p4_w2_relu(self.p4_w2)
        # weight = p4_w2 / (paddle.sum(p4_w2,  axis=0) + self.epsilon)
        # Connections for P6_0, P6_1 and P5_2 to P6_2 respectively
        p4_out = p4_in + p4_up + self.p4_downsample(p3_out)
        # Weights for P7_0 and P6_2 to P7_2
        # p5_w2 = self.p5_w2_relu(self.p5_w2)
        # weight = p5_w2 / (paddle.sum(p5_w2,  axis=0) + self.epsilon)
        # Connections for P7_0 and P6_2 to P7_2
        p5_out = p5_in + self.p5_downsample(p4_out)
        p5 = self.p5_conv(p5_out)
        p4 = self.p4_conv(p4_out)
        p3 = self.p3_conv(p3_out)
        p2 = self.p2_conv(p2_out)

        p5 = F.upsample(p5, scale_factor=8, mode="nearest", align_mode=1)
        p4 = F.upsample(p4, scale_factor=4, mode="nearest", align_mode=1)
        p3 = F.upsample(p3, scale_factor=2, mode="nearest", align_mode=1)

        fuse = paddle.concat([p5, p4, p3, p2], axis=1)
        return fuse
