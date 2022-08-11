#学生：何迅
#创建时间：2022/5/27 8:30
import paddle
import paddle.nn as nn
import math
import paddle.nn.functional as F


class LKA(nn.Layer):
    def __init__(self, dim):
        super(LKA, self).__init__()
        self.conv0 = nn.Conv2D(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2D(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2D(dim, dim, 1)


    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class Attention(nn.Layer):
    def __init__(self, d_model):
        super(Attention, self).__init__()

        self.proj_1 = nn.Conv2D(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv2D(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x

if __name__ == '__main__':
    x = paddle.randn(shape=[1, 16, 640, 640])  # b, c, h, w

    ca_model = LKA(dim=16)
    ca = LKA(dim=16)
    y = ca_model(x)
    print(y.shape)
    params_info = paddle.summary(ca, (1, 16, 640, 640))
    print(params_info)
