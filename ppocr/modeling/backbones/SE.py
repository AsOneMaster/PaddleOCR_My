#学生：何迅
#创建时间：2022/3/10 10:05
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
class SEModule(nn.Layer):
    def __init__(self, in_channels, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.conv1 = nn.Conv2D(
            in_channels=in_channels,
            out_channels=in_channels // reduction,
            kernel_size=1,
            stride=1,
            padding=0)
        self.conv2 = nn.Conv2D(
            in_channels=in_channels // reduction,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0)

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = F.hardsigmoid(outputs, slope=0.2, offset=0.5)
        return inputs * outputs
if __name__ == '__main__':
    x = paddle.randn(shape=[1, 16, 1, 1])  # b, c, h, w

    se_model = SEModule(in_channels=16)
    y = se_model(x)
    print(y.shape)
    se = SEModule(in_channels=16)
    params_info = paddle.summary(se, (1, 16, 960, 960))
    print(params_info)