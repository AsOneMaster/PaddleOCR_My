#学生：何迅
#创建时间：2022/3/10 10:05
import paddle
import paddle.nn as nn


class CBAM(paddle.nn.Layer):
    def __init__(self, feature_channel, feature_height, feature_width):
        super(CBAM, self).__init__()
        self.c_maxpool = nn.MaxPool2D((feature_height, feature_width), 1)
        self.c_avgpool = nn.AvgPool2D((feature_height, feature_width), 1)
        self.s_maxpool = nn.MaxPool2D(1, 1)
        self.s_avgpool = nn.AvgPool2D(1, 1)
        self.s_conv = nn.Conv2D(int(feature_channel * 2), 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.shared_MLP = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(feature_channel, int(feature_channel / 2)),
            nn.ReLU(),
            nn.Linear(int(feature_channel / 2), feature_channel),
            nn.ReLU()
        )

    def Channel_Attention(self, x):
        b, c, h, w = x.shape

        x_m = self.c_maxpool(x)
        x_a = self.c_avgpool(x)

        mlp_m = self.shared_MLP(x_m)
        mlp_a = self.shared_MLP(x_a)

        mlp_m = paddle.reshape(mlp_m, [b, c, 1, 1])
        mlp_a = paddle.reshape(mlp_a, [b, c, 1, 1])

        c_c = paddle.add(mlp_a, mlp_m)
        Mc = self.sigmoid(c_c)
        return Mc

    def Spatial_Attention(self, x):
        x_m = self.s_maxpool(x)
        x_a = self.s_avgpool(x)

        x_concat = paddle.concat([x_m, x_a], axis=1)
        x_x = self.s_conv(x_concat)
        Ms = self.sigmoid(x_x)

        return Ms

    def forward(self, x):
        Mc = self.Channel_Attention(x)
        F1 = Mc * x

        Ms = self.Spatial_Attention(F1)
        refined_feature = Ms * F1

        return refined_feature
if __name__ == '__main__':
    x = paddle.randn(shape=[1, 16, -1, -1])  # b, c, h, w

    ca_model = CBAM(feature_channel=16, feature_height=-1, feature_width=-1)
    y = ca_model(x)
    print(y.shape)
