#学生：何迅
#创建时间：2022/5/10 10:48
import paddle

# 1. 从PaddleOCR中import DBFPN
from ppocr.modeling.backbones.det_mobilenet_v3_CA import MobileNetV3_CA
from ppocr.modeling.necks.bi_fpn import BiFPN

# 2. 获得Backbone网络输出结果
fake_inputs = paddle.randn([1, 3, 960, 960], dtype="float32")
model_backbone = MobileNetV3_CA()
in_channles = model_backbone.out_channels

# 3. 声明FPN网络
model_fpn = BiFPN(in_channels=in_channles, out_channels=256)

# 4. 打印FPN网络
print(model_fpn)

# 5. 计算得到FPN结果输出
outs = model_backbone(fake_inputs)
fpn_outs = model_fpn(outs)

# 6. 打印FPN输出特征形状
print(f"The shape of fpn outs {fpn_outs.shape}")