#学生：何迅
#创建时间：2022/5/24 17:01

# 具体代码实现位于
# https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.4/ppocr/optimizer/__init__.py
# 导入学习率优化器构建的函数
from ppocr.optimizer import build_lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
# 咱们也可以看看warmup_epoch为2时的效果
lr_config = {'name': 'Cosine', 'learning_rate': 0.001, 'warmup_epoch': 2}
epochs = 240 # config['Global']['epoch_num']
iters_epoch = 100  # len(train_dataloader)
lr_scheduler=build_lr_scheduler(lr_config, epochs, iters_epoch)

iters = 0
lr = []
for epoch in range(epochs):
    for _ in range(iters_epoch):
        lr_scheduler.step() # 对应 https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.4/tools/program.py#L262
        iters += 1
        lr.append(lr_scheduler.get_lr())

x = np.arange(iters,dtype=np.int64)
y = np.array(lr,dtype=np.float64)

plt.figure(figsize=(15, 6))
plt.plot(x,y,color='red',label='lr')

plt.tick_params(labelsize=15)
# plt.title(u'Cosine with Warmup',fontsize=20)
plt.xlabel(u'iters',fontsize=20)
plt.ylabel(u'lr',fontsize=20)

plt.legend()
plt.savefig("learning_late.png")
plt.show()