#学生：何迅
#创建时间：2022/4/26 21:31
import paddle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from paddle.fluid.reader import DataLoader
from paddle.vision.transforms import transforms
from paddle.vision.transforms import Compose, ColorJitter, Resize
transform1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize()
])

transform = Compose([ColorJitter(contrast=0.2)])
train_loader = paddle.vision.datasets.ImageFolder('E:/iron_IMG/iron/', transform=transform1)
def YOCO(images, h, w):
    print(images)
    images = np.array(images)
    print(images)
    img_w1 = images[0:, 0:, 0:int(w / 2)]
    img_w1 = np.transpose(img_w1, (1, 2, 0))
    img_w2 = images[0:, 0:, int(w / 2):w]
    img_w2 = np.transpose(img_w2, (1, 2, 0))
    # print(img_w1)
    x1 = transform(img_w1)
    x1 = paddle.to_tensor(x1)
    x2 = img_w2
    x2 = paddle.to_tensor(x2)
    # print(x1)
    # x2 = transform(img_w2)
    # x1 = transform(images[:, :, 0:int(w / 2)])
    # # print(type(x1))
    # x2 = images[:, int(w / 2):w, :]
    images = paddle.concat(x=[x1, x2], axis=1)
    # images = paddle.concat(x=[transform(images[:, :, 0:int(w / 2)]),transform(images[:, :, int(w / 2):w])], axis=2) if \
    #     paddle.rand(1) > 0.5 else paddle.concat(x=[transform(images[:, 0:int(h / 2), :]), transform(images[:, int(h / 2):h, :])], axis=1)
    return images


for i, (images) in enumerate(train_loader):
    if i == 1:
        print(images[0].shape)
    c, h, w = images[0].shape
    # perform augmentations with YOCO
    images = YOCO(images[0], h, w)

    if i == 2:
        print(type(images))
        print(images.shape)
        img = images.numpy()  # FloatTensor转为ndarray
        # img = np.transpose(img, (1, 2, 0))  # 把channel那一维放到最后
        # 显示图片
        plt.imshow(img)
        plt.show()

