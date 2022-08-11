# 读取预测数据
# 读取训练好的模型
import paddle
from ppocr.modeling.backbones.det_mobilenet_v3_CA import MobileNetV3_CA

model = MobileNetV3_CA(scale=0.5, model_name='large', disable_se=False)
model.set_state_dict(paddle.load('out/db_A+B+C/best_accuracy.pdparams'))


# 111111111111111111111111
# 读取预测数据
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


def img_trans(img, size=960):
    h, w, _ = img.shape
    if h == w:
        img = cv2.resize(img, (size, size))
    if h < w:
        hn = int(h * size / w)
        up_border = int((size - hn) / 2)
        down_border = size - hn - up_border
        img = cv2.resize(img, (size, hn))
        img = cv2.copyMakeBorder(img, up_border, down_border, 0, 0, cv2.BORDER_CONSTANT)
    elif h > w:
        wn = int(w * size / h)
        left_border = int((size - wn) / 2)
        right_border = size - wn - left_border
        img = cv2.resize(img, (wn, size))
        img = cv2.copyMakeBorder(img, 0, 0, left_border, right_border, cv2.BORDER_CONSTANT)
    # print(h, w, img.shape)
    # plt.imshow(img)
    # plt.show()
    img = np.transpose(img, (2, 0, 1))
    img = img / 255.
    # print(img[0][100])
    return img


def get_image_info_list(file_list='11'):
    if isinstance(file_list, str):
        file_list = [file_list]
    data_lines = []
    for idx, file in enumerate(file_list):
        with open(file, "rb") as f:
            lines = f.readlines()
            data_lines.extend(lines)
    return data_lines


def get_batch_data(data_path, batch_size):
    # data_list = np.loadtxt(data_path+'Label.txt', dtype='str')
    data_list = get_image_info_list(data_path + 'Label.txt')
    data_idx_order_list = list(range(len(data_list)))

    # with open(r''+data_path+'Label.txt') as f:
    #    data_list = f.read().splitlines()
    # data_list = np.loadtxt(data_path+'Eval.txt', dtype='str')

    np.random.seed(135)
    # np.random.shuffle(data_list)

    imgs, labels = [], []
    for i in range(batch_size):
        print(i)
        # file_idx = data_idx_order_list[i]
        data_list1 = data_list[i + 27]
        data_line = data_list1.decode('utf-8')
        substr = data_line.strip("\n").split("\t")
        print('1------------------', substr[0])

        file_name = substr[0]
        # label = substr[1]
        # print(str(data_list[i][0]))
        # print(str(data_list[i][1]))
        img = cv2.imread(str(data_path) + str(substr[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img_trans(img)
        print(img.shape)
        imgs.append(img)
        # label = [int(substr[1])]
        # labels.append(label)

    return np.array(imgs).astype('float32')


imgs = get_batch_data(
    data_path='E:/iron_IMG/jiaogang/test1/',
    batch_size=4
)


def save_show_pics(pics, file_name='tmp', save_path='./out/cam/', save_root_path='./out/cam/', figsize=None,
                   pic_size=960):
    if not os.path.exists(save_root_path):
        os.makedirs(save_root_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    shape = pics.shape
    pic = pics.transpose((0, 2, 3, 1)).reshape([-1, 4, pic_size, pic_size, 3])
    pic = np.concatenate(tuple(pic), axis=1)
    pic = np.concatenate(tuple(pic), axis=1)
    # pic = (pic + 1.) / 2.
    plt.imsave(save_path + file_name + '.jpg', pic)

    if figsize != None:
        plt.figure(figsize=figsize, dpi=80)
    plt.imshow(pic)
    plt.xticks([])
    plt.yticks([])
    plt.show()


# 存储测试batch的图片
print('读取数据的形状', imgs.shape)
save_show_pics(imgs, figsize=(14, 14), file_name='orginal')


def get_cam(model, img):
    img = paddle.to_tensor(img)

    # MobileNetV3的forward过程
    y = model.conv(img)
    for block in model.stages:
        y = block(y)
    feature_map = y  # 得到模型最后一个卷积层的特征图
    # y = model.pool2d_avg(y)s

    # y = paddle.reshape(y, shape=[-1, model.out_channels[0]])
    # predict = model.out(y) # 得到前向计算的结果
    weight = model.parameters()[-1]  # 得到GAP后面最后一个全连接层的权重
    # print( model.parameters())
    weight = weight.numpy()
    print(feature_map.numpy().shape)
    # weight = weight[label.numpy()[0]] # 抽取全连接层中激活目标类别c的权重
    cam = feature_map.numpy() * weight.reshape([1, feature_map.shape[1], 1, 1])  # 使用抽取的权重对卷积层输出的特征图加权
    cam = np.sum(cam[0], axis=0)  # 加和各个通道的激活图
    # 将激活图归一化到0～1之间
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)
    return cam


def show_cam(model, data, pic_size=960, figsize=None):
    imgs = data
    heat_maps = []
    for i in range(data.shape[0]):
        img = (imgs[i] * 255.).astype('uint8').transpose([1, 2, 0])  # 归一化至[0,255]区间，形状：[h,w,c]
        cam = get_cam(model, data[i:i + 1])
        cam = cv2.resize(cam * 255., (imgs.shape[2], imgs.shape[3])).astype('uint8')  # 调整热图尺寸与图片一致、归一化
        cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)  # 将热图转化为“伪彩热图”显示模式
        superimposed_img = cv2.addWeighted(cam, .3, img, .7, 1.)  # 将特图叠加到原图片上
        heat_maps.append(superimposed_img)
    heat_maps = np.array(heat_maps)

    heat_maps = heat_maps.reshape([-1, 4, pic_size, pic_size, 3])
    heat_maps = np.concatenate(tuple(heat_maps), axis=1)
    heat_maps = np.concatenate(tuple(heat_maps), axis=1)
    cv2.imwrite('./out/cam/cam_ca.jpg', heat_maps)

    if figsize != None:
        plt.figure(figsize=figsize, dpi=80)
    heat_maps = plt.imread('./out/cam/cam_ca.jpg')
    plt.imshow(heat_maps)
    plt.xticks([])
    plt.yticks([])
    plt.show()


show_cam(model, imgs, pic_size=960, figsize=(14, 14))