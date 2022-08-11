#学生：何迅
#创建时间：2022/5/16 20:24
# 参考代码：
# https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.4/ppocr/data/simple_dataset.py
import logging
import random
import numpy as np

import os
import sys
from ppocr.data.imaug import create_operators, transform

logger = logging.basicConfig()


# CopyPaste示例的类
class CopyPasteDemo(object):
    def __init__(self, ):
        self.data_dir = "E:/iron_IMG/jiaogang/test"
        self.label_file_list = "E:/iron_IMG/jiaogang/test/Label.txt"
        self.data_lines = self.get_image_info_list(self.label_file_list)
        self.data_idx_order_list = list(range(len(self.data_lines)))
        transforms = [
            {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
            {"DetLabelEncode": {}},
            {"CopyPaste": {"objects_paste_ratio": 1.0}},
        ]
        self.ops = create_operators(transforms)

    # 选择一张图像，将其中的内容拷贝到当前图像中
    def get_ext_data(self, idx):
        ext_data_num = 1  # 多少图
        ext_data = []

        load_data_ops = self.ops[:2]

        next_idx = idx

        while len(ext_data) < ext_data_num:
            next_idx = (next_idx + 1) % len(self)
            file_idx = self.data_idx_order_list[next_idx]
            data_line = self.data_lines[file_idx]
            data_line = data_line.decode('utf-8')
            substr = data_line.strip("\n").split("\t")
            file_name = substr[0]
            label = substr[1]
            img_path = os.path.join(self.data_dir, file_name)
            data = {'img_path': img_path, 'label': label}
            if not os.path.exists(img_path):
                continue
            with open(data['img_path'], 'rb') as f:
                img = f.read()
                data['image'] = img
            data = transform(data, load_data_ops)
            if data is None:
                continue
            ext_data.append(data)
        return ext_data

    # 获取图像信息
    def get_image_info_list(self, file_list):
        if isinstance(file_list, str):
            file_list = [file_list]
        data_lines = []
        for idx, file in enumerate(file_list):
            with open(file, "rb") as f:
                lines = f.readlines()
                data_lines.extend(lines)
        return data_lines

    # 获取DataSet中的一条数据
    def __getitem__(self, idx):
        file_idx = self.data_idx_order_list[idx]
        data_line = self.data_lines[file_idx]
        try:
            data_line = data_line.decode('utf-8')
            substr = data_line.strip("\n").split("\t")
            file_name = substr[0]
            label = substr[1]
            img_path = os.path.join(self.data_dir, file_name)
            data = {'img_path': img_path, 'label': label}
            if not os.path.exists(img_path):
                raise Exception("{} does not exist!".format(img_path))
            with open(data['img_path'], 'rb') as f:
                img = f.read()
                data['image'] = img
            data['ext_data'] = self.get_ext_data(idx)
            outs = transform(data, self.ops)
        except Exception as e:
            print(
                "When parsing line {}, error happened with msg: {}".format(
                    data_line, e))
            outs = None
        if outs is None:
            return
        return outs

    def __len__(self):
        return len(self.data_idx_order_list)


copy_paste_demo = CopyPasteDemo()

idx = 1
data1 = copy_paste_demo[idx]
print(copy_paste_demo[2])
print(data1["polys"])
# print(data1["image"])
print(data1["texts"])
# print(data1["ext_data"][0]["polys"])
# print(data1["ext_data"][1]["polys"])

import cv2
import matplotlib.pyplot as plt
img1 = cv2.imread(data1["img_path"])
img2 = cv2.imread(data1["ext_data"][0]["img_path"])
# img2_2 = cv2.imread(data1["ext_data"][1]["img_path"])
# img2 = cv2.imread(data1["ext_data"][0]["img_path"])
plt.figure(figsize=(10,6))
plt.imshow(img1[:,:,::-1])
plt.show()
plt.figure(figsize=(10,6))
plt.imshow(img2[:,:,::-1])
plt.show()
# plt.figure(figsize=(10,6))
# plt.imshow(img2_2[:,:,::-1])
# plt.show()


import json
infos = copy_paste_demo.data_lines[idx]
infos = json.loads(infos.decode('utf-8').split("\t")[1])

img3 = data1["image"].copy()
plt.figure(figsize=(15, 10))
plt.imshow(img3[:,:,::-1])
# plt.show()
# 原始标注信息
for info in infos:
    xs, ys = zip(*info["points"])
    xs = list(xs)
    ys = list(ys)
    xs.append(xs[0])
    ys.append(ys[0])
    plt.plot(xs, ys, "r")
# 新增的标注信息
for poly_idx in range(len(infos), len(data1["polys"])):
    poly = data1["polys"][poly_idx]
    xs, ys = zip(*poly)
    xs = list(xs)
    ys = list(ys)
    print(xs)
    print(ys)
    xs.append(xs[0])
    ys.append(ys[0])
    plt.plot(xs, ys, "b")
    # plt.show()
plt.show()
# plt.show()