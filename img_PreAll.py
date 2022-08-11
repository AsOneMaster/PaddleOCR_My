#学生：何迅
#创建时间：2021/12/10 14:11
#批量预处理图像
from PIL import Image
import os
import os.path
import numpy as np
import cv2
#自适应调节亮度
def compute(img, min_percentile, max_percentile):
    """计算分位点，目的是去掉图1的直方图两头的异常情况"""
    max_percentile_pixel = np.percentile(img, max_percentile)
    min_percentile_pixel = np.percentile(img, min_percentile)

    return max_percentile_pixel, min_percentile_pixel

def aug(src):
    """图像亮度增强"""
    print(get_lightness(src))
    if (get_lightness(src) > 60 and get_lightness(src)<160):
        print("图片亮度足够，不做增强")
    # 先计算分位点，去掉像素值中少数异常值，这个分位点可以自己配置。
    # 比如1中直方图的红色在0到255上都有值，但是实际上像素值主要在0到20内。


    max_percentile_pixel, min_percentile_pixel = compute(src, 1, 99)

    # 去掉分位值区间之外的值
    src[src >= max_percentile_pixel] = max_percentile_pixel
    src[src <= min_percentile_pixel] = min_percentile_pixel

    # 将分位值区间拉伸到0到255，这里取了255*0.1与255*0.9是因为可能会出现像素值溢出的情况，所以最好不要设置为0到255。
    out = np.zeros(src.shape, src.dtype)
    cv2.normalize(src, out, 255 * 0.1, 255 * 0.9, cv2.NORM_MINMAX)

    return out

def get_lightness(src):
    # 计算亮度
    hsv_image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lightness = hsv_image[:, :, 2].mean()

    return lightness
#rootdir = r'D:\用户目录\我的图片\From Yun\背景图\背景图'  # 指明被遍历的文件夹
rootdir = r'E:\iron_IMG\picture\jiaogang\test'


for parent, dirnames, filenames in os.walk(rootdir):#遍历图片
    print(filenames.__sizeof__())
    for filename in filenames:
        print('parent is :' + parent)
        print('filename is :' + filename)
        currentPath = os.path.join(parent, filename)
        print('the fulll name of the file is :' + currentPath)
        # im = Image.open(currentPath)
        img = cv2.imread(currentPath)
        dst = aug(img)
        newname = 'E:/iron_IMG/picture/jiaogang/testPre/' + filename  # 重新命名
        cv2.imwrite(newname, dst)  # 保存结束
        # if(im.size[0]>im.size[1]):
        #     print('宽：%d,高：%d' % (im.size[0], im.size[1]))
        #     out = im.transpose(Image.ROTATE_270)  # 实现旋转270
        #     newname = r"G:\picture\jiaogang\crop_img\testPre" + '\\' + filename  # 重新命名
        #     out.save(newname)  # 保存结束
        # else:continue