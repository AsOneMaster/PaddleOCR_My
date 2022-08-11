#学生：何迅
#创建时间：2022/5/16 15:46
import PIL.Image as Image
from PIL import ImageDraw, ImageFont

import os

IMAGES_PATH = r'E:/iron_IMG/lunwen/data_plus/'  # 图片集地址
IMAGES_FORMAT = ['.jpg', '.JPG','.png']  # 图片格式
IMAGE_SIZE = 256  # 每张小图片的大小
IMAGE_ROW = 1  # 图片间隔，也就是合并成一张图后，一共有几行
IMAGE_COLUMN = 3  # 图片间隔，也就是合并成一张图后，一共有几列
IMAGE_SAVE_PATH = 'my_lunWen1.jpg'  # 图片转换后的地址

# 获取图片集地址下的所有图片名称
image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if
               os.path.splitext(name)[1] == item]


#排序，这里需要根据自己的图片名称切割，得到数字
image_names.sort(key=lambda x:int(x.split(("."),2)[0]))
# image_names.sort(key=lambda x:int(x.split(("."),2)[1]))
#image_names.sort(key=lambda x:int(x[:-4]))

# 简单的对于参数的设定和实际图片集的大小进行数量判断
if len(image_names) != IMAGE_ROW * IMAGE_COLUMN:
    raise ValueError("合成图片的参数和要求的数量不能匹配！")

padding= 10
head_padding= 50
# 定义图像拼接函数
def image_compose():
    to_image = Image.new('RGB',( IMAGE_COLUMN * IMAGE_SIZE+padding*(IMAGE_COLUMN-1), head_padding+IMAGE_ROW * IMAGE_SIZE+padding*(IMAGE_ROW-1)),'white' )  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for y in range(1, IMAGE_ROW + 1):
        for x in range(1, IMAGE_COLUMN + 1):
            from_image = Image.open(IMAGES_PATH + image_names[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
                (IMAGE_SIZE , IMAGE_SIZE), Image.ANTIALIAS)
            to_image.paste(from_image, ((x - 1) * IMAGE_SIZE+padding* (x - 1), (y - 1) * IMAGE_SIZE+padding* (y - 1)))

    draw = ImageDraw.Draw(to_image)
    text = ['','','']
    # text_offset = [0, 0, 0]
    text_offset = [100, 100, 100]
    font = ImageFont.truetype(r'C:\Windows\Fonts\simsun.ttc', 18, encoding='unic')

    for i in range(1, IMAGE_ROW+ 1):
        draw.text((10,text_offset[i-1]+256*(i-1)), text[i-1], fill='#666', font=font)

    return to_image.save(IMAGE_SAVE_PATH)  # 保存新图


image_compose()  # 调用函数
