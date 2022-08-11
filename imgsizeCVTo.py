#批量改变分割图片角度
from PIL import Image
import os
import os.path

#rootdir = r'D:\用户目录\我的图片\From Yun\背景图\背景图'  # 指明被遍历的文件夹
rootdir = r'E:\iron_IMG\jiaogang\test1'
i = 0
for parent, dirnames, filenames in os.walk(rootdir):#遍历图片
    print(filenames.__sizeof__())

    for filename in filenames:
        i = i + 1
        print('parent is :' + parent)

        print('filename is :' + filename)
        currentPath = os.path.join(parent, filename)
        print('the fulll name of the file is :' + currentPath)
        im = Image.open(currentPath)
        newname = r"E:\iron_IMG\jiaogang\test1" + '\\' + str(i)+".jpg" # 重新命名
        print(newname)
        im.save(newname)
        # im.save(newname)  # 保存结束
        # if(im.size[0]>im.size[1]):
        #     print('宽：%d,高：%d' % (im.size[0], im.size[1]))
        #     out = im.transpose(Image.ROTATE_270)  # 实现旋转270
        #     newname = r"G:\picture\jiaogang\crop_img" + '\\' + filename  # 重新命名
        #     out.save(newname)  # 保存结束
        # else:continue

