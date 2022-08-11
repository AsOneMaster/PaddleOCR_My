#学生：何迅
#创建时间：2022/3/1 16:06
import cv2
from matplotlib import pyplot as plt
#
# image = cv2.imread("Tougao.jpg")
# # 将输入图像转为灰度图
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# # 绘制灰度图
# plt.subplot(311), plt.imshow(gray, "gray")
# plt.title("gray image"), plt.xticks([]), plt.yticks([])
# # 绘制原图
# plt.subplot(312), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.title("input image"), plt.xticks([]), plt.yticks([])
# # 对灰度图使用 Ostu 算法
# ret1, th1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
# # # 绘制灰度直方图
# # plt.subplot(313), plt.hist(gray.ravel(), 256)
# # # 标注 Ostu 阈值所在直线
# # plt.axvline(x=ret1, color='red', label='otsu')
# # plt.legend(loc='upper right')
# # plt.title("Histogram"), plt.xticks([]), plt.yticks([])
# ##绘制二值化图像
# plt.subplot(313), plt.imshow(th1, "gray")
# plt.title("output image"), plt.xticks([]), plt.yticks([])
#
# reverse=cv2.bitwise_not(th1)
# color=cv2.cvtColor(reverse,cv2.COLOR_GRAY2BGR)
# plt.figure(figsize=(16,9))
# plt.imshow(color), plt.xticks([]), plt.yticks([])
#
# reverse1=cv2.bitwise_not(gray)
# color=cv2.cvtColor(reverse1,cv2.COLOR_GRAY2BGR)
# plt.figure(figsize=(16,9))
# plt.imshow(color), plt.xticks([]), plt.yticks([])
#
# reverse2=cv2.bitwise_not(reverse1)
# color=cv2.cvtColor(reverse2,cv2.COLOR_GRAY2BGR)
# plt.figure(figsize=(16,9))
# plt.imshow(color), plt.xticks([]), plt.yticks([])
# plt.show()
# import paddle
# import numpy as np
# train_dataset = paddle.vision.datasets.mnist


# a = 9292035398230089
# b = 9999999999999999
# c = 2/(1/a+1/b)
# print(c)

# from math import ceil
# # a = 13
# # b = 5
# # c = ceil(a/b)
# # print(c)

from PyQt5.QtWidgets import QFileDialog, QWidget
import xlwt
from db.ocrService import OcrService
"""启动数据库操作服务"""
ocr_service = OcrService()
class saveDao:
    def __init__(self):
        super(saveDao, self).__init__()

    def save_excel(self):
        savefile_name = QFileDialog.getSaveFileName(QWidget(), '选择保存路径', '', 'Excel files(*.xls)')

        global path_savefile_name

        path_savefile_name = savefile_name[0]

        book = xlwt.Workbook()
        sheet = book.add_sheet('新数据')
        result = ocr_service.select_all()
        row = len(result)
        col = len((result[0]))
        content = ['检测结果', '图片']

        # for i in range(col):
        #     # self.tableWidget.horizontalHeaderItem(m).text()
        #     content.append('','')
        # print(content)
        for i in range(1):
            for j in range(col):
                sheet.write(i, j, content[j])
        for i in range(row):
            for j in range(col):
                try:
                    sheet.write(i + 1, j, result[i][j])
                except:
                    continue
                # print(self.tableWidget.item(i,j).text())
        book.save(path_savefile_name)

if __name__ == '__main__':
    saveDao().save_excel()