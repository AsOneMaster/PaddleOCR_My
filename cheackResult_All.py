#学生：何迅
#创建时间：2021/11/3 16:37
#运行检测与识别模型，并导出图像
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import os
import os.path
# Paddleocr目前支持中英文、英文、法语、德语、韩语、日语，可以通过修改lang参数进行切换
# 参数依次为`ch`, `en`, `french`, `german`, `korean`, `japan`。
# ocr = PaddleOCR(det_model_dir='./output/best_DB_Mv3/', rec_model_dir='./output/rec_en_number_inference/', lang='en')
ocr = PaddleOCR(det_model_dir='./inference/det/', rec_model_dir='./inference/rec/', lang='en')
rootdir = r'E:\OCRData\Image_eleRong'
for parent, dirnames, filenames in os.walk(rootdir):#遍历图片
    print(filenames.__sizeof__())
    for filename in filenames:
        print('parent is :' + parent)
        print('filename is :' + filename)
        img_path = os.path.join(parent, filename)
        result = ocr.ocr(img_path)
        for line in result:
            print(line)

        # 显示结果
        image = Image.open(img_path).convert('RGB')
        n = len(result)
        ###字符从左往右排序
        # for i in range(n):
        #     for j in range(0,n-i-1):
        #         if result[j][0][0][0] > result[j+1][0][0][0]:
        #             result[j],result[j+1] = result[j+1],result[j]
        # ##获取连续字符串,并存储数据
        # word = ""
        # with open('字符识别率textPre.txt', 'a+', encoding='utf-8') as f:
        #     for i in range(n):
        #             word += (result[i][1][0])
        #             f.write(result[i][1][0]+':'+str(result[i][1][1])+'\n')
        #     f.close()
        # print(word)
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]
        im_show = draw_ocr(image, boxes, txts, scores, font_path='/path/to/PaddleOCR/doc/fonts/simfang.ttf')
        im_show = Image.fromarray(im_show)
        newname = 'E:/OCRData/Image_eleRongTest/'+filename  # 重新命名
        im_show.save(newname)

