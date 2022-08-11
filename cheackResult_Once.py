#学生：何迅
#创建时间：2021/11/3 16:37
# from PIL import Image
# import matplotlib.pyplot as plt
# ## 显示轻量级模型识别结果
# img_path = "inference_results/11.jpg"
# img = Image.open(img_path)
# plt.figure("results_img", figsize=(30,30))
# plt.imshow(img)
# plt.show()
from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
# Paddleocr目前支持中英文、英文、法语、德语、韩语、日语，可以通过修改lang参数进行切换
# 参数依次为`ch`, `en`, `french`, `german`, `korean`, `japan`。
ocr = PaddleOCR(det_model_dir='./inference/det/', rec_model_dir='./inference/rec/', lang='en')
img_path = r'./doc/imgs/4.jpeg'
result = ocr.ocr(img_path)
for line in result:
    print(line)

# 显示结果
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='/path/to/PaddleOCR/doc/fonts/simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')