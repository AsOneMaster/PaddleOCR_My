import base64
import os
import cv2
import time
import uuid

from datetime import timedelta

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

from paddleocr import PaddleOCR,draw_ocr


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(hours=1)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
# app.config['CACHES_FOLDER'] = 'D:/PythonProject/PaddleOCR/web/caches/'

def allowed_file(fname):
    return '.' in fname and fname.rsplit('.', 1)[1].lower() in ['png', 'jpg', 'jpeg']

@app.route("/")
def index():
    return render_template('index.html')
@app.route("/ocr")
def upfile():
    return render_template('fileOcr.html')
@app.route("/ocr_mob")
def upfile_mob():
    return render_template('fileOcr_mob.html')
    
@app.route('/ocr', methods=['POST', 'GET'])
def detect():
    files = request.files.getlist('file')

    print("1===========================", files)
    paths = []
    boxes = []
    texts = []
    confidence = []
    for file in files:
        print(file.filename)
        if file and allowed_file(file.filename):
            ext = file.filename.rsplit('.', 1)[1]
            random_name = '{}.{}'.format(uuid.uuid4().hex, ext)
            path = os.path.join(os.getcwd(), 'caches')
            save_path = os.path.join(path, secure_filename(random_name))
            paths.append(save_path)
            print("2==========================", save_path,"------------",type(save_path))
            file.save(save_path)
            # time-1
            # t1 = time.time()
    for path in paths:
        img = cv2.imread(path)
        # img = Image.open(save_path).convert('RGB')
        # print("3==========================",img)
        img_result = ocr.ocr(img, cls=False)
        print("3=========================", img_result)
        # time-2
        # t2 = time.time()
        # image = Image.open(save_path).convert('RGB')
        '''
        识别结果将以列表返回在img_result，根据具体需求进行改写
        '''

        for i in range(len(img_result)):
            # print("4--------------------",img_result[i])
            # boxes.append(img_result[i][0][0])
            texts.append(img_result[i][1][0])
            # confidence.append(img_result[i][1][1])
    # draw_img = draw_ocr(image, boxes, texts, confidence,font_path='/path/to/PaddleOCR/doc/fonts/simfang.ttf')
    # path_oc = os.path.join(os.getcwd(), 'static/ocr')
    # draw_img_save = os.path.join(path_oc, "/1.jpg")
    # # if not os.path.exists(draw_img_save):
    # #     os.makedirs(draw_img_save)
    # draw_img = Image.fromarray(draw_img)
    # draw_img.save(draw_img_save)
    # # draw = base64.b64encode(draw_img.tobytes())
    # print("4=========================", draw_img_save)
    print("4=========================", texts)
    return jsonify({
        'success': 200,
        'scores': confidence,
        'txt': texts
        # 't': '{:.4f}s'.format(t2-t1)
    })
    # return jsonify({'status': 'faild'})

if __name__ == '__main__':
    ocr = PaddleOCR(det_model_dir='D:/PythonProject/PaddleOCR_My/inference/db_ABC/', rec_model_dir='D:/PythonProject/PaddleOCR_My/inference/rec_ca/', lang='en',use_gpu= False)# 查看README的参数说明
    app.run(host='127.0.0.1', port=8090, debug=True, threaded=True, processes=1)
    '''
    app.run()中可以接受两个参数，分别是threaded和processes，用于开启线程支持和进程支持。
    1.threaded : 多线程支持，默认为False，即不开启多线程;
    2.processes：进程数量，默认为1.
    '''
