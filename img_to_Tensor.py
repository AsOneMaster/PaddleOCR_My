#学生：何迅
#创建时间：2022/5/25 21:02
import cv2
import numpy as np
## to tensor
import paddle


def img_trans(img, size=480):
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

def get_batch_data(data_path):
    print(str(data_path))
    img = cv2.imread(str(data_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img_trans(img)
    img = np.array(img).astype('float32')
    img = paddle.to_tensor(img)
    img = paddle.reshape(img, [-1, 80, 1, 1])
    return img
img = get_batch_data('test2.jpg')
print(type(img))
print(img.shape)