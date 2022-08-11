#学生：何迅
#创建时间：2021/12/9 16:12
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
    if (get_lightness(src) > 60 and get_lightness(src)<130):
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

img = cv2.imread(r"26-2.png")
dst = aug(img)
print(get_lightness(dst))
result = np.concatenate([img, dst], axis=1)
# cv2.namedWindow('result', 0)
# cv2.imshow('result', result)
# cv2.waitKey(0)
cv2.imwrite('26-2.jpg', dst)

##光照不均用于灰度图
# import cv2
# import numpy as np
#
# def unevenLightCompensate(img, blockSize):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     average = np.mean(gray)
#
#     rows_new = int(np.ceil(gray.shape[0] / blockSize))
#     cols_new = int(np.ceil(gray.shape[1] / blockSize))
#
#     blockImage = np.zeros((rows_new, cols_new), dtype=np.float32)
#     for r in range(rows_new):
#         for c in range(cols_new):
#             rowmin = r * blockSize
#             rowmax = (r + 1) * blockSize
#             if (rowmax > gray.shape[0]):
#                 rowmax = gray.shape[0]
#             colmin = c * blockSize
#             colmax = (c + 1) * blockSize
#             if (colmax > gray.shape[1]):
#                 colmax = gray.shape[1]
#
#             imageROI = gray[rowmin:rowmax, colmin:colmax]
#             temaver = np.mean(imageROI)
#             blockImage[r, c] = temaver
#
#     blockImage = blockImage - average
#     blockImage2 = cv2.resize(blockImage, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
#     gray2 = gray.astype(np.float32)
#     dst = gray2 - blockImage2
#     dst = dst.astype(np.uint8)
#     dst = cv2.GaussianBlur(dst, (3, 3), 0)
#     dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
#
#     return dst
#
# if __name__ == '__main__':
#     file = 'E:/iron_IMG/picture/jiaogang/test/16.jpg'
#     blockSize = 16
#     img = cv2.imread(file)
#     dst = unevenLightCompensate(img, blockSize)
#
#     result = np.concatenate([img, dst], axis=1)
#     cv2.namedWindow('result', 0)
#     cv2.imshow('result', result)
#     cv2.waitKey(0)
##直方图均衡法，适用于灰度图
# import cv2
# import numpy as np
#
# img = cv2.imread('E:/iron_IMG/picture/jiaogang/test/16.jpg',1)
# b,g,r = cv2.split(img)
# cv2.equalizeHist(b,b)
# cv2.equalizeHist(g,g)
# cv2.equalizeHist(r,r)
# img_new = cv2.merge([b,g,r])
# result = np.concatenate([img, img_new], axis=1)
# cv2.namedWindow('result', 0)
# cv2.imshow('result', result)
# cv2.waitKey(0)
# cv2.imwrite('out.jpg', img_new)

# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# # Automatic brightness and contrast optimization with optional histogram clipping
# def automatic_brightness_and_contrast(image, clip_hist_percent=1):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     # Calculate grayscale histogram
#     hist = cv2.calcHist([gray],[0],None,[256],[0,256])
#     hist_size = len(hist)
#
#     # Calculate cumulative distribution from the histogram
#     accumulator = []
#     accumulator.append(float(hist[0]))
#     for index in range(1, hist_size):
#         accumulator.append(accumulator[index -1] + float(hist[index]))
#
#     # Locate points to clip
#     maximum = accumulator[-1]
#     clip_hist_percent *= (maximum/100.0)
#     clip_hist_percent /= 2.0
#
#     # Locate left cut
#     minimum_gray = 0
#     while accumulator[minimum_gray] < clip_hist_percent:
#         minimum_gray += 1
#
#     # Locate right cut
#     maximum_gray = hist_size -1
#     while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
#         maximum_gray -= 1
#
#     # Calculate alpha and beta values
#     alpha = 255 / (maximum_gray - minimum_gray)
#     beta = -minimum_gray * alpha
#
#     '''
#     # Calculate new histogram with desired range and show histogram
#     new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
#     plt.plot(hist)
#     plt.plot(new_hist)
#     plt.xlim([0,256])
#     plt.show()
#     '''
#
#     auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
#     return (auto_result, alpha, beta)
#
# image = cv2.imread('test22.jpg')
# auto_result, alpha, beta = automatic_brightness_and_contrast(image)
# print('alpha', alpha)
# print('beta', beta)
# result = np.concatenate([image, auto_result], axis=1)
# cv2.namedWindow('result', 0)
# cv2.imshow('result', result)
# # cv2.imshow('auto_result', auto_result)
# cv2.waitKey()
# cv2.imwrite('E:/iron_IMG/picture/jiaogang/test/tt/out.jpg', auto_result)
