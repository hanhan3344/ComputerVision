from imutils import contours
import numpy as np
import imutils
import cv2
import myutils

def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    img = cv2.imread('HeiTiTemplete.png')
    cv_show('img', img)
    # 灰度图
    ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv_show('ref', ref)
    # 二值图像
    ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
    cv_show('ref', ref)
    # 计算轮廓
    refCnts, hierarchy = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img, refCnts, -1, (0, 0, 255), 3)
    cv_show('img', img)
    # print(np.array(refCnts).shape)

    refCnts = myutils.sort_contours(refCnts, method = 'left-to-right')[0]
    digits = {}

    for (i, c) in enumerate(refCnts):
        (x, y, w, h) = cv2.boundingRect(c)
        roi = ref[y: y+h, x: x+w]
        roi = cv2.resize(roi, (57, 88))

        digits[i] = roi

    # 初始化卷积核
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # 读取输入图像，预处理
    image = cv2.imread('after_rotated_rgb.png')
    cv_show('image', image)
    image = myutils.resize(image, width = 300)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv_show('gray', gray)

    # 礼帽操作，突出更明亮的区域
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
    cv_show('tophat', tophat)