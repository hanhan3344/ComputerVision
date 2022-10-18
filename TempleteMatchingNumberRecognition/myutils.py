import cv2

def sort_contours(cnts, method = 'left-to-right'):
    reverse = False
    i = 0

    if method == 'right-to-left' or method == 'bottom-to-top':
        reverse = True
    
    if method == 'top-to-bottom' or method == 'bottom-to-top':
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key = lambda b: b[1][i], reverse=reverse))

    return cnts, boundingBoxes

#定义按比例放大缩小图像的函数
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)        #计算出现在的高与原来的高的比
        dim = (int(w * r), height)    #得到对应图像宽高
    else:
        r = width / float(w)             #计算出现在的宽与原来的宽的比
        dim = (width, int(h * r))        #得到对应图像宽高
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized
