import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def imconv(image_array, suanzi):
    '''计算卷积
        参数
        image_array 原灰度图像矩阵
        suanzi      算子
        返回
        原图像与算子卷积后的结果矩阵
    '''
    # 原图像矩阵的深拷贝
    image = image_array.copy()
    dim1, dim2 = image.shape
    # 对每个元素与算子进行乘积再求和(忽略最外圈边框像素)
    for i in range(1, dim1 - 1):
        for j in range(1, dim2 - 1):
            image[i, j] = (image_array[(i - 1):(i + 2), (j - 1):(j + 2)] * suanzi).sum()
    # 由于卷积后灰度值不一定在0-255之间，统一化成0-255
    image = image * (255.0 / image.max())
    # 返回结果矩阵
    return image


# x方向的Sobel算子
suanzi_x = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]])

# y方向的Sobel算子
suanzi_y = np.array([[-1, -2, -1],
                     [0, 0, 0],
                     [1, 2, 1]])

# 打开图像并转化成灰度图像
image = Image.open("./images/test_man.jpg").convert("L")

# 转化成图像矩阵
image_array = np.array(image)

# 得到x方向矩阵
image_x = imconv(image_array, suanzi_x)

# 得到y方向矩阵
image_y = imconv(image_array, suanzi_y)

# 得到梯度矩阵
image_xy = np.sqrt(image_x ** 2 + image_y ** 2)

# 梯度矩阵统一到0-255
image_xy = (255.0 / image_xy.max()) * image_xy

# 绘出图像
plt.subplot(2, 2, 1)
plt.imshow(image_array)
plt.axis("off")
plt.subplot(2, 2, 2)
plt.imshow(image_x)
plt.axis("off")
plt.subplot(2, 2, 3)
plt.imshow(image_y)
plt.axis("off")
plt.subplot(2, 2, 4)
plt.imshow(image_xy)
plt.axis("off")
plt.show()

'''
    OpenCV实现
'''
img = cv2.imread("./images/test_man.jpg", 0)

x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

# 转回uint8
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)

dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

# 绘出图像
plt.subplot(2, 2, 1)
plt.imshow(img)
plt.axis("off")
plt.subplot(2, 2, 2)
plt.imshow(absX)
plt.axis("off")
plt.subplot(2, 2, 3)
plt.imshow(absY)
plt.axis("off")
plt.subplot(2, 2, 4)
plt.imshow(dst)
plt.axis("off")
plt.show()