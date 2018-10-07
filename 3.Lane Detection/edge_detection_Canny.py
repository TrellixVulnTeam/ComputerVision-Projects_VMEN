import cv2
import matplotlib.pyplot as plt

'''
    Canny边缘检测算法可以分为以下5个步骤：
    
        1)使用高斯滤波器，以平滑图像，滤除噪声。
    
        2)计算图像中每个像素点的梯度强度和方向。
    
        3)应用非极大值（Non-Maximum Suppression）抑制，以消除边缘检测带来的杂散响应。
    
        4)应用双阈值（Double-Threshold）检测来确定真实的和潜在的边缘。
    
        5)通过抑制孤立的弱边缘最终完成边缘检测。
'''

# Canny只能处理灰度图，所以将读取的图像转成灰度图
img = cv2.imread("./images/test_man.jpg", 0)

# 高斯平滑处理原图像降噪
img = cv2.GaussianBlur(img, (3, 3), 0)

# canny算法实现边缘检测
canny = cv2.Canny(img, 50, 150)

# 显示图片
plt.subplot(1, 1, 1)
plt.imshow(canny)
plt.axis("off")
plt.show()
