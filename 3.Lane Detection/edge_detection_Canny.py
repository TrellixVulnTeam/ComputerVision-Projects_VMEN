import cv2

# Canny只能处理灰度图，所以将读取的图像转成灰度图
img = cv2.imread("./images/test.jpg", 0)

# 高斯平滑处理原图像降噪
img = cv2.GaussianBlur(img, (3, 3), 0)

# canny算法实现边缘检测
canny = cv2.Canny(img, 50, 150)

# 显示图片
cv2.imshow('Canny', canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
