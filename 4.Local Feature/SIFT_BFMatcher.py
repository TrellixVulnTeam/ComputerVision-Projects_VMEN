import numpy as np
import cv2
from matplotlib import pyplot as plt

'''
    基于BFmatcher的SIFT实现：

    BFmatcher（Brute-Force Matching）暴力匹配，应用BFMatcher.knnMatch( )函数来进行核心的匹配
    knnMatch（k-nearest neighbor classification）k近邻分类算法。 
    
    kNN算法则是从训练集中找到和新数据最接近的k条记录，然后根据他们的主要分类来决定新数据的类别。
    kNN方法在类别决策时，只与极少量的相邻样本有关。
    由于kNN方法主要靠周围有限的邻近的样本，而不是靠判别类域的方法来确定所属类别的，因此对于类域的交叉或重叠较多的待分样本集来说，kNN方法较其他方法更为适合。 
 经检验。
    BFmatcher在做匹配时会耗费大量的时间。 
'''

imgname1 = './images/test_SIFT1.png'
imgname2 = './images/test_SIFT2.png'

sift = cv2.xfeatures2d.SIFT_create()

img1 = cv2.imread(imgname1)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 灰度处理图像
kp1, des1 = sift.detectAndCompute(img1, None)  # des是描述子

img2 = cv2.imread(imgname2)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # 灰度处理图像
kp2, des2 = sift.detectAndCompute(img2, None)  # des是描述子

hmerge = np.hstack((gray1, gray2))  # 水平拼接
plt.subplot(111)
plt.imshow(hmerge)
plt.title('gray', fontsize=15)
plt.show()

img3 = cv2.drawKeypoints(img1, kp1, img1, color=(255, 0, 255))  # 画出特征点，并显示为红色圆圈
img4 = cv2.drawKeypoints(img2, kp2, img2, color=(255, 0, 255))  # 画出特征点，并显示为红色圆圈
hmerge = np.hstack((img3, img4))  # 水平拼接
plt.subplot(111)
plt.imshow(hmerge)
plt.title('point', fontsize=15)
plt.show()

# BFMatcher解决匹配
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
# 调整ratio
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])

img5 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
plt.subplot(111)
plt.imshow(img5)
plt.title('BFmatch', fontsize=15)
plt.show()
