import numpy as np
import cv2
from matplotlib import pyplot as plt

'''
    ORB(Oriented Fast and Rotated BRIEF)，结合Fast与Brief算法，并给Fast特征点增加了方向性，使得特征点具有旋转不变性，并提出了构造金字塔方法，解决尺度不变性，但文章中没有具体详述。
    
    特征提取是由FAST（Features from  Accelerated Segment Test）算法发展来的，特征点描述是根据BRIEF（Binary Robust Independent Elementary Features）特征描述算法改进的。
    ORB特征是将FAST特征点的检测方法与BRIEF特征描述子结合起来，并在它们原来的基础上做了改进与优化。
    ORB主要解决BRIEF描述子不具备旋转不变性的问题。
    实验证明，ORB远优于之前的SIFT与SURF算法，ORB算法的速度是sift的100倍，是surf的10倍。
'''

imgname1 = './images/test_SIFT1.png'
imgname2 = './images/test_SIFT2.png'

orb = cv2.ORB_create()

img1 = cv2.imread(imgname1)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 灰度处理图像
kp1, des1 = orb.detectAndCompute(img1, None)  # des是描述子

img2 = cv2.imread(imgname2)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
kp2, des2 = orb.detectAndCompute(img2, None)

hmerge = np.hstack((gray1, gray2))  # 水平拼接

img3 = cv2.drawKeypoints(img1, kp1, img1, color=(255, 0, 255))
img4 = cv2.drawKeypoints(img2, kp2, img2, color=(255, 0, 255))

hmerge = np.hstack((img3, img4))  # 水平拼接

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
plt.title('ORB', fontsize=15)
plt.show()
