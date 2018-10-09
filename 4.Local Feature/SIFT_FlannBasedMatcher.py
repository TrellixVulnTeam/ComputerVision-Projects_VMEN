import numpy as np
import cv2
from matplotlib import pyplot as plt

'''
    FLANN(Fast_Library_for_Approximate_Nearest_Neighbors)快速最近邻搜索包，它是一个对大数据集和高维特征进行最近邻搜索的算法的集合,而且这些算法都已经被优化过了。
    在面对大数据集时它的效果要好于BFMatcher。 
    经验证，FLANN比其他的最近邻搜索软件快10倍。
    使用FLANN匹配,我们需要传入两个字典作为参数。这两个用来确定要使用的算法和其他相关参数等。 
'''

imgname1 = './images/test_SIFT1.png'
imgname2 = './images/test_SIFT2.png'

sift = cv2.xfeatures2d.SIFT_create()

# FLANN 参数设计
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

img1 = cv2.imread(imgname1)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 灰度处理图像
kp1, des1 = sift.detectAndCompute(img1, None)  # des是描述子

img2 = cv2.imread(imgname2)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
kp2, des2 = sift.detectAndCompute(img2, None)

hmerge = np.hstack((gray1, gray2))  # 水平拼接

img3 = cv2.drawKeypoints(img1, kp1, img1, color=(255, 0, 255))
img4 = cv2.drawKeypoints(img2, kp2, img2, color=(255, 0, 255))

hmerge = np.hstack((img3, img4))  # 水平拼接

matches = flann.knnMatch(des1, des2, k=2)
matchesMask = [[0, 0] for i in range(len(matches))]

good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append([m])

img5 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
plt.subplot(111)
plt.imshow(img5)
plt.title('FLANN', fontsize=15)
plt.show()
