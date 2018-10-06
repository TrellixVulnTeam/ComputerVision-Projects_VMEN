import cv2

img = cv2.imread('simple.jpg', 0)

# 初始化corner detection算子
fast = cv2.FastFeatureDetector_create(threshold=25)

# 寻找并绘制关键点
kp = fast.detect(img, None)
img2 = cv2.drawKeypoints(img, kp, img, color=(0, 255, 0), flags=0)

# 输出所有模型参数
print("Threshold: ", fast.getThreshold())
print("nonmaxSuppression: ", fast.getNonmaxSuppression())
print("neighborhood: ", fast.getType())
print("Total Keypoints with nonmaxSuppression: ", len(kp))

cv2.imwrite('fast_true.png', img2)

# 解除非极大值抑制
fast.setNonmaxSuppression(0)
kp = fast.detect(img, None)

print("Total Keypoints without nonmaxSuppression: ", len(kp))

img3 = cv2.drawKeypoints(img, kp, None, color=(255, 0, 0))

cv2.imwrite('fast_false.png', img3)
