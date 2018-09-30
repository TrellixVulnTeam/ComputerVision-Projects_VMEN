import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.datasets import load_sample_image
from sklearn.datasets.samples_generator import make_blobs

sns.set()

'''
    K-Means
'''

# 设置随机样例点
X, y = make_blobs(n_samples=300, centers=4, random_state=0, cluster_std=0.60)
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.show()

# 4中心聚类实现对上面样例数据的聚类
est = KMeans(4)
est.fit(X)
y_kmeans = est.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='rainbow')
plt.show()

'''
    手写数字应用
'''
# 加载数据
digits = load_digits()

# 加载模型
est = KMeans(n_clusters=10)
clusters = est.fit_predict(digits.data)
print(est.cluster_centers_.shape)  # (10, 64)

# 显示10个数字
fig = plt.figure(figsize=(8, 3))
for i in range(10):
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    ax.imshow(est.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
plt.show()

'''
    图片色彩压缩应用
'''
# 加载图片
china = load_sample_image("china.jpg")
plt.imshow(china)
plt.grid(False)
plt.show()

# 显示图片尺寸
print(china.shape)  # (427, 640, 3)

# 重整图片尺寸
X = (china / 255.0).reshape(-1, 3)
print(X.shape)  # (273280, 3)

# 降低图片尺寸加速实现效果
image = china[::3, ::3]
n_colors = 64

X = (image / 255.0).reshape(-1, 3)

model = KMeans(n_colors)
labels = model.fit_predict(X)
colors = model.cluster_centers_
new_image = colors[labels].reshape(image.shape)
new_image = (255 * new_image).astype(np.uint8)

# 对比色彩压缩图片
with sns.axes_style('white'):
    plt.figure()
    plt.imshow(image)
    plt.title('input')

    plt.figure()
    plt.imshow(new_image)
    plt.title('{0} colors'.format(n_colors))

    plt.show()