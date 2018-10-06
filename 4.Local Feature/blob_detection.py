from matplotlib import pyplot as plt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt
from skimage.color import rgb2gray

# 生成斑点图片
image = data.hubble_deep_field()[0:500, 0:500]
# 转成灰度图
image_gray = rgb2gray(image)
# 显示图片
plt.imshow(image)

# LoG算法获得斑点信息
blobs_log = blob_log(image_gray, max_sigma=30, num_sigma=10, threshold=.1)
print(blobs_log)  # 得到斑点坐标和高斯核的标准差
'''
    [[499.         435.           1.        ]
     [499.         386.           4.22222222]
     [499.         342.           1.        ]
     ...
     [  3.         196.           1.        ]
     [  3.         152.           1.        ]
     [  0.         305.           1.        ]]
'''

# 计算第三列半径
blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
print(blobs_log)
'''
    [[499.         435.           1.41421356]
     [499.         386.           5.97112393]
     [499.         342.           1.41421356]
     ...
     [  3.         196.           1.41421356]
     [  3.         152.           1.41421356]
     [  0.         305.           1.41421356]]
'''

# DoG算法获得斑点信息
blobs_dog = blob_dog(image_gray, max_sigma=30, threshold=.1)
# print(blobs_dog)

# 计算第三列半径
blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)
# print(blobs_dog)

# DoH算法获取斑点信息
blobs_doh = blob_doh(image_gray, max_sigma=30, threshold=.01)
# print(blobs_doh)

# 将斑点信息存入列表
blobs_list = [blobs_log, blobs_dog, blobs_doh]
# 分别对应黄色绿色红色
colors = ['yellow', 'lime', 'red']
# 分别对应三个标题
titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
          'Determinant of Hessian']
# 将结果信息组合
sequence = zip(blobs_list, colors, titles)

# 绘图
fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, subplot_kw={'adjustable': 'box-forced'})
axes = axes.ravel()
for blobs, color, title in sequence:
    ax = axes[0]
    axes = axes[1:]
    ax.set_title(title)
    ax.imshow(image, interpolation='nearest')
    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
        ax.add_patch(c)

plt.show()
