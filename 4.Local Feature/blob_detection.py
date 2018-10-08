from matplotlib import pyplot as plt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from math import sqrt
from skimage.color import rgb2gray

'''
    Laplacian of Gaussian (LoG)：
    这是速度最慢，可是最准确的一种算法。简单来说，就是对一幅图先进行一系列不同尺度的高斯滤波，然后对滤波后的图像做Laplacian运算。将全部的图像进行叠加。局部最大值就是所要检測的blob，这个算法对于大的blob检測会非常慢，还有就是该算法适合于检測暗背景下的亮blob。
    
    Difference of Gaussian (DoG)：
    这是LoG算法的一种高速近似，对图像进行高斯滤波之后，不做Laplacian运算，直接做减法。相减后的图做叠加。找到局部最大值，这个算法的缺陷与LoG相似。
    
    Determinant of Hessian (DoH)：
    这是最快的一种算法，不须要做多尺度的高斯滤波，运算速度自然提升非常多，这个算法对暗背景上的亮blob或者亮背景上的暗blob都能检測。
'''

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
titles = ['LoG', 'DoG', 'DoH']
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
