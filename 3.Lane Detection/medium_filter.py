import numpy as np
from PIL import Image
import scipy.signal as signal

'''
    medium filter（中值滤波）
'''

# 读入图片并建立Image对象im
im = Image.open('./images/boy.jpg')
# 存储图像中所有像素值的list(二维)
data = []
# 将图片尺寸记录下来
width, height = im.size

# 读取图像像素的值
# 对每一行遍历
for h in range(height):
    # 记录每一行像素
    row = []
    # 对每行的每个像素列位置w
    for w in range(width):
        # 用getpixel读取这一点像素值
        value = im.getpixel((h, w))
        # 把它加到这一行的list中去
        row.append(value)
    # 把记录好的每一行加到data的子list中去，就建立了模拟的二维list
    data.append(row)

# 二维中值滤波
data = signal.medfilt(data, kernel_size=3)
# 转换为int类型，以使用快速二维滤波
data = np.int32(data)

# 创建并保存结果
for h in range(height):  # 对每一行
    for w in range(width):  # 对该行的每一个列号
        # 将data中该位置的值存进图像,要求参数为tuple
        im.putpixel((h, w), tuple(data[h][w]))

    # 存储图像
im.save('medium_filter.jpg')
