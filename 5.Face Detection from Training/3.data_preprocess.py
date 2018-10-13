import numpy as np
import random
import cv2
import os

IMAGE_SIZE = 64


# 重新规范人脸数据
def resize_pic(path):
    imgs = list()
    images = list()
    labels = list()
    # 遍历所有样本图片
    for img in os.listdir(path):
        # 名称存入一个list
        imgs.append(img)
    # 打乱正负样本顺序
    random.shuffle(imgs)
    # 遍历所有的图片
    for img in imgs:
        if img.endswith('.jpg'):
            # 如果格式是图片，则进行大小处理
            image = cv2.imread(path + '/' + img)
            image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
            # 将图片加入到队列
            images.append(image)
            # 判断是否为人脸样本
            name = img.split('.')
            if name[0].endswith('#'):
                # 如果是负样本
                labels.append(0)
            else:
                labels.append(1)
    return images, labels


# 判断path_name文件夹下是否有son_path_name的函数
def file_exit(path_name, son_path_name):
    lists = os.listdir(path_name)
    # 该目录下的所有文件夹
    for list in lists:
        # 遍历所有文件，如果存在与son_path_name同名的文件夹，返回1即找到测试集文件
        if list == son_path_name:
            print('file exits')
            return 1
    return 0


# 重新规定图片尺寸函数
def resize_image(image, height=IMAGE_SIZE, width=IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)
    # 获得图像尺寸
    h, w, _ = image.shape
    # 找到最长的一边
    longest_edge = max(h, w)
    # 计算需要补充的像素
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass
    black = [0, 0, 0]
    # 给图像增加边界，是图片长、宽等长
    # cv2.BORDER_CONSTANT指定边界颜色
    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=black)
    # 返回调整之后的图像
    return cv2.resize(constant, (height, width))


# 读取目录下数据集
images, labels = resize_pic('./data/origin_sample')

# 一共283个正负样本
print(len(images))  # 283
print(len(labels))  # 283

# 转换图片格式，images尺寸为 ( 图片数量 * IMAGE_SIZE * IMAGE_SIZE * 3 )
images = np.array(images)

# 转型成功
print(images.shape)  # (283, 64, 64, 3)
