import os
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow_vgg_master import vgg16
from tensorflow_vgg_master import utils

# 数据目录
data_dir = 'flower_photos/'
contents = os.listdir(data_dir)
# 类别数据名称统计归纳，一会用于遍历
classes = [each for each in contents if os.path.isdir(data_dir + each)]
print(classes)
# ['roses', 'sunflowers', 'tulips', 'daisy', 'dandelion']

# 设置batch_size大小
batch_size = 10
codes_list = []
labels = []
batch = []

codes = None

with tf.Session() as sess:
    # 引入模型
    vgg = vgg16.Vgg16()
    # 设置输入承载
    input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])

    # 将输入格式导入模型
    with tf.name_scope("content_vgg"):
        vgg.build(input_)

    # 遍历所有类别
    for each in classes:
        # 开始某一数据处理
        print("Starting {} images".format(each))
        # 获取类别路径
        class_path = data_dir + each
        # 加载该类别下所有文件
        files = os.listdir(class_path)
        for ii, file in enumerate(files, 1):
            '''
                ii：就是遍历每个类别图片数据文件的名称
                file：是一个伴随的计数
            '''
            # 将图像加载到当前批次
            # 获取当前图片
            img = utils.load_image(os.path.join(class_path, file))
            # 图像放入当前批次并规范大小
            batch.append(img.reshape((1, 224, 224, 3)))
            # 记录对应标签
            labels.append(each)

            # 每10个图片成一个batch，将数据喂入模型获得输出结果数据
            if ii % batch_size == 0 or ii == len(files):  # 每10个一组或者末尾一批结束全部数据
                images = np.concatenate(batch)
                # print(images.shape)  # (10, 224, 224, 3)正好一共10张图像
                # 喂入之前设置好的input
                feed_dict = {input_: images}
                # 获得VGG16模型的输出
                codes_batch = sess.run(vgg.relu6, feed_dict=feed_dict)
                # print(codes_batch.shape)  # (10, 4096)正好是是10*4096的矩阵结果

                # 建立批次序列号
                if codes is None:
                    codes = codes_batch
                else:
                    codes = np.concatenate((codes, codes_batch))

                # 清空batch开始下一批处理
                batch = []
                print('{} images processed'.format(ii))
                # print(codes.shape)

codes = pd.DataFrame(codes)
codes.to_csv('./codes.csv', header=False, index=False)

labels = pd.DataFrame(labels)
labels.to_csv('./labels.csv', header=False, index=False)