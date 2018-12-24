from __future__ import print_function
import os
import PIL.Image
import scipy.misc
import numpy as np
from io import BytesIO
import tensorflow as tf
from functools import partial

graph = tf.Graph()
model_fn = 'tensorflow_inception_graph.pb'
sess = tf.InteractiveSession(graph=graph)
with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
t_input = tf.placeholder(np.float32, name='input')
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input - imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input': t_preprocessed})


# 将numpy.ndarry保存成文件的形式
def savearray(img_array, img_name):
    scipy.misc.toimage(img_array).save(img_name)
    print('img saved: %s' % img_name)


# 渲染函数
def render_naive(t_obj, img0, iter_n=20, step=1.0):
    # t_score是优化目标。它是t_obj的平均值，也就是是layer_output[:, :, :, channel]的平均值
    t_score = tf.reduce_mean(t_obj)
    # 计算t_score对t_input的梯度
    t_grad = tf.gradients(t_score, t_input)[0]

    # 复制原始噪声图像
    img = img0.copy()
    # 迭代训练
    for i in range(iter_n):
        # 在sess中计算梯度，以及当前的score
        g, score = sess.run([t_grad, t_score], {t_input: img})
        # 对img应用梯度。step可以看做“学习率”
        g /= g.std() + 1e-8
        img += g * step
        print('score(mean)=%f' % (score))
    # 保存图片
    savearray(img, 'naive.jpg')


# 定义卷积层、通道数，并取出对应的tensor
name = 'mixed4d_3x3_bottleneck_pre_relu'
channel = 139  # 选择144个通道中第139个通道做最大化处理
layer_output = graph.get_tensor_by_name("import/%s:0" % name)

# 定义原始的图像噪声形状为224*224*3
img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0

# 调用render_naive函数渲染
render_naive(layer_output[:, :, :, channel], img_noise, iter_n=50)
