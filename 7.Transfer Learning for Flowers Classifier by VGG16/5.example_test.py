import csv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.ndimage import imread
from tensorflow_vgg_master import vgg16
from tensorflow_vgg_master import utils
from sklearn.preprocessing import LabelBinarizer

# 设置输入承载
inputs_ = tf.placeholder(tf.float32, shape=[None, 4096])
labels_ = tf.placeholder(tf.int64, shape=[None, 5])

# 全连接层初始化
fc = tf.contrib.layers.fully_connected(inputs_, 256)

logits = tf.contrib.layers.fully_connected(fc, 5, activation_fn=None)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels_, logits=logits)
cost = tf.reduce_mean(cross_entropy)

# 使用Adam优化器
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Softmax层输出结果
predicted = tf.nn.softmax(logits)
correct_pred = tf.equal(tf.argmax(predicted, 1), tf.argmax(labels_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 实例验证
test_img_path = 'flower_photos/roses/10894627425_ec76bbc757_n.jpg'
test_img = imread(test_img_path)
plt.imshow(test_img)
plt.show()

# 加载VGG16过一遍该图像数据
with tf.Session() as sess:
    input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16.Vgg16()
    vgg.build(input_)

with tf.Session() as sess:
    img = utils.load_image(test_img_path)
    img = img.reshape((1, 224, 224, 3))

    feed_dict = {input_: img}
    code = sess.run(vgg.relu6, feed_dict=feed_dict)

# 加载迁移学习训练模型，识别该图像
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, 'checkpoints/flowers.ckpt')
    feed = {inputs_: code}
    prediction = sess.run(predicted, feed_dict=feed).squeeze()
    print(prediction)
    # [1.4340951e-14 2.2546543e-15 9.9995875e-01 2.2978191e-13 4.1282965e-05]

# 显示预测概率分布图
plt.bar(['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'],
        [1.4340951e-14, 2.2546543e-15, 9.9995875e-01, 2.2978191e-13, 4.1282965e-05])
plt.show()
