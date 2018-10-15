import numpy as np
import random
import cv2
import os
import tensorflow as tf

'''
    数据预处理
'''
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
    # random.shuffle(imgs)
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


# 重新规定图片尺寸并补偿的函数
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


# onehot函数
def onehot(labels, length):
    sess = tf.Session()
    batch_size = tf.size(labels)
    labels = tf.expand_dims(labels, 1)
    indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
    concated = tf.concat([indices, labels], 1)
    onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, length]), 1.0, 0.0)
    return onehot_labels


# 读取目录下数据集
images, labels = resize_pic('./data/origin_sample')

# 弹出3个元素，凑个整，不然不好做batch，麻烦理解一下
for i in range(3):
    images.pop(1)
    labels.pop(1)

# 一共280个正负样本
print(len(images))  # 280
print(len(labels))  # 280

# 转换图片格式，images尺寸为 ( 图片数量 * IMAGE_SIZE * IMAGE_SIZE * 3 )
images = np.array(images)
# 转型成功
print(images.shape)  # (280, 64, 64, 3)

# 转换label为onehot型
labels = onehot(labels, 2)
print(labels.shape)  # (280, 2)

'''
    搭建训练网络
'''
# 每个批次的大小
batch_size = 35

# 计算一共有多少个训练批次遍历一次训练集
n_batch = 280 // batch_size  # 8


# 参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)  # 平均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)  # 标准差
        tf.summary.scalar('max', tf.reduce_max(var))  # 最大值
        tf.summary.scalar('min', tf.reduce_min(var))  # 最小值
        tf.summary.histogram('histogram', var)  # 直方图


# 初始化权值
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)  # 生成一个截断的正态分布，标准差为0.1
    return tf.Variable(initial, name=name)


# 初始化偏置
def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


# 卷积层
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='VALID')


# 池化层
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


# 输入层
with tf.name_scope('input'):
    # 定义两个placeholder
    x = tf.placeholder(tf.float32, [None, 64, 64, 3], name='x-input')
    y = tf.placeholder(tf.float32, [None, 2], name='y-input')

# 第一层：卷积+激活+池化
with tf.name_scope('Conv1'):
    # 初始化第一层的W和b
    with tf.name_scope('W_conv1'):
        W_conv1 = weight_variable([2, 2, 3, 32], name='W_conv1')  # 2*2的采样窗口，32个卷积核从3通道平面抽取特征
        variable_summaries(W_conv1)
    with tf.name_scope('b_conv1'):
        b_conv1 = bias_variable([32], name='b_conv1')  # 每一个卷积核一个偏置值
        variable_summaries(b_conv1)

    # 把x和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
    with tf.name_scope('conv2d_1'):
        conv2d_1 = conv2d(x, W_conv1) + b_conv1
    with tf.name_scope('relu'):
        h_conv1 = tf.nn.relu(conv2d_1)
    with tf.name_scope('h_pool1'):
        h_pool1 = max_pool(h_conv1)  # 进行max-pooling

# 第二层：卷积+激活+池化
with tf.name_scope('Conv2'):
    # 初始化第二个卷积层的权值和偏置
    with tf.name_scope('W_conv2'):
        W_conv2 = weight_variable([2, 2, 32, 64], name='W_conv2')  # 2*2的采样窗口，64个卷积核从32个平面抽取特征
        variable_summaries(W_conv2)
    with tf.name_scope('b_conv2'):
        b_conv2 = bias_variable([64], name='b_conv2')  # 每一个卷积核一个偏置值
        variable_summaries(b_conv2)

    # 把h_pool1和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
    with tf.name_scope('conv2d_2'):
        conv2d_2 = conv2d(h_pool1, W_conv2) + b_conv2
    with tf.name_scope('relu'):
        h_conv2 = tf.nn.relu(conv2d_2)
    with tf.name_scope('h_pool2'):
        h_pool2 = max_pool(h_conv2)  # 进行max-pooling

# 64*64的图片第一次卷积后还是32*32，第一次池化后变为16*16
# 第二次卷积后为8*8，第二次池化后变为了4*4
# 进过上面操作后得到64张4*4的平面

# 全连接层
with tf.name_scope('fc1'):
    # 初始化第一个全连接层的权值
    with tf.name_scope('W_fc1'):
        W_fc1 = weight_variable([4 * 4 * 64, 1024], name='W_fc1')  # 输入层有4*4*64个列的属性，全连接层有1024个隐藏神经元
        variable_summaries(W_fc1)
    with tf.name_scope('b_fc1'):
        b_fc1 = bias_variable([1024], name='b_fc1')  # 1024个节点
        variable_summaries(b_fc1)

    # 把第二层的输出扁平化为1维，-1代表任意值
    with tf.name_scope('h_pool2_flat'):
        h_pool2_flat = tf.reshape(h_pool2, [-1, 4 * 4 * 64], name='h_pool2_flat')
    # 求第一个全连接层的输出
    with tf.name_scope('wx_plus_b1'):
        wx_plus_b1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
    with tf.name_scope('relu'):
        h_fc1 = tf.nn.relu(wx_plus_b1)

    # Dropout处理，keep_prob用来表示处于激活状态的神经元比例
    with tf.name_scope('keep_prob'):
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    with tf.name_scope('h_fc1_drop'):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob, name='h_fc1_drop')

# 全连接层
with tf.name_scope('fc2'):
    # 初始化第二个全连接层
    with tf.name_scope('W_fc2'):
        W_fc2 = weight_variable([1024, 256], name='W_fc2')  # 输入为1024个隐藏层神经元，输出层为256
        variable_summaries(W_fc2)
    with tf.name_scope('b_fc2'):
        b_fc2 = bias_variable([256], name='b_fc2')
        variable_summaries(b_fc2)
    with tf.name_scope('wx_plus_b2'):
        wx_plus_b2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
    with tf.name_scope('softmax'):
        # 计算输出
        h_fc2 = tf.nn.relu(wx_plus_b2)

    # Dropout处理，keep_prob用来表示处于激活状态的神经元比例
    with tf.name_scope('h_fc1_drop'):
        h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob, name='h_fc2_drop')

# 全连接层
with tf.name_scope('fc3'):
    # 初始化第二个全连接层
    with tf.name_scope('W_fc3'):
        W_fc3 = weight_variable([256, 2], name='W_fc3')  # 输入为256个隐藏层神经元，输出层为2种结果
        variable_summaries(W_fc3)
    with tf.name_scope('b_fc3'):
        b_fc3 = bias_variable([2], name='b_fc3')
        variable_summaries(b_fc3)
    with tf.name_scope('wx_plus_b3'):
        wx_plus_b3 = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
    with tf.name_scope('softmax'):
        # 计算输出
        prediction = tf.nn.softmax(wx_plus_b3)

# 交叉熵代价函数
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=prediction),
                                   name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

# 使用AdamOptimizer进行优化
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 求准确率
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        # 结果存放在一个布尔列表中
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))  # argmax返回一维张量中最大的值所在的位置
    with tf.name_scope('accuracy'):
        # 求准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

data_train = images
data_labels = labels

print(data_train.shape)  # (280, 64, 64, 3)
print(data_labels.shape)  # (280, 2)

print(type(data_train))  # <class 'numpy.ndarray'>
print(type(data_labels))  # <class 'tensorflow.python.framework.ops.Tensor'>

# 初始化变量
init = tf.global_variables_initializer()

# 合并所有的Summary
merge = tf.summary.merge_all()

# 训练模型存储
saver = tf.train.Saver()


# 获取35个280以内的随机数
def get_random_35():
    random_35 = []
    while (len(random_35) < 35):
        x = random.randint(0, 279)
        if x not in random_35:
            random_35.append(x)
    return random_35

with tf.Session() as sess:
    sess.run(init)
    # 将labels转为numpy型
    labels = labels.eval(session=sess)
    data_labels = data_labels.eval(session=sess)
    # 将图写入制定目录
    writer = tf.summary.FileWriter('./logs/', sess.graph)
    for i in range(100):
        for batch in range(n_batch):
            # 训练模型
            random_35 = get_random_35()
            batch_xs = data_train[random_35]
            batch_ys = data_labels[random_35]
            summary, _ = sess.run([merge, train_step],
                                  feed_dict={x: batch_xs, y: batch_ys, keep_prob: 0.7})  # dropout比例
        writer.add_summary(summary, i)
        test_acc = sess.run(accuracy, feed_dict={x: images, y: labels, keep_prob: 1.0})
        print("Training Iters：" + str(i) + " , Testing Accuracy = " + str(test_acc))
    # 保存模型
    saver.save(sess, './net/CNN.ckpt')
