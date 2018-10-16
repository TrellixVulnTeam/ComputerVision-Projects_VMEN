import cv2
import numpy as np
import tensorflow as tf

IMAGE_SIZE = 64

# 创建window
cv2.namedWindow('video')

# 打开视频
cap = cv2.VideoCapture("./videos/face.mp4")

# 采样计数器
num = 1

# 定义两个placeholder
x = tf.placeholder(tf.float32, [None, 64, 64, 3])
y = tf.placeholder(tf.float32, [None, 2])


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
    with tf.name_scope('b_conv1'):
        b_conv1 = bias_variable([32], name='b_conv1')  # 每一个卷积核一个偏置值
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
    with tf.name_scope('b_conv2'):
        b_conv2 = bias_variable([64], name='b_conv2')  # 每一个卷积核一个偏置值
    # 把h_pool1和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
    with tf.name_scope('conv2d_2'):
        conv2d_2 = conv2d(h_pool1, W_conv2) + b_conv2
    with tf.name_scope('relu'):
        h_conv2 = tf.nn.relu(conv2d_2)
    with tf.name_scope('h_pool2'):
        h_pool2 = max_pool(h_conv2)  # 进行max-pooling

# 全连接层
with tf.name_scope('fc1'):
    # 初始化第一个全连接层的权值
    with tf.name_scope('W_fc1'):
        W_fc1 = weight_variable([4 * 4 * 64, 1024], name='W_fc1')  # 输入层有4*4*64个列的属性，全连接层有1024个隐藏神经元
    with tf.name_scope('b_fc1'):
        b_fc1 = bias_variable([1024], name='b_fc1')  # 1024个节点
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
    with tf.name_scope('b_fc2'):
        b_fc2 = bias_variable([256], name='b_fc2')
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
    with tf.name_scope('b_fc3'):
        b_fc3 = bias_variable([2], name='b_fc3')
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

# 初始化变量
init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    # 读取训练模型
    saver.restore(sess, './net/CNN.ckpt')

    # 检测100帧
    for i in range(100):

        # 获取当前帧
        ret, frame = cap.read()

        if frame is None:
            break

        # 将当前帧标准化尺寸
        frame = cv2.resize(frame, (930, 550), interpolation=cv2.INTER_CUBIC)

        width = 210
        height = 210

        max_score = 0
        fit_x = 0
        fit_y = 0

        windows = list()

        # 滑窗扫描当前帧
        for window_x in range(0, 600, 10):
            for window_y in range(0, 300, 5):
                # 截取窗口
                image = frame[window_y:(window_y + height), window_x: (window_x + width)]
                # 标准化大小
                image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_CUBIC)
                # print(image.shape)  # (64, 64, 3)
                # 增加一个维度
                image = image[np.newaxis, :]
                # print(type(image))  # (1, 64, 64, 3)

                # 评估预测打分
                label = [[0, 1]]
                test_acc = sess.run(accuracy, feed_dict={x: image, y: label, keep_prob: 1.0})
                print("Class: " + str(test_acc))
                # 记录最优窗口
                if test_acc == 1.0:
                    windows.append([window_x, window_y])

        print(windows)

        for [window_x, window_y] in windows:
            cv2.rectangle(frame, (window_x, window_y), (window_x + width, window_y + height), (255, 0, 0), 2)

        cv2.imshow('video', frame)

        cv2.waitKey(0)
