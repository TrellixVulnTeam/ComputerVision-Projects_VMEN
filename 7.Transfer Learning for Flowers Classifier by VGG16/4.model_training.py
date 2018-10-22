import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedShuffleSplit

with open('labels.csv') as f:
    reader = csv.reader(f, delimiter='\n')
    labels = np.array([each for each in reader if len(each) > 0]).squeeze()
    print(labels.shape)
    # (3670,)

codes = pd.read_csv('codes.csv', header=None)
codes = codes.as_matrix()
print(codes.shape)
# (3670, 4096)

# label转型onehot
lb = LabelBinarizer()

lb.fit(labels)

labels_vecs = lb.transform(labels)

# 拆分数据集
ss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

train_idx, val_idx = next(ss.split(codes, labels))

half_val_len = int(len(val_idx) / 2)

val_idx, test_idx = val_idx[:half_val_len], val_idx[half_val_len:]

# 获得三份数据集
train_x, train_y = codes[train_idx], labels_vecs[train_idx]
val_x, val_y = codes[val_idx], labels_vecs[val_idx]
test_x, test_y = codes[test_idx], labels_vecs[test_idx]

# 打印各自数据集尺寸
print("Train shapes (x, y):", train_x.shape, train_y.shape)
# Train shapes (x, y): (2936, 4096) (2936, 5)

print("Validation shapes (x, y):", val_x.shape, val_y.shape)
# Validation shapes (x, y): (367, 4096) (367, 5)

print("Test shapes (x, y):", test_x.shape, test_y.shape)
# Test shapes (x, y): (367, 4096) (367, 5)

# 在上述vgg的基础上，增加一个256个元素的全连接层，最后加上一个softmax层，计算交叉熵进行最后的分类。
inputs_ = tf.placeholder(tf.float32, shape=[None, codes.shape[1]])
labels_ = tf.placeholder(tf.int64, shape=[None, labels_vecs.shape[1]])

# 全连接层初始化
fc = tf.contrib.layers.fully_connected(inputs_, 256)

logits = tf.contrib.layers.fully_connected(fc, labels_vecs.shape[1], activation_fn=None)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels_, logits=logits)
cost = tf.reduce_mean(cross_entropy)

# 使用Adam优化器
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Softmax层输出结果
predicted = tf.nn.softmax(logits)
correct_pred = tf.equal(tf.argmax(predicted, 1), tf.argmax(labels_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# 获取batch函数
def get_batches(x, y, n_batches=10):
    """ Return a generator that yields batches from arrays x and y. """
    batch_size = len(x) // n_batches

    for ii in range(0, n_batches * batch_size, batch_size):
        # 如果不是最后一个batch，将数据整理成batch尺度
        if ii != (n_batches - 1) * batch_size:
            X, Y = x[ii: ii + batch_size], y[ii: ii + batch_size]
        # 如果是最后一个batch，剩余数据自然成组
        else:
            X, Y = x[ii:], y[ii:]
        yield X, Y


# 训练超参数自定义
epochs = 10
iteration = 0
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        for x, y in get_batches(train_x, train_y):
            feed = {inputs_: x,
                    labels_: y}
            loss, _ = sess.run([cost, optimizer], feed_dict=feed)
            print("Epoch: {}/{}".format(e + 1, epochs),
                  "Iteration: {}".format(iteration),
                  "Training loss: {:.5f}".format(loss))
            iteration += 1

            if iteration % 5 == 0:
                feed = {inputs_: val_x,
                        labels_: val_y}
                val_acc = sess.run(accuracy, feed_dict=feed)
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Validation Acc: {:.4f}".format(val_acc))
    saver.save(sess, "checkpoints/flowers.ckpt")

# 测试集验证判断效果
with tf.Session() as sess:
    # 加载上面训练好模型
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    feed = {inputs_: test_x,
            labels_: test_y}
    test_acc = sess.run(accuracy, feed_dict=feed)
    print("Test accuracy: {:.4f}".format(test_acc))
    # Test accuracy: 0.8638


