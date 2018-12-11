import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# 读取mnist数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 看前20张训练图片的label
for i in range(20):
    # 得到one-hot表示，形如(0, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    one_hot_label = mnist.train.labels[i, :]
    # 通过np.argmax我们可以直接获得原始的label
    label = np.argmax(one_hot_label)
    print('mnist_train_%d.jpg label: %d' % (i, label))

'''
    mnist_train_0.jpg label: 7
    mnist_train_1.jpg label: 3
    mnist_train_2.jpg label: 4
    mnist_train_3.jpg label: 6
    mnist_train_4.jpg label: 1
    mnist_train_5.jpg label: 8
    mnist_train_6.jpg label: 1
    mnist_train_7.jpg label: 0
    mnist_train_8.jpg label: 9
    mnist_train_9.jpg label: 8
    mnist_train_10.jpg label: 0
    mnist_train_11.jpg label: 3
    mnist_train_12.jpg label: 1
    mnist_train_13.jpg label: 2
    mnist_train_14.jpg label: 7
    mnist_train_15.jpg label: 0
    mnist_train_16.jpg label: 2
    mnist_train_17.jpg label: 9
    mnist_train_18.jpg label: 6
    mnist_train_19.jpg label: 0
'''