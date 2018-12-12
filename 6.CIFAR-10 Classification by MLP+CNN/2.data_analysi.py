import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)
# (50000, 32, 32, 3)

print(x_test.shape)
# (10000, 32, 32, 3)

# 可视化前36幅图像
fig = plt.figure(figsize=(20, 5))

for i in range(36):
    ax = fig.add_subplot(3, 12, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_train[i]))

plt.show()
