import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from keras.utils import np_utils
from sklearn.datasets import load_files


# 加载数据集函数
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets


# 加载训练集，验证集，和测试集
train_files, train_targets = load_dataset('./dogImages/train')
valid_files, valid_targets = load_dataset('./dogImages/valid')
test_files, test_targets = load_dataset('./dogImages/test')

# 加载所有狗品种名称
dog_names = [item[25:-1] for item in glob('./dogImages/train/*/')]

# 输出数据集统计数据
print('There are %d total dog categories.' % len(dog_names))
# There are 133 total dog categories.

print('There are %s total dog images.\n' % str(len(train_files) + len(valid_files) + len(test_files)))
# There are 8351 total dog images.

print('There are %d training dog images.' % len(train_files))
# There are 6680 training dog images.

print('There are %d validation dog images.' % len(valid_files))
# There are 835 validation dog images.

print('There are %d test dog images.' % len(test_files))
# There are 836 test dog images.


# 可视化部分图片
def visualize_img(img_path, ax):
    img = cv2.imread(img_path)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


fig = plt.figure(figsize=(20, 10))
for i in range(12):
    ax = fig.add_subplot(3, 4, i + 1, xticks=[], yticks=[])
    visualize_img(train_files[i], ax)

plt.show()
