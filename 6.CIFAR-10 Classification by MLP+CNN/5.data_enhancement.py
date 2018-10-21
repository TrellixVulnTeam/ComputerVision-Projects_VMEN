import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint

'''
    跟增强前的CNN相比，在第4个步骤“切分数据集”之后加了：
        创建图像增强产生器,可视化增强图像;
        在训练模型时，使用跟增强匹配的model.fit_generator
'''

# 读取训练数据和测试数据
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)
# (50000, 32, 32, 3)

print(x_test.shape)
# (10000, 32, 32, 3)

# 归一化
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# 切分训练集、验证集、测试集


num_classes = len(np.unique(y_train))
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

(x_train, x_valid) = x_train[5000:], x_train[:5000]
(y_train, y_valid) = y_train[5000:], y_train[:5000]

print('x_train shape:', x_train.shape)
# x_train shape: (45000, 32, 32, 3)

print(x_train.shape[0], 'train examples')
# 45000 train examples

print(x_valid.shape[0], 'valid examples')
# 5000 valid examples

print(x_test.shape[0], 'test examples')
# 10000 test examples

# 创建和配置图像增强产生器
datagen_train = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True)

datagen_train.fit(x_train)

'''
keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-06,
    rotation_range=0.0,  # 随机旋转角度范围
    width_shift_range=0.0,  # 宽度移动范围
    height_shift_range=0.0,  # 高度移动范围
    brightness_range=None,  # 亮度范围
    shear_range=0.0,  # 剪切范围
    zoom_range=0.0,  # 缩放方位
    channel_shift_range=0.0,  # 通道转换范围
    fill_mode='nearest',  # 填充模式(4种，constant, nearest, wrap, reflection)
    cval=0.0,  # 当填充模式为constant时，填充的值
    horizontal_flip=False,  # 水平翻转
    vertical_flip=False,  # 垂直翻转
    rescale=None,  # 数据缩放
    preprocessing_function=None,  # 图像缩放、增强后使用
    data_format=None,  # 图像集格式，(samples,height,width,channels)还是channels在samples后
    validation_split=0.0)  # 数据集用来做验证集的比例
'''

# 取12张训练集图片
x_train_subset = x_train[:12]

# 可视化部分图片
fig = plt.figure(figsize=(20, 2))
for i in range(0, len(x_train_subset)):
    ax = fig.add_subplot(1, 12, i + 1)
    ax.imshow(x_train_subset[i])
fig.suptitle('Subset of Original Training Images', fontsize=20)
plt.show()

# 可视化数据增强后的图片
fig = plt.figure(figsize=(20, 2))
for x_batch in datagen_train.flow(x_train_subset, batch_size=12):
    for i in range(0, 12):
        ax = fig.add_subplot(1, 12, i + 1)
        ax.imshow(x_batch[i])
    fig.suptitle('Augmented Images', fontsize=20)
    plt.show()
    break

# 定义模型
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(32, 32, 3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(num_classes, activation='softmax'))

print(model.summary())

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# 训练模型
batch_size = 32
checkpoint = ModelCheckpoint(filepath='MLP.weights.best.hdf5', verbose=1, save_best_only=True)
model.fit_generator(datagen_train.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    epochs=100,
                    verbose=2,
                    callbacks=[checkpoint],
                    validation_data=(x_valid, y_valid),
                    validation_steps=x_valid.shape[0] // batch_size)

# 加载在验证集上分类正确率最高的一组模型参数
model.load_weights('MLP.weights.best.hdf5')

# 测试集上计算分类正确率
score = model.evaluate(x_test, y_test, verbose=0)
print('\n', 'Test accuracy:', score[1])
# 0.6744
