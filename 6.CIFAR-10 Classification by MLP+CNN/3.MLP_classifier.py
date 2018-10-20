import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10

'''
    尝试使用多层感知机MLP实现分类识别效果
'''

# 加载数据集
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape)
# (50000, 32, 32, 3)

print(x_test.shape)
# (10000, 32, 32, 3)

# 归一化数据集
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# 切分数据集
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

# 构建模型
model = Sequential()
model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

print(model.summary())
'''
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten_1 (Flatten)          (None, 3072)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 1000)              3073000   
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 1000)              0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 512)               512512    
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 512)               0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 10)                5130      
    =================================================================
    Total params: 3,590,642
    Trainable params: 3,590,642
    Non-trainable params: 0
    _________________________________________________________________
    None
'''

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# 训练模型
checkpoint = ModelCheckpoint(filepath='MLP.weights.best.hdf5', verbose=1, save_best_only=True)
hist = model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_valid, y_valid), callbacks=[checkpoint],
                 verbose=2, shuffle=True)

# 加载在验证集上分类正确率最高的一组模型参数
model.load_weights('MLP.weights.best.hdf5')

# 测试集上计算分类正确率
score = model.evaluate(x_test, y_test, verbose=0)
print('\n', 'Test accuracy:', score[1])
# 0.4157
