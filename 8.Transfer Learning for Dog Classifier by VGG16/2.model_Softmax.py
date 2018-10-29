import numpy as np
from glob import glob
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.datasets import load_files
from keras.callbacks import ModelCheckpoint


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

# 读入VGG16瓶颈特征
bottleneck_features = np.load('./DogVGG16Data.npz')
train_vgg16 = bottleneck_features['train']
valid_vgg16 = bottleneck_features['valid']
test_vgg16 = bottleneck_features['test']

model = Sequential()
model.add(Flatten(input_shape=(7, 7, 512)))
model.add(Dense(133, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
              metrics=['accuracy'])

print(model.summary())
'''
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten_1 (Flatten)          (None, 25088)             0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 133)               3336837   
    =================================================================
    Total params: 3,336,837
    Trainable params: 3,336,837
    Non-trainable params: 0
    _________________________________________________________________
    None
'''

# 模型编译
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 模型训练
checkpointer = ModelCheckpoint(filepath='dogvgg16.weights.best.hdf5', verbose=1,
                               save_best_only=True)
model.fit(train_vgg16, train_targets, epochs=20, validation_data=(valid_vgg16, valid_targets),
          callbacks=[checkpointer], verbose=1, shuffle=True)

# 加载最优模型
model.load_weights('dogvgg16.weights.best.hdf5')

# 测试
vgg16_predictions = [np.argmax(model.predict(np.expand_dims(feature, axis=0)))
                     for feature in test_vgg16]

# 计算准确率
test_accuracy = 100 * np.sum(np.array(vgg16_predictions) ==
                             np.argmax(test_targets, axis=1)) / len(vgg16_predictions)
print('\nTest accuracy: %.4f%%' % test_accuracy)
