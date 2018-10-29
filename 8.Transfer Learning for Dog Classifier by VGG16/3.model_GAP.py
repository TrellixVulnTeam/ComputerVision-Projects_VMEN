import numpy as np
from glob import glob
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.datasets import load_files
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D


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
model.add(GlobalAveragePooling2D(input_shape=(7, 7, 512)))
model.add(Dense(133, activation='softmax'))
model.summary()

print(model.summary())
'''

'''

# 模型编译
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 模型训练
checkpointer = ModelCheckpoint(filepath='dogvgg16.weights.best.hdf5', verbose=1,
                               save_best_only=True)
model.fit(train_vgg16, train_targets, epochs=20, validation_data=(valid_vgg16, valid_targets),
          callbacks=[checkpointer], verbose=1, shuffle=True)
'''
    ... ...
    ... ...
    ... ...
    4800/6680 [====================>.........] - ETA: 0s - loss: 6.2558 - acc: 0.6104
    5088/6680 [=====================>........] - ETA: 0s - loss: 6.2756 - acc: 0.6093
    5344/6680 [=======================>......] - ETA: 0s - loss: 6.2857 - acc: 0.6087
    5632/6680 [========================>.....] - ETA: 0s - loss: 6.3164 - acc: 0.6069
    5920/6680 [=========================>....] - ETA: 0s - loss: 6.3277 - acc: 0.6062
    6208/6680 [==========================>...] - ETA: 0s - loss: 6.3042 - acc: 0.6078
    6464/6680 [============================>.] - ETA: 0s - loss: 6.3039 - acc: 0.6078
    6680/6680 [==============================] - 1s 207us/step - loss: 6.3245 - acc: 0.6066 - val_loss: 7.0378 - val_acc: 0.4946
'''

# 加载最优模型
model.load_weights('dogvgg16.weights.best.hdf5')

# 测试
vgg16_predictions = [np.argmax(model.predict(np.expand_dims(feature, axis=0)))
                     for feature in test_vgg16]

# 计算准确率
test_accuracy = 100 * np.sum(np.array(vgg16_predictions) ==
                             np.argmax(test_targets, axis=1)) / len(vgg16_predictions)
print('\nTest accuracy: %.4f%%' % test_accuracy)
# Test accuracy: 48.0861%
