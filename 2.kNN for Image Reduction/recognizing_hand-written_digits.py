import matplotlib.pyplot as plt
from numpy import *
from os import listdir
from sklearn import datasets, svm, metrics

# 导入数据集
digits = datasets.load_digits()
print(digits.data)
'''
    [[ 0.  0.  5. ...  0.  0.  0.]
     [ 0.  0.  0. ... 10.  0.  0.]
     [ 0.  0.  0. ... 16.  9.  0.]
     ...
     [ 0.  0.  1. ...  6.  0.  0.]
     [ 0.  0.  2. ... 12.  0.  0.]
     [ 0.  0. 10. ... 12.  1.  0.]]
'''

# 加载指定数据集
dataset = datasets.fetch_mldata("MNIST Original", data_home='./')

# 提取数据集特征和标签
features = np.array(dataset.data, 'int16')
labels = np.array(dataset.target, 'int')
print(labels)  # [0 0 0 ..., 9 9 9]

print(digits.images[1])
'''
    array([[  0.,   0.,   0.,  12.,  13.,   5.,   0.,   0.],
           [  0.,   0.,   0.,  11.,  16.,   9.,   0.,   0.],
           [  0.,   0.,   3.,  15.,  16.,   6.,   0.,   0.],
           [  0.,   7.,  15.,  16.,  16.,   2.,   0.,   0.],
           [  0.,   0.,   1.,  16.,  16.,   3.,   0.,   0.],
           [  0.,   0.,   1.,  16.,  16.,   6.,   0.,   0.],
           [  0.,   0.,   1.,  16.,  16.,   6.,   0.,   0.],
           [  0.,   0.,   0.,  11.,  16.,  10.,   0.,   0.]])
'''

images_and_labels = list(zip(digits.images, digits.target))

# 显示前3张图片
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

# 将图片拉伸成一维矩阵
n_samples = len(digits.images)
print('number of sample:%s' % n_samples)  # number of sample:1797

data = digits.images.reshape((n_samples, -1))

# 生成SVM分类器
classifier = svm.SVC(gamma=0.001)

# 训练分类器
classifier.fit(data[:n_samples / 2], digits.target[:n_samples / 2])

# 预测分类
expected = digits.target[n_samples / 2:]
predicted = classifier.predict(data[n_samples / 2:])

# 打印分类器特征
print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
'''
    Classification report for classifier SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape=None, degree=3, gamma=0.001, kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False):
                 precision    recall  f1-score   support
    
              0       1.00      0.99      0.99        88
              1       0.99      0.97      0.98        91
              2       0.99      0.99      0.99        86
              3       0.98      0.87      0.92        91
              4       0.99      0.96      0.97        92
              5       0.95      0.97      0.96        91
              6       0.99      0.99      0.99        91
              7       0.96      0.99      0.97        89
              8       0.94      1.00      0.97        88
              9       0.93      0.98      0.95        92
    
    avg / total       0.97      0.97      0.97       899
'''

print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
'''
    Confusion matrix:   
    [[87  0  0  0  1  0  0  0  0  0]
     [ 0 88  1  0  0  0  0  0  1  1]
     [ 0  0 85  1  0  0  0  0  0  0]
     [ 0  0  0 79  0  3  0  4  5  0]
     [ 0  0  0  0 88  0  0  0  0  4]
     [ 0  0  0  0  0 88  1  0  0  2]
     [ 0  1  0  0  0  0 90  0  0  0]
     [ 0  0  0  0  0  1  0 88  0  0]
     [ 0  0  0  0  0  0  0  0 88  0]
     [ 0  0  0  1  0  1  0  0  0 90]]
'''

images_and_predictions = list(zip(digits.images[n_samples / 2:], predicted))

# 预测前4个图片样例
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()


# 定义分类函数
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


# 图片转向量函数
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


# 手写数据测试函数
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)

    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount / float(mTest)))


# 运行程序
handwritingClassTest()

'''
    the classifier came back with: 0, the real answer is: 0
    the classifier came back with: 0, the real answer is: 0
    the classifier came back with: 0, the real answer is: 0
    the classifier came back with: 0, the real answer is: 0
    the classifier came back with: 0, the real answer is: 0
    the classifier came back with: 1, the real answer is: 1
    the classifier came back with: 1, the real answer is: 1
    the classifier came back with: 1, the real answer is: 1
    the classifier came back with: 1, the real answer is: 1
    the classifier came back with: 1, the real answer is: 1
    the classifier came back with: 2, the real answer is: 2
    the classifier came back with: 2, the real answer is: 2
    the classifier came back with: 2, the real answer is: 2
    the classifier came back with: 2, the real answer is: 2
    the classifier came back with: 2, the real answer is: 2
    the classifier came back with: 3, the real answer is: 3
    the classifier came back with: 3, the real answer is: 3
    the classifier came back with: 3, the real answer is: 3
    the classifier came back with: 3, the real answer is: 3
    the classifier came back with: 3, the real answer is: 3
    the classifier came back with: 4, the real answer is: 4
    the classifier came back with: 4, the real answer is: 4
    the classifier came back with: 4, the real answer is: 4
    the classifier came back with: 4, the real answer is: 4
    the classifier came back with: 4, the real answer is: 4
    the classifier came back with: 5, the real answer is: 5
    the classifier came back with: 5, the real answer is: 5
    the classifier came back with: 5, the real answer is: 5
    the classifier came back with: 5, the real answer is: 5
    the classifier came back with: 5, the real answer is: 5
    the classifier came back with: 6, the real answer is: 6
    the classifier came back with: 6, the real answer is: 6
    the classifier came back with: 6, the real answer is: 6
    the classifier came back with: 6, the real answer is: 6
    the classifier came back with: 6, the real answer is: 6
    the classifier came back with: 7, the real answer is: 7
    the classifier came back with: 7, the real answer is: 7
    the classifier came back with: 7, the real answer is: 7
    the classifier came back with: 7, the real answer is: 7
    the classifier came back with: 7, the real answer is: 7
    the classifier came back with: 8, the real answer is: 8
    the classifier came back with: 8, the real answer is: 8
    the classifier came back with: 8, the real answer is: 8
    the classifier came back with: 8, the real answer is: 8
    the classifier came back with: 8, the real answer is: 8
    the classifier came back with: 9, the real answer is: 9
    the classifier came back with: 9, the real answer is: 9
    the classifier came back with: 9, the real answer is: 9
    the classifier came back with: 9, the real answer is: 9
    the classifier came back with: 9, the real answer is: 9
    
    the total number of errors is: 0
    
    the total error rate is: 0.000000
'''
