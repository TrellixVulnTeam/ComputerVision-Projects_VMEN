## 9.MNIST ALL IN
## (MNIST数据集基础解析及操作)

### 项目背景
背景：数据集60000张图，每张都是手写数字0~9，共10种

目标：用Softmax回归和CNN实现手写数字分类

### 数据集
[1.MNIST图像数据集](http://yann.lecun.com/exdb/mnist/)

### 代码流程
|名称|作用|
|:-------------|:-------------:|
|1.download|数据集下载|
|2.save_pic|解析数据集图片并保存|
|3.label|查看数据label|
|4.softmax_regression|Softmax回归实现手写数字识别分类|
|5.convolutional|卷积网络实现手写数字识别分类|

