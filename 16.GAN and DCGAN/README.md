## GAN and DCGAN
## (GAN生成对抗网络和DCGAN深度卷积生成对抗网络)

### 项目背景
本节的程序来自于项目 https://github.com/carpedm20/DCGAN-tensorflow 。

### 代码流程
|名称|作用|
|:-------------|:-------------:|
|1.main|调用生成对抗网络训练的主函数|
|2.model|DCGAN网络结构实现|
|3.utils|附属图像处理工具函数包|
|4.download|MNIST手写数字数据集下载|

### 生成MNIST图像

下载MNIST数据集：
```
python download.py mnist
```

训练：
```
python main.py --dataset mnist --input_height=28 --output_height=28 --train
```

生成图像保存在samples文件夹中。

### 使用自己的数据集训练

在数据目录中已经准备好了一个动漫人物头像数据集faces.zip。在源代码的data目录中再新建一个anime目录（如果没有data 目录可以自行新建），并将faces.zip中所有的图像文件解压到anime目录中。

训练命令：
```
python main.py --input_height 96 --input_width 96 \
  --output_height 48 --output_width 48 \
  --dataset anime --crop -–train \
  --epoch 300 --input_fname_pattern "*.jpg"
```

生成图像保存在samples文件夹中。

### 效果图
#### ·生成网络生成的MNIST图像
<img width="500" height="500" src="./assets/mnist1.png"/>

#### ·生成网络生成的人脸数据图像
<img width="500" height="500" src="./assets/result_16_01_04_.png"/>

#### ·生成网络生成人脸数据训练过程
<img width="500" height="500" src="./assets/training.gif"/>