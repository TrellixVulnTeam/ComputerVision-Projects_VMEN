## 8.Transfer Learning for Dog Classifier by VGG16
## (迁移学习-基于VGG-16模型快速实现大数据集下狗狗的分类器的训练和分类)

### 项目背景
>  

背景：数据集8351张图，每张都是狗狗照片，共133种。

目标：用CNN实现狗类品种分类

方法：使用ImageNet上预先训练好的VGG16

分析场景：狗类数据集较小，与ImageNet相似度较高

将最后的全连接层删除，换成新的连接层。冻结前面模型的权重，只训练最后一个连接层的权重。

  
### 代码流程
|名称|作用|
|:-------------:|:-------------:|
|1.VGG16_model|加载VGG-16模型并下载内部对应参数|
|2.data_load|下载并加载5类花朵数据集|
|3.data_preprocess|将花朵数据通过VGG-16模型并保存模型输出结果|
|4.model_training|迁移VGG-16模型尾部衔接全连接层并加载上面结果数据训练|
|5.example_test|挑选一个数据集实例使用模型预测并输出预测概率分布|

### 效果图
#### ·目标预测实例花朵图像
<img width="500" height="400" src="./images/example.png"/>

#### ·模型预测概率分布统计
<img width="500" height="400" src="./images/prediction.png"/>
