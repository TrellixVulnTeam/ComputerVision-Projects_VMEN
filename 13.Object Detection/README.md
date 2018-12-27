## Object Detection
## (目标检测项目)

### 项目背景
>  本项目在学习了R-CNN，Fast R-CNN，以及Faster R-CNN三个主流基础的目标检测算法后，尝试调用TensorFlow Object Detection API查看已有训练模型的效果。分别直接调用SSD+MobileNet模型实现目标检测效果，以及在VOC2012数据集上训练新的Faster-R-CNN+Inception_ResNet_V2模型，调整模型和训练细节，实现目标检测效果。


### 代码流程
|名称|作用|
|:-------------|:-------------:|
|slim|TensorFlow目标检测API是基于Slim实现|
|Command|加载API中SSD+MobileNet模型实现目标检测流程指令|
|object_detection_tutorial|执行已经训练好的模型实现目标检测效果|

### 效果图
#### ·加载API中SSD+MobileNet模型实现目标检测效果
<img width="500" height="400" src="./figures/tutor_1.jpg"/>

#### ·加载API中SSD+MobileNet模型实现目标检测效果
<img width="500" height="400" src="./figures/tutor_2.jpg"/>

### 训练新模型
先在地址http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar 下载VOC2012数据集并解压。

在项目的object_detection文件夹中新建voc目录，并将解压后的数据集拷贝进来，最终形成的目录为：

```
research/
  object_detection/
    voc/
      VOCdevkit/
        VOC2012/
          JPEGImages/
            2007_000027.jpg
            2007_000032.jpg
            2007_000033.jpg
            2007_000039.jpg
            2007_000042.jpg
            ………………
          Annotations/
            2007_000027.xml
            2007_000032.xml
            2007_000033.xml
            2007_000039.xml
            2007_000042.xml
            ………………
          ………………
```

在object_detection目录中执行如下命令将数据集转换为tfrecord：

```
python3 create_pascal_tf_record.py --data_dir voc/VOCdevkit/ --year=VOC2012 --set=train --output_path=voc/pascal_train.record
python3 create_pascal_tf_record.py --data_dir voc/VOCdevkit/ --year=VOC2012 --set=val --output_path=voc/pascal_val.record
```

此外，将pascal_label_map.pbtxt 数据复制到voc 文件夹下：
```
cp data/pascal_label_map.pbtxt voc/
```

下载模型文件http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017.tar.gz 并解压，解压后得到frozen_inference_graph.pb 、graph.pbtxt 、model.ckpt.data-00000-of-00001 、model.ckpt.index、model.ckpt.meta 5 个文件。在voc文件夹中新建一个
pretrained 文件夹，并将这5个文件复制进去。

复制一份config文件：
```
cp samples/configs/faster_rcnn_inception_resnet_v2_atrous_pets.config \
  voc/voc.config
```

并在voc/voc.config中修改7处需要重新配置的地方。

训练模型的命令：
```
python train.py --train_dir voc/train_dir/ --pipeline_config_path voc/voc.config
```

使用TensorBoard：
```
tensorboard --logdir voc/train_dir/
```

导出模型并预测单张图片

运行(需要根据voc/train_dir/里实际保存的checkpoint，将1582改为合适的数值)：
```
python export_inference_graph.py \
  --input_type image_tensor \
  --pipeline_config_path voc/voc.config \
  --trained_checkpoint_prefix voc/train_dir/model.ckpt-1582
  --output_directory voc/export/
```

导出的模型是voc/export/frozen_inference_graph.pb 文件。