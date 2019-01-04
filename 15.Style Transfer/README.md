## Style Transfer
## (风格迁移图像处理技术)

### 使用预训练模型进行风格迁移

models中提供了7个预训练模型： wave.ckpt-done 、cubist.ckpt-done、denoised_starry.ckpt-done、mosaic.ckpt-done、scream.ckpt-done、feathers.ckpt-done。

以wave.ckpt-done的为例，运行下列指令
```
python eval.py --model_file models/wave.ckpt-done --image_file img/test.jpg
```

成功风格化的图像会被写到generated/res.jpg。

### 训练自己的风格迁移模型

|名称|作用|
|:-------------|:-------------:|
|model.py|图像生成网络|
|losses.py|图像损失网络|
|train.py|调用图像生成和损失网络训练并进行训练参数总结|

模型数据准备工作：

- 在地址http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz 下载VGG16模型，将下载到的压缩包解压后会得到一个vgg16.ckpt 文件。将vgg16.ckpt 复制到pretrained 文件夹中。最后的文件路径是pretrained/vgg16.ckpt。

- 在地址http://msvocds.blob.core.windows.net/coco2014/train2014.zip 下载COCO数据集。将该数据集解压后会得到一个train2014 文件夹，其中应该含有大量jpg 格式的图片。在项目中建立到这个文件夹的符号链接：
```
ln –s <到train2014 文件夹的路径> train2014
```

训练wave模型：
```
python train.py -c conf/wave.yml
```

打开TensorBoard：
```
tensorboard --logdir models/wave/
```

训练中保存的模型在文件夹models/wave/中。

### 效果图
#### ·原始图像
<img width="600" height="400" src="./img/test.jpg"/>

#### ·目标风格图像
<img width="600" height="400" src="./img/wave.jpg"/>

#### ·风格迁移效果图像
<img width="600" height="400" src="./img/results/wave.jpg"/>

#### ·目标风格图像
<img width="600" height="400" src="./img/udnie.jpg"/>

#### ·风格迁移效果图像
<img width="600" height="400" src="./img/results/udnie.jpg"/>

#### ·目标风格图像
<img width="600" height="400" src="./img/cubist.jpg"/>

#### ·风格迁移效果图像
<img width="600" height="400" src="./img/results/cubist.jpg"/>

#### ·目标风格图像
<img width="600" height="400" src="./img/denoised_starry.jpg"/>

#### ·风格迁移效果图像
<img width="600" height="400" src="./img/results/denoised_starry.jpg"/>

#### ·目标风格图像
<img width="600" height="400" src="./img/mosaic.jpg"/>

#### ·风格迁移效果图像
<img width="600" height="400" src="./img/results/mosaic.jpg"/>
