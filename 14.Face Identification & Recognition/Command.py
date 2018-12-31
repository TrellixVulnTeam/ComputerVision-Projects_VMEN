# export PYTHONPATH=[...]/src

# 项目下运行
# export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/src

# 对LFW数据库进行人脸检测和对齐的方法命令
'''
python3 src/align/align_dataset_mtcnn.py \
  ~/Project/ComputerVision-Projects/14.Face\ Identification\ \&\ Recognition/dataset/lfw/raw  \
  ~/Project/ComputerVision-Projects/14.Face\ Identification\ \&\ Recognition/dataset/lfw/lfw_mtcnnpy_160 \
  --image_size 160 --margin 32 \
  --random_order
'''
'''
Output:
    Total number of images: 13233
    Number of successfully aligned images: 13233
'''

# 校验LFW数据库准确率
'''
python3 src/validate_on_lfw.py \
  ~/Project/ComputerVision-Projects/14.Face\ Identification\ \&\ Recognition/dataset/lfw/lfw_mtcnnpy_160 \
  ~/Project/ComputerVision-Projects/14.Face\ Identification\ \&\ Recognition/models/facenet/20170512-110547/
'''
'''
Output：
    Runnning forward pass on LFW images
    Accuracy: 0.992+-0.003
    Validation rate: 0.97467+-0.01477 @ FAR=0.00133
    Area Under Curve (AUC): 1.000
    Equal Error Rate (EER): 0.007
'''

# 计算人脸两两之间的距离
'''
python3 src/compare.py \
  ~/Project/ComputerVision-Projects/14.Face\ Identification\ \&\ Recognition/models/facenet/20170512-110547/ \
  ./test_imgs/1.jpg ./test_imgs/2.jpg ./test_imgs/3.jpg
'''
'''
Output：
    Images:
    0: ./test_imgs/1.jpg
    1: ./test_imgs/2.jpg
    2: ./test_imgs/3.jpg
    
    Distance matrix
            0         1         2     
    0    0.0000    0.7270    1.1283  
    1    0.7270    0.0000    1.0913  
    2    1.1283    1.0913    0.0000  
'''

# 用MTCNN进行检测和对齐CASIA数据集
'''
python3 src/align/align_dataset_mtcnn.py \
  ~/Project/ComputerVision-Projects/14.Face\ Identification\ \&\ Recognition/dataset/casia/raw/ \
  ~/Project/ComputerVision-Projects/14.Face\ Identification\ \&\ Recognition/dataset/casia/casia_maxpy_mtcnnpy_182 \
  --image_size 182 --margin 44
'''

# 重新进行训练新模型
'''
python3 src/train_softmax.py \
  --logs_base_dir ~/Project/ComputerVision-Projects/14.Face\ Identification\ \&\ Recognition/logs/facenet/ \
  --models_base_dir ~/Project/ComputerVision-Projects/14.Face\ Identification\ \&\ Recognition/models/facenet/ \
  --data_dir ~/Project/ComputerVision-Projects/14.Face\ Identification\ \&\ Recognition/dataset/casia/casia_maxpy_mtcnnpy_182 \
  --image_size 160 \
  --model_def models.inception_resnet_v1 \
  --lfw_dir ~/Project/ComputerVision-Projects/14.Face\ Identification\ \&\ Recognition/dataset/lfw/lfw_mtcnnpy_160 \
  --optimizer RMSPROP \
  --learning_rate -1 \
  --max_nrof_epochs 80 \
  --keep_probability 0.8 \
  --random_crop --random_flip \
  --learning_rate_schedule_file
  data/learning_rate_schedule_classifier_casia.txt \
  --weight_decay 5e-5 \
  --center_loss_factor 1e-2 \
  --center_loss_alfa 0.9
'''
