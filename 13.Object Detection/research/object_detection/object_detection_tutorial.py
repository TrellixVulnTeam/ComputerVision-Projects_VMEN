import os
import sys
import tarfile
import zipfile
import numpy as np
import tensorflow as tf
import six.moves.urllib as urllib
from PIL import Image
from io import StringIO
from collections import defaultdict
from matplotlib import pyplot as plt

sys.path.append("..")

from utils import label_map_util
from utils import visualization_utils as vis_util

'''
    Step 1：设置需要使用的模型
'''

# 需要下载的模型
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# 用来做目标检测的模型
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# 用来校对label的字符串列表
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

'''
    Step 2：下载预训练模型，也就是SSD+MobileNet模型
'''

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())

'''
    Step 3：下载模型后，程序直接将其读入默认的计算图中
'''

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

'''
    Step 4：进行真正检测之前，还需要定义一些辅助函数
'''

# 这部分代码是将神经网络检测到的index转换成类别名字符串
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# 辅助函数将图片转换成Numpy形式
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


'''
    Step 5：开始检测图片
'''

# 只检测两张图片
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3)]

# 定义输出图像大小
IMAGE_SIZE = (12, 8)

# 检测操作代码
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        # 定义检测图的输入输出张量
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # 每个box代表图中一个检测的物体
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # 每个score代表预测所属目标类别的信心度
        # score会和类别最终显示在图中
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        # class表示每个box框的类别
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        for image_path in TEST_IMAGE_PATHS:
            image = Image.open(image_path)
            # 将图像转换为Numpy格式
            image_np = load_image_into_numpy_array(image)
            # 将图像扩展一个维度，最后输入格式是[1,?,?,3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # 检测部分操作
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # 对得到的检测结果进行可视化
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8)
            plt.figure(figsize=IMAGE_SIZE)
            plt.imshow(image_np)
            plt.show()
