import cv2


# 识别人脸
def detectFaces(image_name):
    # 读取图片
    img = cv2.imread(image_name)
    # 获得训练好的人脸的参数数据，这里直接使用GitHub上的默认值
    face_cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
    # 加载模型
    face_cascade.load('haarcascade_frontalface_default.xml')
    # 如果img维度为3，说明不是灰度图，先转化为灰度图gray，如果不为3，也就是2，原图就是灰度图
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    # 探测人脸
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    result = []
    for (x, y, width, height) in faces:
        result.append((x, y, x + width, y + height))
    return result


def drawFaces(image_name):
    faces = detectFaces(image_name)
    if faces:
        img = cv2.imread(image_name)
        for (x1, y1, x2, y2) in faces:
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0))
        img_new_name = 'drawfaces_' + image_name
        cv2.imwrite(img_new_name, img)

    else:
        print("not found faces")


if __name__ == '__main__':
    # drawFaces('./images/test_group.jpg')
    drawFaces('face.JPG')
