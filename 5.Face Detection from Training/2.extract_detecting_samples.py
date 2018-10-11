import cv2

# 创建window
cv2.namedWindow('video')

# 打开视频
cap = cv2.VideoCapture("./videos/face.mp4")

# 采样计数器
num = 1

while (cap.isOpened()):

    # ret表示返回的状态  frame存储着图像数据矩阵mat类型
    ret, frame = cap.read()

    # 如果img维度为3，说明不是灰度图，先转化为灰度图gray，如果不为3，也就是2，原图就是灰度图
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    if ret == True:
        # 翻转图片
        # rows, cols = gray.shape
        # M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 180, 1)
        # gray = cv2.warpAffine(gray, M, (cols, rows))

        # 显示视频
        gray = cv2.resize(gray, (930, 550), interpolation=cv2.INTER_CUBIC)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

    # 获得训练好的人脸的参数数据，这里直接使用GitHub上的默认值
    classifier = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')

    # 加载模型
    classifier.load('haarcascade_frontalface_default.xml')

    # 加载分类器
    faceRects = classifier.detectMultiScale(gray, 1.3, 5)

    print(faceRects)

    if faceRects != () and len(faceRects) == 1:
        # 获取脸部坐标数据
        (x, y, width, height) = faceRects[0]

        # 这里为每个捕捉到的图片进行命名，每个图片按数字递增命名。
        image_name = './training_data/%d.jpg' % num

        # 将当前帧含人脸部分保存为图片
        frame = cv2.resize(frame, (930, 550), interpolation=cv2.INTER_CUBIC)
        image = frame[(y - 5):(y + height + 5), (x - 5): (x + width + 5)]

        # 存储图片
        cv2.imwrite(image_name, image)

        num = num + 1

        cv2.rectangle(gray, (x, y), (x + width, y + height), (255, 0, 0), 2)

        cv2.imshow('video', gray)
