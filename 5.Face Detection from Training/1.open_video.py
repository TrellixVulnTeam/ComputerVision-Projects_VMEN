import cv2

# 创建window
cv2.namedWindow('video')

# 打开视频
cap = cv2.VideoCapture("D:/Movies/face.avi")

while (cap.isOpened()):
    ret, frame = cap.read()  # ret表示返回的状态  frame存储着图像数据矩阵mat类型

    if ret == True:
        # 显示视频
        cv2.imshow('video', frame)
        # 退出选项
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# 释放资源
cap.release()

# 关闭窗口
cv2.destroyAllWindows()
