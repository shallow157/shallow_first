## 人脸采集模块

import cv2
import os
import numpy as np

# # 打开摄像头
# capture = cv2.VideoCapture(0)
# # 判断摄像头是否正常
# if capture.isOpened():
#    # 连续抓拍5张人脸照片
#    for i in range(5):
#        # 抓拍一张人脸照片
#        retval, frame = capture.read()
#        # 判断是否抓拍成功
#        if retval == True:
#            i = i + 1
#            print("正在抓拍第%d张人脸照片..." % i)
#            # 创建人脸识别器
#            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
#            # 检测人脸
#            faces = face_cascade.detectMultiScale(frame)
#             # 遍历所有人脸
#            for (x, y, w, h) in faces:
#                # 截取人脸区域
#                face_img = frame[y:y+h, x:x+w]
#                # 保存人脸照片
#                cv2.imwrite(os.path.join('faces', 'face_%d.jpg' % i), face_img)
#                # 绘制矩形框
#                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
#            # 显示图像
#            cv2.imshow('frame', frame)
#            # 等待1000ms
#            cv2.waitKey(1000)

#        else:
#            print("抓拍失败！")
#            exit()
#        # 释放摄像头
#        capture.release()
#        # 关闭所有窗口
#        cv2.destroyAllWindows()

# else:  
#     print("摄像头打开失败！")
#     exit()


# ## 训练人脸特征模型
# # 保存人脸
# myfaces = []
# # 保存标签
# labels = []
# labels_text = {"0":"ly"}

# # 遍历所有人脸照片
# for filename in os.listdir('faces'):
#     # 不是 。jpg就跳过
#     if not filename.endswith('.jpg'):
#         continue
#     # 读取人脸照片
#     img = cv2.imread(os.path.join('faces', filename))


import cv2
import os

# 创建保存人脸的文件夹
if not os.path.exists('faces'):
    os.makedirs('faces')

# 打开摄像头
capture = cv2.VideoCapture(0)
# 判断摄像头是否正常
if capture.isOpened():
    

    # 连续抓拍5张人脸照片
    for i in range(5):
        # 抓拍一张人脸照片
        retval, frame = capture.read()
        # 判断是否抓拍成功
        if retval == True:
            print("正在抓拍第%d张人脸照片..." % (i + 1))
            # 创建人脸识别器
            face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt2.xml')
            # 检测人脸
            faces = face_cascade.detectMultiScale(frame)
            # 遍历所有人脸
            for (x, y, w, h) in faces:
                # 截取人脸区域
                face_img = frame[y:y+h, x:x+w]
                # 保存人脸照片
                cv2.imwrite(os.path.join('faces', 'face_%d.jpg' % (i + 1)), face_img)
                # 绘制矩形框
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # 显示图像
            cv2.imshow('frame', frame)
            # 等待1000ms
            cv2.waitKey(1000)
        else:
            print("抓拍失败！第%d张照片" % (i + 1))
            exit()

    # 释放摄像头
    capture.release()
    # 关闭所有窗口
    cv2.destroyAllWindows()

else:  
    print("摄像头打开失败！")
    exit()

# === 使用采集到的人脸照片列表，训练特征模型
# ======================================================

# 读取多张脸
faces = []      # 用于保存人脸
labels = []     # 用于保存训练后的标签
labels_text = {"0": "shallow"}      # 标签对应的名称

# 将face目录下的jpg文件以灰度图的方式读取到图像列表中
for f in os.listdir("faces"):
    # 不是jpg文件就跳过
    if '.jpg' not in f:
        continue
    
    # 读取人脸图片到列表中，注意这里要使用灰度图
    faces.append(cv2.imread("faces\\" + f, cv2.IMREAD_GRAYSCALE))
    
    # 添加标签
    labels.append(0)

# 判断是否有人脸图像
if len(faces) > 0:
    # 创建识别器
    recognizer = cv2.face.LBPHFaceRecognizer().create()

    # 训练样本模型
    recognizer.train(faces, np.array(labels))

    # 保存训练模型
    recognizer.save("myface_model.yml")
    
    print("人脸特征训练完成，并保存到 .yml 文件中！")
    
    # 拿一张人脸来测试下模型，比如拿第一张人脸 
    # 与样本库去匹配
    label, confidence = recognizer.predict(faces[0])    # 第一张人脸

    # 打印匹配结果
    print("Name        ", labels_text[str(label)])
    print("Confidence  ", str(confidence))

    if confidence < 50:
        print("匹配成功！")   

else:
    print("没有人脸图片文件！")