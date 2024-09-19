import cv2
import os
import numpy as np

## 训练特征模型
# 读取多张脸
# 用于保存人脸
faces = []
# 用于保存训练后的标签      
labels = []
# 标签对应的名称     
labels_text = {"0": "yangmi"}      

# 将目录下的jpg文件以灰度图的方式读取到图像列表中
for f in os.listdir("yangmi"):
    # 不是jpg文件就跳过
    if '.jpg' not in f:
        continue
    
    # 读取人脸图片到列表中，注意这里要使用灰度图
    faces.append(cv2.imread("yangmi\\" + f, cv2.IMREAD_GRAYSCALE))
    
    # 添加标签
    labels.append(0)

# 判断是否有人脸图像
if len(faces) > 0:
    # 创建识别器
    recognizer = cv2.face.LBPHFaceRecognizer().create()

    # 训练样本模型
    recognizer.train(faces, np.array(labels))

    # 保存训练模型
    recognizer.save("yangmi_face_model.yml")
    
    print("人脸特征训练完成，并保存到 .yml 文件中！")
    
    # 拿一张人脸来测试下模型
    # 与样本库去匹配
    label, confidence = recognizer.predict(faces[3])    

    # 打印匹配结果
    print("Name        ", labels_text[str(label)])
    print("Confidence  ", str(confidence))

    if confidence < 50:
        print("匹配成功！")   

else:
    print("没有人脸图片文件！")