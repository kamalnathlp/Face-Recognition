import cv2
import os
import numpy as np

import faceRecognition as fr

test_img = cv2.imread('Input\\3.jpg')
faces_detected, gray_img = fr.faceDetection(test_img)
if len(faces_detected) ==0:
    print("No face Detected")
else:
    print("FaceDetected")


for(x,y,w,h) in faces_detected:
     cv2.rectangle(test_img, (x,y), (x+w, y+h), (255,0,0), thickness = 10)
#
# resized_img = cv2.resize(test_img, (750,700))
# cv2.imshow("face-Detected",resized_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows




# faces,faceId = fr.labels_for_training_data("test-img")
# face_recognizer = fr.train_classifier(faces,faceId)
# face_recognizer.save('trainingData.xml')


face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingData.xml')

name={ 1:"Santhosh",2:'Sasi',3:'Dharun',4:"Dharun"}

for face in faces_detected:
    x,y,w,h = face
    roi_gray = gray_img[y:y+h,x:x+h]
    label,confidence = face_recognizer.predict(roi_gray)
    print("Confidence {} \n Label: {}".format(confidence,label))
    fr.draw_rect(test_img,face)
    predicted_name = name[label]
    if confidence > 37:
        continue
    fr.put_text(test_img,predicted_name,x,y)

resized_img = cv2.resize(test_img, (500,500))
cv2.imshow("face-Detected",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows

#
# for(x,y,w,h) in faces_detected:
#     cv2.rectangle(test_img, (x,y), (x+w, y+h), (255,0,0), thickness = 10)
#
# resized_img = cv2.resize(test_img, (750,700))
# cv2.imshow("face-Detected",resized_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows
