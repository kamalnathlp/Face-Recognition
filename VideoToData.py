import os
import cv2
import numpy as np
import faceRecognition as fr


#This module captures images via webcam and performs face recognition
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('trainingData.xml')#Load saved training data

name={1:'Santhosh',2:'Vasu', 3:"Sasi"}

face_recognizer = []
cap=cv2.VideoCapture(0)
i=0
name = 1
faces = []
faceID=[]

while True:
    i=i+1
    if i>500:
        name = int(input("Enter the Number"))
        i=0
    if name == 0 :
        face_recognizer.save('Video-to-data.xml')
        break
    
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
    faces_detected,gray_img= fr.faceDetection(test_img)
    if len(faces_detected) != 1:
        continue
    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=3)
    print(i)
    roi_gray = gray_img[y:y + w, x:x + h]
    faces.append(roi_gray)
    faceID.append(name)

    face_recognizer = fr.train_classifier(faces, faceID)

    #
    # for (x,y,w,h) in faces_detected:
    #   cv2.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=3)
    # #
    # # resized_img = cv2.resize(test_img, (1000, 700))
    # # cv2.imshow('face detection Tutorial ',resized_img)
    # # cv2.waitKey(10)
    #
    #
    # for face in faces_detected:
    #     (x,y,w,h)=face
    #     roi_gray=gray_img[y:y+w, x:x+h]
    #     label,confidence=face_recognizer.predict(roi_gray)#predicting the label of given image
    #     print("confidence:",confidence)
    #     print("label:",label)
    #     fr.draw_rect(test_img,face)
    #     predicted_name=name[label]
    #     if confidence < 39:#If confidence less than 37 then don't print predicted face text on screen
    #        fr.put_text(test_img,predicted_name,x,y)
    #
    #
    # resized_img = cv2.resize(test_img, (1000, 700))
    # cv2.imshow('face recognition tutorial ',resized_img)
    # if cv2.waitKey(10) == ord('q'):#wait until 'q' key is pressed
    #     break


cap.release()
cv2.destroyAllWindows

