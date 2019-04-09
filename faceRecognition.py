import cv2
import os
import numpy as np

def faceDetection(test_img):
    gray_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
    face_haar_cascade = cv2.CascadeClassifier('haarcascades\haarcascade_frontalface_default.xml')
    faces = face_haar_cascade.detectMultiScale(gray_img,scaleFactor = 1.32, minNeighbors = 5)
    for(x,y,w,h) in faces:
        cv2.rectangle(test_img, (x,y), (x+w, y+h), (255,0,0), thickness = 10)
    
    resized_img = cv2.resize(test_img, (750,700))
    cv2.imshow("face-Detected",resized_img)
    cv2.waitKey(100)
    return faces, gray_img

def labels_for_training_data(directory):
    faces=[]
    faceID=[]

    for path,subdirnames,filenames in os.walk(directory):
        for filename in filenames:
            id = os.path.basename(path)
            img_path = os.path.join(path,filename)
            print("Path",img_path)

            test_img = cv2.imread(img_path)
            if test_img is None:
                print("Image Not Loaded Properly")
                continue
            faces_rect,gray_img = faceDetection(test_img)
            if len(faces_rect)!=1:
                print("Not Processed")
                continue
            (x,y,w,h) = faces_rect[0]
            #RegionOfIntrest
            roi_gray = gray_img[y:y+w, x:x+h]
            faces.append(roi_gray)
            faceID.append(int(id))
    return faces,faceID

def train_classifier(faces,faceId):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, np.array(faceId))

    return face_recognizer

def draw_rect(test_img, face):
    x,y,w,h = face
    cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=10)

def put_text(test_img, text , x,y):
    cv2.putText(test_img, text, (x,y), cv2.FONT_HERSHEY_DUPLEX,3,(255,0,0), 6)


