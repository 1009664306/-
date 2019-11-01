import cv2
import numpy as np
#视频人脸检测
def showimg(img):
    # load xml 1 file name
    face_xml = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_xml = cv2.CascadeClassifier('haarcascade_eye.xml')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # detect faces 1 data 2 scale 3 5
    faces = face_xml.detectMultiScale(gray,1.3,5)
    print('face=',len(faces))
    # draw
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_face = gray[y:y+h,x:x+w]
        roi_color = img[y:y+h,x:x+w]
        # 1 gray
        eyes = eye_xml.detectMultiScale(roi_face)
        print('eye=',len(eyes))
        for (e_x,e_y,e_w,e_h) in eyes:
            cv2.rectangle(roi_color,(e_x,e_y),(e_x+e_w,e_y+e_h),(0,255,0),2)
    cv2.imshow('dst',img)
    
cap = cv2.VideoCapture(0)
while (1): 
    ret, img = cap.read()
    #print(ret)
    showimg(img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()  # 释放摄像头
cv2.destroyAllWindows()  # 释放窗口资源

