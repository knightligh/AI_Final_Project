#1.Import the libraries
import cv2 
import numpy as np
from tensorflow.keras import models 
import sys


models = models.load_model('detect.h5')
#img_path = 'projects1.jpg'
lst_result = ['DOAN THANH NAM','LE VAN QUANG','NGUYEN QUOC TIEN','TRAN LE NHAT HUY','VU DUC BINH']
#lst_result = ['CTP','NAM','QUANG','TRUONG','PHONG','THANG']

face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
#img = cv2.imread(img_path)
#mg_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

cam = cv2.VideoCapture('Nam.mp4')
#cam = cv2.VideoCapture(0)

while True:
    OK, frame = cam.read()
    
    faces = face_detector.detectMultiScale(frame,1.3,5)
    global result
    #ve khung hinh guong mat
    for (x,y,w,h) in faces:
        
        roi = cv2.resize(frame[y:y+h , x: x + w],(300,300))
        result = np.argmax(models.predict(roi.reshape((-1, 300, 300, 3))))
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame,lst_result[result],(x+15,y-15),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
       
       
    
    predict = result

    print("This is: ", lst_result[np.argmax(predict[0])],(predict[0]))

    a=np.max(predict[0])
    a=a*100
    print("reliability:",a,'%')

    cv2.imshow('FRAME',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cam.release()
cv2.destroyAllWindows()