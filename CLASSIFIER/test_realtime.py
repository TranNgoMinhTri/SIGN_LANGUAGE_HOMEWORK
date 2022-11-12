import cv2 as cv
import numpy as np
from sklearn.neural_network import MLPClassifier
import pickle
import matplotlib.pyplot as plt
# Load MLP model for digit classifier
mlp = pickle.load(open('train_lan_1_data_moi_(64,64).sav','rb'))


cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    # load test image
    frame = cv.flip(frame,1)
    img = frame
    # Convert color image to gray
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    roi = img[100:300,300:500]
    cv.imshow('roi',roi)

    roi = roi.astype(float)
    roi = cv.resize(roi, (28,28))
    roi = cv.GaussianBlur(roi,(7,7),0)
    cv.normalize(roi,roi,0, 1.0, cv.NORM_MINMAX)
    roi = np.reshape(roi,(1,784))
    cv.rectangle(frame,(300,100), (500,300),(0,255,0),2)
    y_predict = mlp.predict(roi)[0]
    print(y_predict)
    cv.putText(frame, str(y_predict),(1,40),cv.FONT_HERSHEY_COMPLEX,2,(0,255,0),2)
    cv.imshow('raw',frame)
    
    # cv.imshow('image',b_img)
    if cv.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()