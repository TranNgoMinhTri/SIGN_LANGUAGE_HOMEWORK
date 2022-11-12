import cv2 as cv
import numpy as np
from sklearn.neural_network import MLPClassifier
import pickle
import matplotlib.pyplot as plt
# Load MLP model for digit classifier
mlp = pickle.load(open('train_500.sav','rb'))



img = cv.imread("test_a.jpg", cv.IMREAD_GRAYSCALE)
img = img.astype(float)
img = cv.resize(img, (28,28))
cv.imshow("img", img)
cv.waitKey(0)
cv.destroyAllWindows()
cv.normalize(img,img,0, 1.0, cv.NORM_MINMAX)
img = np.reshape(img,(1,784))
y_predict = mlp.predict(img)[0]
print(y_predict)


