import numpy as np
import cv2 as cv
import pickle
# Load cascade filter
hand_cascade = cv.CascadeClassifier()
hand_cascade.load('weights_detect/train_lan_2_data_moi.xml')

#load classifier
mlp = pickle.load(open('weights_classify/nammoadidaphat_(15,20).sav','rb'))
dic = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H", 8: "I", 9: "K"}

def hand_detector(img_detect, frame):
    #convert BGR to GRAY and preprocessing
    img_detect = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    img_detect = cv.GaussianBlur(img_detect,(3,3),0)
    #run detector
    hand = hand_cascade.detectMultiScale(img_detect,scaleFactor=1.7, minNeighbors=7, minSize = (80,80))
    #convert to numpy array
    hang_np = np.array(hand, dtype=int)
    #create list of boxes
    list_box = []
    for (x,y,w,h) in hang_np:
        list_box.append((x,y,w,h))
        # cv.rectangle(frame, (x,y), (x+w,y+h), (0,0,255),2)
    list_box = np.array(list_box, dtype=int)
    #find the largest box and draw it
    max_wh = []
    if len(list_box) != 0:
        for (x,y,w,h) in list_box:
            avg_wh = int((w+h)/2)
            max_wh.append(avg_wh)
        max_box_index = np.argmax(max_wh)
        (x,y,w,h) = list_box[max_box_index]
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0),2)
        return list_box[max_box_index]
    else:
        return []



def classifier(roi_classify):
    #convert BGR to GRAY and preprocessing
    roi_classify = cv.cvtColor(roi_classify, cv.COLOR_BGR2GRAY)
    roi_classify = roi_classify.astype(float)
    #reshape particullar roi to 28x28 roi
    roi_classify = cv.resize(roi_classify, (28,28))
    #normalize before detecting
    cv.normalize(roi_classify,roi_classify,0, 1.0, cv.NORM_MINMAX)
    #convert to 786-length vector
    roi_classify = np.reshape(roi_classify,(1,784))
    #classify
    y_predict = mlp.predict(roi_classify)[0]
    return y_predict


def main():
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print('can not open video clip/camera')
        exit()
    while True:
        # read frame by frame
        ret, frame = cap.read()
        if not ret:
            print(' Cannot read video frame. Video ended?')
            break
        #detect hand
        img_detect = frame.copy()
        bbox = hand_detector(img_detect, frame)
        #classify
        if len(bbox) == 0:
            # print("Cannot detect hand!!!")
            cv.putText(frame,"Cannot detect your hand!!!",(20,50),cv.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2)
        else:
            y,x,w,h = bbox[0],bbox[1],bbox[2],bbox[3]
            x_max = x+w
            y_max = y+h
            roi = img_detect[x:x_max,y:y_max]
            predicting = classifier(roi_classify=roi)
            cv.putText(frame,dic[predicting],(y,x),cv.FONT_HERSHEY_SIMPLEX,2,(0,255,0),2)
            # cv.imshow("roi_detect", roi)
        #Show result
        cv.namedWindow("Sign Language Recognition",cv.WINDOW_AUTOSIZE)
        cv.putText(frame,"Press Q to quit",(10,450),cv.FONT_HERSHEY_COMPLEX,0.3,(0,0,0),1)
        cv.putText(frame,"Please keep the distance betwwen your hand and camera about 1m and use the only-1-color background",(120,450),cv.FONT_HERSHEY_SIMPLEX,0.3,(0,0,0),1)
        cv.putText(frame,"Tran Ngo Minh Tri-19146055 | Nguyen Hoai Nam-19146219",(340,20),cv.FONT_HERSHEY_SIMPLEX,0.3,(0,255,255),1)
        cv.imshow('Sign Language Recognition', frame)
        # close camera
        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()