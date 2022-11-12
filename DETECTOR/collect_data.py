import cv2
import numpy as np
import time
cap = cv2.VideoCapture(0)

#collect positive data
#show your hand within the green box to capture image containing hand
def collect_data_positive():
    counter = 0
    ret, frame = cap.read()
    time.sleep(1)
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame,1)
        point_1= (350,100)
        point_2 = (550,300)
        frame_c = frame[point_1[1]:point_2[1],point_1[0]:point_2[0]]
        frame_c = cv2.cvtColor(frame_c, cv2.COLOR_BGR2GRAY)
        frame_show = frame.copy()
        cv2.rectangle(frame_show,point_1, point_2,(0,255,0),2)
        path_positive = "data_ver_2\\p"
        cv2.imshow("Collecting positive data...", frame_show)
        cv2.imshow("ROI", frame_c)
        cv2.imwrite(path_positive +"\\phai_200x200_" + str(counter)+".jpg", frame_c)
        counter += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if counter >= 700:
            break
    print("Number of collected positive image:", counter)
    cap.release()
    cv2.destroyAllWindows()

#collect negative data
def collect_data_negative():
    counter = 0
    no_img = 0
    path_negative = "data_ver_2\\n"
    #choose size of image
    size = 200
    while True:
        #read from camera frame by frame 
        ret, frame = cap.read()
        #flip frame 
        frame = cv2.flip(frame,1)
        # print(frame.shape)
        w = frame.shape[1]
        h = frame.shape[0]
        #copy frame to show
        frame_show = frame.copy()
        #The window will slide on the camera and collect negative data
        for i in range(0,h,5):
            for j in range(0,w,5):
                ret, frame = cap.read()
                frame = cv2.flip(frame,1)
                # print(frame.shape)
                w = frame.shape[1]
                h = frame.shape[0]
                frame_show = frame.copy()
                point_1= (i,j)
                point_2 = (i+size,j+size)
                cv2.rectangle(frame_show,point_1, point_2,(0,255,0),2)
                cv2.imshow("Collecting negative data...", frame_show)
                frame_c = frame[point_1[1]:point_2[1],point_1[0]:point_2[0]]
                #convert to gray image
                frame_c = cv2.cvtColor(frame_c, cv2.COLOR_BGR2GRAY)
                cv2.imshow("ROI", frame_c)
                cv2.imwrite(path_negative +"\\n_200x200_tri_" + str(counter)+".jpg", frame_c)
                counter += 1
                no_img += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if (j+size) >= 480:
                    break
            if (i+size) >= 640:
                break
        break
    print("========Number of collected negative image:", no_img)
    cap.release()
    cv2.destroyAllWindows()

# collect_data_positive()
# collect_data_negative()