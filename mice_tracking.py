from typing import Any, Union

import cv2
import argparse
import numpy as np
from datetime import datetime
from guizero import App, Text, PushButton, TextBox




def nothing(int):
    pass


def crop():
    cv2.namedWindow('crop selection', cv2.WINDOW_NORMAL)
    switch = '1:ok'
    cv2.createTrackbar(switch, 'crop selection', 0, 1, nothing)
    cap = cv2.VideoCapture("C:/Users/sharaj/Documents/phd project/pi cam projects/Video 3099.mp4")
    frame_to_crop = cap.read()[1]
    cv2.imshow('crop selection', frame_to_crop)
    fourcc = cv2.VideoWriter_fourcc(* 'mp4v')
    out_crop = cv2.VideoWriter('cropped.mp4', fourcc, 30.0, (640, 480))
    r = cv2.selectROI('crop selection', frame_to_crop)
    while True:
        s = cv2.getTrackbarPos(switch, 'crop selection')
        cv2.waitKey(1)
        if (s == 1):
            cv2.destroyAllWindows()
            break
    print(r)
    if (s == 1):
        while True:
            ret, frame = cap.read()
            if ret == True:
                clone = frame.copy()
                imCrop = clone[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
                img = cv2.resize(imCrop, (640, 480))
                out_crop.write(img)
                cv2.imshow('cropped', imCrop)
                cv2.waitKey(1)
            elif (ret == False):
                cap.release()
                cv2.destroyAllWindows()
                break
    out_crop.release()
    print ("released")
    return


def bgrnd():
    cap = cv2.VideoCapture("C:/Users/sharaj/Documents/phd project/pi cam projects/cropped.mp4")
    i = input_noise_factor.value
    k = int(i)
    j = 0
    # switch = '1:ok'
    ret, frame = cap.read()
    (h, w) = frame.shape[:2]
    template = np.zeros((int(h/2), int(w/2),3), dtype="uint8")
    keyframe = np.zeros((h, w, 3), dtype="uint8")
    background = np.zeros((h, w, 3), dtype="uint8")
    cv2.namedWindow('select mice quadrant', cv2.WINDOW_NORMAL)
    while j<k:
        frame = cap.read()[1]
        clone = frame.copy()
        image = cv2.rectangle(frame, pt1=(0, 0), pt2=(int(w / 2), int(h / 2)), color=(255, 0, 0), thickness=2)
        image1 = cv2.rectangle(image, pt1=(int(w / 2), int(h / 2)), pt2=(w, h), color=(0, 255, 0), thickness=2)
        # cv2.createTrackbar(switch, 'select mice quadrant', 0, 1, nothing)
        cv2.imshow('select mice quadrant', image1)
        key = cv2.waitKey(0) & 0xff
        # s = cv2.getTrackbarPos(switch, 'select mice quadrant')
        if j<k:
            cv2.namedWindow('generating background', cv2.WINDOW_NORMAL)
            if key == ord('a'):
                background[0:h,0:w] = clone
                background[int(0):int(h/2), 0:int(w/2)] = template
                while True:
                    ret, im = cap.read()
                    clone1 = im.copy()
                    img = cv2.rectangle(im, pt1=(0, 0), pt2=(int(w / 2), int(h / 2)), color=(255, 0, 0), thickness=2)
                    imge = cv2.rectangle(img, pt1=(int(w / 2), int(h / 2)), pt2=(w, h), color=(0, 255, 0), thickness=2)
                    cv2.imshow("Press y when mice is not in A", imge)
                    cv2.imshow('generating background', background)
                    key1 = cv2.waitKey(100) & 0xff
                    if key1 == ord('y'):
                        cv2.destroyWindow("press y when mice is not in A")
                        cv2.destroyWindow("generating background")
                        background[int(0):int(h/2), 0:int(w/2)] = clone1[int(0):int(h/2), 0:int(w/2)]
                        cv2.namedWindow('generated background', cv2.WINDOW_NORMAL)
                        # cv2.createTrackbar(switch, 'generated background', 0, 1, nothing)
                        cv2.imshow('generated background', background)
                        cv2.waitKey(0)
                        # s = cv2.getTrackbarPos(switch, 'generated background')
                        j = j + 1
                        break
            elif key == ord('b'):
                background[0:h,0:w] = clone
                background[int(0):int(h / 2), int(w/2):int(w)] = template
                while True:
                    ret, im = cap.read()
                    clone1 = im.copy()
                    img = cv2.rectangle(im, pt1=(0, 0), pt2=(int(w / 2), int(h / 2)), color=(255, 0, 0), thickness=2)
                    imge = cv2.rectangle(img, pt1=(int(w / 2), int(h / 2)), pt2=(w, h), color=(0, 255, 0), thickness=2)
                    cv2.imshow("Press y when mice is not in B", imge)
                    cv2.imshow('generating background', background)
                    key1 = cv2.waitKey(100) & 0xff
                    if key1 == ord('y'):
                        background[int(0):int(h / 2), int(w/2):int(w)] = clone1[int(0):int(h / 2), int(w/2):int(w)]
                        cv2.namedWindow('generated background', cv2.WINDOW_NORMAL)
                        # cv2.createTrackbar(switch, 'generated background', 0, 1, nothing)
                        cv2.imshow('generated background', background)
                        cv2.waitKey(0)
                        # s = cv2.getTrackbarPos(switch, 'generated background')
                        j = j + 1
                        break
            elif key == ord('c'):
                background[0:h,0:w] = clone
                background[int(h/2):int(h), int(0):int(w/2)] = template
                while True:
                    ret, im = cap.read()
                    clone1 = im.copy()
                    img = cv2.rectangle(im, pt1=(0, 0), pt2=(int(w / 2), int(h / 2)), color=(255, 0, 0), thickness=2)
                    imge = cv2.rectangle(img, pt1=(int(w / 2), int(h / 2)), pt2=(w, h), color=(0, 255, 0), thickness=2)
                    cv2.imshow("Press y when mice is not in C", imge)
                    cv2.imshow('generating background', background)
                    key1 = cv2.waitKey(100) & 0xff
                    if key1 == ord('y'):
                        background[int(h/2):int(h), int(0):int(w/2)] = clone1[int(h/2):int(h), int(0):int(w/2)]
                        cv2.namedWindow('generated background', cv2.WINDOW_NORMAL)
                        # cv2.createTrackbar(switch, 'generated background', 0, 1, nothing)
                        cv2.imshow('generated background', background)
                        cv2.waitKey(0)
                        # s = cv2.getTrackbarPos(switch, 'generated background')
                        j = j + 1
                        break
            elif key == ord('d'):
                background[0:h,0:w] = clone
                background[int(h / 2):int(h), int(w/2):int(w)] = template
                while True:
                    ret, im = cap.read()
                    clone1 = im.copy()
                    img = cv2.rectangle(im, pt1=(0, 0), pt2=(int(w / 2), int(h / 2)), color=(255, 0, 0), thickness=2)
                    imge = cv2.rectangle(img, pt1=(int(w / 2), int(h / 2)), pt2=(w, h), color=(0, 255, 0), thickness=2)
                    cv2.imshow("Press y when mice is not in D", imge)
                    cv2.imshow('generating background', background)
                    key1 = cv2.waitKey(100) & 0xff
                    if key1 == ord('y'):
                        background[int(h / 2):int(h), int(w/2):int(w)] = clone1[int(h / 2):int(h), int(w/2):int(w)]
                        cv2.namedWindow('generated background', cv2.WINDOW_NORMAL)
                        # cv2.createTrackbar(switch, 'generated background', 0, 1, nothing)
                        cv2.imshow('generated background', background)
                        # s = cv2.getTrackbarPos(switch, 'generated background')
                        cv2.waitKey(0)
                        j = j + 1
                        break
        keyframe = cv2.addWeighted(keyframe,(1),background,(1/k),0)
    cv2.imwrite("background.jpg",keyframe)
    cv2.imshow("final background",keyframe)
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
    return


def trackmice():
    cap = cv2.VideoCapture('cropped.mp4')
    temp = cv2.imread("background.jpg")
    # (h, w) = temp.shape[:2]
    # img = np.array(50, (int(h),int(w)), dtype = int)
    template = temp
    print("background checkpoint")
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 1000
    params.filterByCircularity = True
    params.minCircularity = 0.9
    params.filterByConvexity = True
    params.minConvexity = 0.87
    while cap.isOpened():
        ret, frame1 = cap.read()
        if ret:
            trackframe = cv2.subtract(template,frame1)
            img = trackframe
            ret, thresh1 = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
            detector = cv2.SimpleBlobDetector_create(params)
            print("dectector checkpoint")
            keypoints = detector.detect(thresh1)
            # print("keypoints obtained")
            trackframe_with_keypoints = cv2.drawKeypoints(thresh1, keypoints, np.array([]), (0, 0, 255),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.imshow("orignial",frame1)
            cv2.imshow("mice tracking",trackframe_with_keypoints)
            cv2.waitKey(10)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    return

# def mice_tracking():

app1 = App(title="Event arena mice tracking", layout="grid")
intro = Text(app1, text="Enter file name and click Crop video.", grid=[0,10], align="left")
input_file_name = TextBox(app1, grid=[1,10,2,1])
croping = PushButton(app1, command=crop, text="Crop video", grid=[10,10], align="bottom")
intro1 = Text(app1, text="Enter noise factor and click generate background.", grid=[0,18], align="left")
input_noise_factor = TextBox(app1, grid=[1,18,2,1])
background = PushButton(app1, text="generate background", grid=[10,18], align="bottom")
background.when_clicked = bgrnd
intro2 = Text(app1, text="select track mice once the background is generated.", grid=[0,26], align="left")
track_mice = PushButton(app1, command=trackmice, text="Track Mice", grid=[10,26], align="bottom")
app1.display()
