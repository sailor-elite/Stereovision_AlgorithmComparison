import os

import numpy as np
import cv2

video_capture_0 = cv2.VideoCapture(0)
video_capture_1 = cv2.VideoCapture(1)
img_counter = 0
path = os.path.join(os.getcwd(), 'pictures')
path_positive = os.path.join(os.getcwd(), 'pictures/positive')
path_negative = os.path.join(os.getcwd(), 'pictures/negative')
if not os.path.exists(path):
    os.makedirs(path)

while True:
    retL, frameL = video_capture_0.read()
    retR, frameR = video_capture_1.read()

    if retL:
        cv2.imshow('Cam L', frameL)

    if retR:
        cv2.imshow('Cam R', frameR)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        key = cv2.waitKey(1) & 0xFF
        print(f"Pressed key: {key} ({chr(key) if key != 255 else 'None'})")
        existing_files = [f for f in os.listdir(path) if f.startswith("opencv_frameL_") and f.endswith(".png")]

        if existing_files:
            indices = [int(f.split("_")[-1].split(".")[0]) for f in existing_files if
                       f.split("_")[-1].split(".")[0].isdigit()]
            if indices:
                img_counter = max(indices) + 1

        img_nameL = "opencv_frameL_{}.png".format(img_counter)
        img_nameR = "opencv_frameR_{}.png".format(img_counter)
        cv2.imwrite(os.path.join(path, img_nameL), frameL)
        cv2.imwrite(os.path.join(path, img_nameR), frameR)

    if cv2.waitKey(1) & 0xFF == ord('t'):
        existing_files = [f for f in os.listdir(path) if f.startswith("opencv_frameP_") and f.endswith(".png")]
        if existing_files:
            indices = [int(f.split("_")[-1].split(".")[0]) for f in existing_files if
                       f.split("_")[-1].split(".")[0].isdigit()]
            if indices:
                img_counter = max(indices) + 1
        img_nameL = "opencv_frameP_{}.png".format(img_counter)
        img_counter += 1
        img_nameR = "opencv_frameP_{}.png".format(img_counter)
        cv2.imwrite(os.path.join(path_positive, img_nameL), frameL)
        cv2.imwrite(os.path.join(path_positive, img_nameR), frameR)

    if cv2.waitKey(1) & 0xFF == ord('f'):
        existing_files = [f for f in os.listdir(path) if f.startswith("opencv_frameN_") and f.endswith(".png")]
        if existing_files:
            indices = [int(f.split("_")[-1].split(".")[0]) for f in existing_files if
                       f.split("_")[-1].split(".")[0].isdigit()]
            if indices:
                img_counter = max(indices) + 1
        img_nameL = "opencv_frameN_{}.png".format(img_counter)
        img_counter += 1
        img_nameR = "opencv_frameN_{}.png".format(img_counter)
        cv2.imwrite(os.path.join(path_negative, img_nameL), frameL)
        cv2.imwrite(os.path.join(path_negative, img_nameR), frameR)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture_0.release()
video_capture_1.release()
cv2.destroyAllWindows()
