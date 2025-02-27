import os
import time
import cv2
import numpy as np

import Cascade_classifier_disparity_map
import FastACVNet
import SteregoSGBM_disp_map
import StereoBM_disp_map


def StereoBM(imgLeft, imgRight):
    disparity_map = StereoBM_disp_map.DisparityMap(
        numDisparities=240, blockSize=1 * 5, preFilterSize=5, preFilterType=1, preFilterCap=31,
        textureThreshold=100, uniquenessR=1, speckleRange=10, speckleWindowSize=4,
        disp12MaxDiff=1, minDisparity=0,
        pictureLeft=imgLeft,
        pictureRight=imgRight
    )
    start = time.time()
    disparity_map.rectify_maps()
    disparity_map.read_image()
    disparity_map.image_rectification()
    disparity_map.stereoBMCreate()
    print(time.time() - start)
    disparity_map.save_figure()


def FastACVNET(imgLeft, imgRight):
    start = time.time()

    model_dir = "data"
    model_filename = "fast_acvnet_plus_generalization_opset16_480x640.onnx"
    model_path = os.path.join(os.getcwd(), model_dir, model_filename)
    depth_estimator = FastACVNet.FastACVNet(model_path)

    # Load images
    left_img = cv2.imread(imgLeft)
    right_img = cv2.imread(imgRight)

    # Estimate depth and colorize it
    disparity_map = depth_estimator(left_img, right_img)
    color_disparity = depth_estimator.draw_disparity()
    combined_img = np.hstack((left_img, color_disparity))

    print(time.time() - start)
    cv2.namedWindow("Estimated disparity", cv2.WINDOW_NORMAL)
    cv2.imshow("Estimated disparity", combined_img)
    cv2.waitKey(0)


def StereoSGBM(imgLeft, imgRight):
    disparity_map = SteregoSGBM_disp_map.DisparityMap(
        numDisparities=11 * 16, blockSize=1 * 5, P1=1 * 1 * 5, P2=12 * 1 * 1, preFilterCap=16,
        textureThreshold=10, uniquenessR=1, speckleRange=80, speckleWindowSize=18,
        disp12MaxDiff=11, minDisparity=30, mode=cv2.STEREO_SGBM_MODE_SGBM,
        pictureLeft=imgLeft,
        pictureRight=imgRight
    )

    start = time.time()
    disparity_map.rectify_maps()
    disparity_map.read_image()
    disparity_map.image_rectification()
    disparity_map.stereoSGBMCreate()
    print(time.time() - start)

    disparity_map.save_figure()


def CascadeClassifier(imgLeft, imgRight):
    classifier = Cascade_classifier_disparity_map.CascadeClassifier(imgLeft, imgRight, False, 280, 0.245,
                                                                    1.05, 6, 30, 0.1, 1.3)
    start = time.time()
    classifier.load_image()
    classifier.detect_apples()
    classifier.match_apples()
    print(time.time() - start)
    classifier.display_images()

# change  imgLeft, imgRight according to your image's path
if __name__ == '__main__':
    imgLeft, imgRight = 'pictures/opencv_frameL_0.png', 'pictures/opencv_frameR_0.png'
    StereoBM(imgLeft, imgRight)
    StereoSGBM(imgLeft, imgRight)
    FastACVNET(imgLeft, imgRight)
    CascadeClassifier(imgLeft, imgRight)
