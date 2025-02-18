# import required libraries
import os
import time

import cv2
from matplotlib import pyplot as plt


class DisparityMap:

    def __init__(self, numDisparities, blockSize, P1, P2, preFilterCap, textureThreshold,
                 uniquenessR, speckleRange, speckleWindowSize, disp12MaxDiff, minDisparity, mode,
                 pictureLeft, pictureRight):

        # parameters for OpenCV Stereo
        self.numDisparities = numDisparities
        self.blockSize = blockSize
        self.P1 = P1
        self.P2 = P2
        self.preFilterCap = preFilterCap
        self.textureThreshold = textureThreshold
        self.uniquenessR = uniquenessR
        self.speckleRange = speckleRange
        self.speckleWindowSize = speckleWindowSize
        self.disp12MaxDiff = disp12MaxDiff
        self.minDisparity = minDisparity
        self.mode = mode

        # variables for image rectification
        self.Left_Stereo_Map_x = None
        self.Left_Stereo_Map_y = None
        self.Right_Stereo_Map_x = None
        self.Right_Stereo_Map_y = None
        self.pictureLeft = pictureLeft
        self.pictureRight = pictureRight
        self.Left_nice = None
        self.Right_nice = None
        self.disparity_cropped = None
        self.ImgL = None
        self.ImgR = None

    def rectify_maps(self):
        """ loads rectification maps """
        cv_file = cv2.FileStorage("data/stereo_rectify_maps.xml", cv2.FILE_STORAGE_READ)
        self.Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
        self.Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
        self.Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
        self.Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
        cv_file.release()

    def read_image(self):
        self.ImgL = cv2.imread(self.pictureLeft, cv2.IMREAD_GRAYSCALE)
        self.ImgR = cv2.imread(self.pictureRight, cv2.IMREAD_GRAYSCALE)

    def image_rectification(self):
        """ rectifies pictures with rectify maps"""

        if self.ImgL is None or self.ImgR is None:
            raise ValueError("Use read_image().")
        if self.Left_Stereo_Map_x is None or self.Right_Stereo_Map_x is None:
            raise ValueError("Use rectify_maps().")

        self.Left_nice = cv2.remap(self.ImgL,
                                   self.Left_Stereo_Map_x,
                                   self.Left_Stereo_Map_y,
                                   cv2.INTER_LANCZOS4,
                                   cv2.BORDER_CONSTANT,
                                   0)

        self.Right_nice = cv2.remap(self.ImgR,
                                    self.Right_Stereo_Map_x,
                                    self.Right_Stereo_Map_y,
                                    cv2.INTER_LANCZOS4,
                                    cv2.BORDER_CONSTANT,
                                    0)

    def stereoSGBMCreate(self):
        stereo = cv2.StereoSGBM.create()
        stereo.setNumDisparities(self.numDisparities)
        stereo.setBlockSize(self.blockSize)
        stereo.setP1(self.P1)
        stereo.setP2(self.P2)
        stereo.setPreFilterCap(self.preFilterCap)
        stereo.setUniquenessRatio(self.uniquenessR)
        stereo.setSpeckleRange(self.speckleRange)
        stereo.setSpeckleWindowSize(self.speckleWindowSize)
        stereo.setDisp12MaxDiff(self.disp12MaxDiff)
        stereo.setMinDisparity(self.minDisparity)
        stereo.setMode(self.mode)

        # compute the disparity map
        disparity = stereo.compute(self.Left_nice, self.Right_nice)
        self.disparity_cropped = disparity[:, self.numDisparities:]

        # show disparity map

    def save_figure(self):

        plt.figure(figsize=(10, 5))
        plt.imshow(self.disparity_cropped, cmap='gray', aspect='auto')
        plt.axis("off")
        plt.tight_layout()
        plt.show()
        path = os.path.join(os.getcwd(), 'result')
        if not os.path.exists(path):
            os.makedirs(path)
        existing_files = [f for f in os.listdir(path) if f.startswith("result") and f.endswith(".png")]
        index = len(existing_files)

        save_path = os.path.join(path, f"result{index}.png")
        settings_save_path = os.path.join(path, f"result{index}.txt")
        if self.disparity_cropped is not None:
            plt.imsave(save_path, self.disparity_cropped, cmap='gray')
            print(f"Saved: {save_path}")

            with open(settings_save_path, "w") as f:
                f.write(f"numDisparities: {self.numDisparities}\n")
                f.write(f"blockSize: {self.blockSize}\n")
                f.write(f"preFilterType: {self.P1}\n")
                f.write(f"preFilterSize: {self.P2}\n")
                f.write(f"preFilterCap: {self.preFilterCap}\n")
                f.write(f"textureThreshold: {self.textureThreshold}\n")
                f.write(f"uniquenessR: {self.uniquenessR}\n")
                f.write(f"speckleRange: {self.speckleRange}\n")
                f.write(f"speckleWindowSize: {self.speckleWindowSize}\n")

            print(f"Saved parameters: {settings_save_path}")
        else:
            raise ValueError("Use stereoSGBMCreate().")


disparity_map = DisparityMap(
    numDisparities=15 * 16, blockSize=1 * 5, P1=9, P2=12, preFilterCap=31,
    textureThreshold=190, uniquenessR=1, speckleRange=100, speckleWindowSize=8,
    disp12MaxDiff=12, minDisparity=0, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    pictureLeft="pictures/opencv_frameL_0.png",
    pictureRight="pictures/opencv_frameR_0.png"
)

start = time.time()
disparity_map.rectify_maps()
disparity_map.read_image()
disparity_map.image_rectification()
disparity_map.stereoSGBMCreate()
print(time.time() - start)

disparity_map.save_figure()
