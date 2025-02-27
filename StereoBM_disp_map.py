# import required libraries
import os
import time

import cv2
import numpy as np
from matplotlib import pyplot as plt


class DisparityMap:

    def __init__(self, numDisparities, blockSize, preFilterType, preFilterSize, preFilterCap, textureThreshold,
                 uniquenessR, speckleRange, speckleWindowSize, disp12MaxDiff, minDisparity, pictureLeft, pictureRight):

        # parameters for OpenCV Stereo
        self.numDisparities = numDisparities
        self.blockSize = blockSize
        self.preFilterSize = preFilterSize
        self.preFilterType = preFilterType
        self.preFilterCap = preFilterCap
        self.textureThreshold = textureThreshold
        self.uniquenessR = uniquenessR
        self.speckleRange = speckleRange
        self.speckleWindowSize = speckleWindowSize
        self.disp12MaxDiff = disp12MaxDiff
        self.minDisparity = minDisparity

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

    def stereoBMCreate(self):
        stereo = cv2.StereoBM.create()
        stereo.setNumDisparities(self.numDisparities)
        stereo.setBlockSize(self.blockSize)
        stereo.setPreFilterType(self.preFilterType)
        stereo.setPreFilterSize(self.preFilterSize)
        stereo.setPreFilterCap(self.preFilterCap)
        stereo.setTextureThreshold(self.textureThreshold)
        stereo.setUniquenessRatio(self.uniquenessR)
        stereo.setSpeckleRange(self.speckleRange)
        stereo.setSpeckleWindowSize(self.speckleWindowSize)
        stereo.setDisp12MaxDiff(self.disp12MaxDiff)
        stereo.setMinDisparity(self.minDisparity)

        # compute the disparity map
        disparity = stereo.compute(self.Left_nice, self.Right_nice)
        self.disparity_cropped = disparity[:, self.numDisparities:]

    import os
    import matplotlib.pyplot as plt
    import numpy as np

    def save_figure(self):
        '''shows and saves figure with a grayscale scale bar'''

        min_disp, max_disp = 0, 3500
        new_min, new_max = 0, 640

        disparity_scaled = np.interp(self.disparity_cropped, (min_disp, max_disp), (new_min, new_max))

        fig, ax = plt.subplots(figsize=(10, 5))
        img = ax.imshow(disparity_scaled, cmap='gray', aspect='auto', vmin=new_min, vmax=new_max)
        ax.axis("off")

        cbar = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("Dysparycja")

        ticks = np.arange(new_min, new_max + 1, 160)
        cbar.set_ticks(ticks)
        cbar.set_ticklabels([f"{int(tick)}" for tick in ticks])

        plt.tight_layout()
        plt.show()

        path = os.path.join(os.getcwd(), 'result')
        if not os.path.exists(path):
            os.makedirs(path)

        existing_files = [f for f in os.listdir(path) if f.startswith("result") and f.endswith(".png")]
        index = len(existing_files)

        save_path = os.path.join(path, f"result{index}.png")
        settings_save_path = os.path.join(path, f"result{index}.txt")

        fig.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {save_path}")

        with open(settings_save_path, "w") as f:
            f.write(f"numDisparities: {self.numDisparities}\n")
            f.write(f"blockSize: {self.blockSize}\n")
            f.write(f"preFilterType: {self.preFilterType}\n")
            f.write(f"preFilterSize: {self.preFilterSize}\n")
            f.write(f"preFilterCap: {self.preFilterCap}\n")
            f.write(f"textureThreshold: {self.textureThreshold}\n")
            f.write(f"uniquenessR: {self.uniquenessR}\n")
            f.write(f"speckleRange: {self.speckleRange}\n")
            f.write(f"speckleWindowSize: {self.speckleWindowSize}\n")
            f.write(f"disp12MaxDiff: {self.disp12MaxDiff}\n")
            f.write(f"minDisparity: {self.minDisparity}\n")

        print(f"Saved parameters: {settings_save_path}")


