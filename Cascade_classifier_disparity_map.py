import cv2
import numpy as np


def get_gray_color(disparity, max_width):
    factor = (disparity / max_width) ** 0.75
    intensity = int(255 * factor)
    return intensity, intensity, intensity


def calculate_disparity(x_left, x_right):
    disparity = abs(x_left - x_right)
    print("disparity", disparity)
    return disparity


def calculate_distance(x_left, x_right, baseline, correction):
    disparity = abs(x_left - x_right)
    print("disparity", disparity)
    focal_length = 4e-3
    pixel_size = 10e-6
    if disparity == 0:
        return None
    Z = ((focal_length / pixel_size) * (baseline / (disparity * correction)))
    return Z


class CascadeClassifier:
    def __init__(self, ImgPathL, ImgPathR, MAX_X_SHIFT, baseline, scaleFactor=1.1, minNeighbors=12, window_size=12,
                 Min_distance_threshold=0.2, ):
        self.baseline = baseline
        self.Min_distance_threshold = Min_distance_threshold
        self.ImgPathR = ImgPathL
        self.ImgPathL = ImgPathR
        self.minNeighbors = minNeighbors
        self.scaleFactor = scaleFactor
        self.window_size = window_size
        self.pixels_R = None
        self.pixels_L = None
        self.grayR = None
        self.grayL = None
        self.ImgL = None
        self.ImgR = None
        self.MAX_X_SHIFT = MAX_X_SHIFT

    def get_avg_color(self, image, x_color, y_color):
        h_color, w_color, _ = image.shape
        x1, x2 = max(0, x_color - self.window_size // 2), min(w_color, x_color + self.window_size // 2)
        y1, y2 = max(0, y_color - self.window_size // 2), min(h_color, y_color + self.window_size // 2)
        patch = image[y1:y2, x1:x2]
        return np.mean(patch, axis=(0, 1))

    def load_image(self):
        self.ImgL = cv2.imread(self.ImgPathL)
        self.ImgR = cv2.imread(self.ImgPathR)
        self.grayL = cv2.cvtColor(self.ImgL, cv2.COLOR_BGR2GRAY)
        self.grayR = cv2.cvtColor(self.ImgR, cv2.COLOR_BGR2GRAY)

    def detect_apples(self):
        apple_cascade = cv2.CascadeClassifier('training_output/cascade.xml')

        apples_rectL = apple_cascade.detectMultiScale(self.grayL, self.scaleFactor, self.minNeighbors)
        apples_rectR = apple_cascade.detectMultiScale(self.grayR, self.scaleFactor, self.minNeighbors)

        self.pixels_L = self.process_detected_apples(self.ImgL, apples_rectL)
        self.pixels_R = self.process_detected_apples(self.ImgR, apples_rectR)

    def process_detected_apples(self, image, apples_rect):
        pixels = []
        for (x, y, w, h) in apples_rect:
            x_center, y_center = int(x + w / 2), int(y + h / 2)
            avg_color = self.get_avg_color(image, x_center, y_center)
            pixels.append((x_center, y_center, avg_color, x, y, w, h))
        return pixels

    def match_apples(self):
        for x_CenterL, y_CenterL, colorL, xL, yL, wL, hL in self.pixels_L:
            best_match = None
            min_distance = float('inf')

            for x_CenterR, y_CenterR, colorR, xR, yR, wR, hR in self.pixels_R:
                if abs(x_CenterL - x_CenterR) > self.MAX_X_SHIFT:
                    continue

                distance = np.linalg.norm(np.array(colorL) - np.array(colorR))

                if distance < min_distance:
                    min_distance = distance
                    best_match = (x_CenterR, y_CenterR, colorR)

            if best_match:
                print(
                    f"Apple in the left image ({x_CenterL}, {y_CenterL}) -> matches ({best_match[0]},"
                    f" {best_match[1]}) | Similarity: {min_distance}")
                Z_distance = calculate_distance(x_CenterL, best_match[0], self.baseline, 1)
                disp = calculate_disparity(x_CenterL, best_match[0])
                print(Z_distance)
                if Z_distance > self.Min_distance_threshold:
                    cv2.rectangle(self.ImgL, (xL, yL), (xL + wL, yL + hL), get_gray_color(disp, 640), 8)
                    cv2.putText(self.ImgL, str(round(Z_distance, 2)), (x_CenterL, y_CenterL), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 0, 0), 2)

    def display_images(self):
        cv2.imshow('disparityMap', self.ImgL)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


classifier = CascadeClassifier('pictures/opencv_frameL_0.png', 'pictures/opencv_frameR_0.png', 380, 0.16,
                               1.1, 3, 30, 0.3)
classifier.load_image()
classifier.detect_apples()
classifier.match_apples()
classifier.display_images()
