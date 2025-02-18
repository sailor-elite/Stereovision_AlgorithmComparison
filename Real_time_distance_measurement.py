import cv2
import numpy as np


class DistanceMeasurement:
    def __init__(self, CamL_id, CamR_id, Img_source, ScaleFactor, minNeighbors, baseline, correction):
        self.scaleFactor = ScaleFactor
        self.minNeighbors = minNeighbors
        self.CamL_id = CamL_id
        self.CamR_id = CamR_id
        self.Img_source = Img_source
        self.baseline = baseline
        self.correction = correction

        self.focal_length = 4e-3
        self.pixel_size = 10e-6

    def nothing(self, value):
        pass

    def calculate_distance(self, xL, xR):
        disparity = abs(xL - xR)
        print("disparity", disparity)
        if disparity == 0:
            return None
        Z = ((self.focal_length / self.pixel_size) * (self.baseline / (disparity*self.correction)))
        return Z

    def video_capture(self):
        apple_cascade = cv2.CascadeClassifier('training_output/cascade.xml')

        if apple_cascade.empty():
            print("can't load classifier")
            return

        CamL = cv2.VideoCapture(self.CamL_id)
        CamR = cv2.VideoCapture(self.CamR_id)

        if not CamL.isOpened() or not CamR.isOpened():
            print("can't open camera")
            return

        cv2.namedWindow('disp', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('disp', 600, 30)
        cv2.createTrackbar('ScaleFactor', 'disp', int(self.scaleFactor * 100), 200, self.nothing)
        cv2.createTrackbar('minNeighbors', 'disp', self.minNeighbors, 50, self.nothing)

        while True:
            retL, imgL = CamL.read()
            retR, imgR = CamR.read()

            if not retL or not retR:
                print("no image from left or right camera")
                break

            self.scaleFactor = max(cv2.getTrackbarPos('ScaleFactor', 'disp') / 100, 1.01)
            self.minNeighbors = max(cv2.getTrackbarPos('minNeighbors', 'disp'), 1)

            scale_factor = 0.5
            small_imgL = cv2.resize(imgL, (0, 0), fx=scale_factor, fy=scale_factor)
            small_imgR = cv2.resize(imgR, (0, 0), fx=scale_factor, fy=scale_factor)

            gray_imgL = cv2.cvtColor(small_imgL, cv2.COLOR_BGR2GRAY)
            gray_imgR = cv2.cvtColor(small_imgR, cv2.COLOR_BGR2GRAY)

            apples_rectL = apple_cascade.detectMultiScale(gray_imgL, self.scaleFactor, self.minNeighbors)
            apples_rectR = apple_cascade.detectMultiScale(gray_imgR, self.scaleFactor, self.minNeighbors)

            for (xL, yL, wL, hL) in apples_rectL:
                best_match = None
                min_diff = float('inf')

                for (xR, yR, wR, hR) in apples_rectR:
                    disparity = abs(xL - xR)
                    if disparity < min_diff:
                        min_diff = disparity
                        best_match = (xR, yR, wR, hR)

                if best_match:
                    xR, yR, wR, hR = best_match
                    Z = self.calculate_distance(xL, xR)
                    if Z is not None:
                        text = f"{Z:.2f} m"
                        cv2.putText(imgL, text, (xL, yL - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    cv2.rectangle(imgR, (int(xR / scale_factor), int(yR / scale_factor)),
                                  (int((xR + wR) / scale_factor), int((yR + hR) / scale_factor)), (0, 255, 0), 2)

                cv2.rectangle(imgL, (int(xL / scale_factor), int(yL / scale_factor)),
                              (int((xL + wL) / scale_factor), int((yL + hL) / scale_factor)), (0, 255, 0), 2)

            cv2.imshow('Cam L', imgL)
            cv2.imshow('Cam R', imgR)
            cv2.waitKey(10)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        CamL.release()
        CamR.release()
        cv2.destroyAllWindows()


distance_measurement = DistanceMeasurement(0, 1, None, 1.1, 12, 0.185, 1)
distance_measurement.video_capture()
