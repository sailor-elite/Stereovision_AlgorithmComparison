import os
import time
from dataclasses import dataclass

import cv2
import numpy as np
import onnxruntime


@dataclass
class CameraConfig:
    baseline: float
    f: float


DEFAULT_CONFIG = CameraConfig(0.23, 4)  # rough estimate from the original calibration


class FastACVNet():

    def __init__(self, model_path, camera_config=None, max_dist=10):
        self.input_names = None
        self.disparity_map = None
        self.depth_map = None

        self.camera_config = camera_config
        self.max_dist = max_dist

        self.initialize_model(model_path)

    def __call__(self, left_img, right_img):
        return self.estimate_depth(left_img, right_img)

    def initialize_model(self, model_path):

        # Initialize model session
        self.session = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider',
                                                                           'CPUExecutionProvider'])
        # Get model info
        self.get_input_details()
        self.get_output_details()

    def estimate_depth(self, left_img, right_img):
        self.img_height, self.img_width = left_img.shape[:2]

        left_tensor = self.prepare_input(left_img)
        right_tensor = self.prepare_input(right_img)

        output = self.inference(left_tensor, right_tensor)
        self.disparity_map = np.squeeze(output)

        # Estimate depth map from the disparity
        if self.camera_config is not None:
            self.depth_map = self.get_depth_from_disparity(self.disparity_map, self.camera_config)
        return self.disparity_map

    def prepare_input(self, img, half=False):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_input = cv2.resize(img, (self.input_width, self.input_height), cv2.INTER_AREA)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        img_input = ((img_input / 255.0 - mean) / std)
        img_input = img_input.transpose(2, 0, 1)
        img_input = img_input[np.newaxis, :, :, :]

        return img_input.astype(np.float32)

    def inference(self, left_input, right_input):
        # start = time.time()
        output = self.session.run(self.output_names, {self.input_names[0]: left_input,
                                                      self.input_names[1]: right_input})[0]
        # print(time.time() - start)
        return output

    def draw_disparity(self):
        disparity_map = cv2.resize(self.disparity_map, (self.img_width, self.img_height))

        norm_disparity_map = 255 * ((disparity_map - np.min(disparity_map)) /
                                    (np.max(disparity_map) - np.min(disparity_map)))
        norm_disparity_map = cv2.convertScaleAbs(norm_disparity_map, 1)

        color_disp = cv2.applyColorMap(norm_disparity_map, cv2.COLORMAP_MAGMA)

        scale_width = 50
        scale_height = self.img_height

        gradient = np.linspace(255, 0, scale_height).astype(np.uint8)
        gradient = np.tile(gradient[:, None], (1, scale_width))

        color_scale = cv2.applyColorMap(gradient, cv2.COLORMAP_MAGMA)

        min_disp = np.min(disparity_map)
        max_disp = np.max(disparity_map)
        mid_disp = (min_disp + max_disp) / 2.0

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        text_color = (255, 255, 255)

        cv2.putText(color_scale, f"{640:.2f}", (5, 20), font, font_scale, text_color, thickness, cv2.LINE_AA)
        cv2.putText(color_scale, f"{320:.2f}", (5, scale_height // 2), font, font_scale, text_color, thickness,
                    cv2.LINE_AA)
        cv2.putText(color_scale, f"{0:.2f}", (5, scale_height - 10), font, font_scale, text_color, thickness,
                    cv2.LINE_AA)

        combined = np.hstack((color_disp, color_scale))

        return combined

    def draw_depth(self):
        if self.depth_map is None:
            return None
        return self.util_draw_depth(self.depth_map, (self.img_width, self.img_height), self.max_dist)

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[-1].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

        self.output_shape = model_outputs[0].shape

    @staticmethod
    def get_depth_from_disparity(disparity_map, camera_config):
        return camera_config.f * camera_config.baseline / disparity_map

    @staticmethod
    def util_draw_depth(depth_map, img_shape, max_dist):
        norm_depth_map = 255 * (1 - depth_map / max_dist)
        norm_depth_map[norm_depth_map < 0] = 0
        norm_depth_map[norm_depth_map >= 255] = 0

        norm_depth_map = cv2.resize(norm_depth_map, img_shape)

        return cv2.applyColorMap(cv2.convertScaleAbs(norm_depth_map, 1), cv2.COLORMAP_JET)


if __name__ == '__main__':
    start = time.time()

    model_dir = "data"
    model_filename = "fast_acvnet_plus_generalization_opset16_480x640.onnx"
    model_path = os.path.join(os.getcwd(), model_dir, model_filename)
    depth_estimator = FastACVNet(model_path)

    # Load images
    left_img = cv2.imread('pictures/opencv_frameL_13.png')
    right_img = cv2.imread('pictures/opencv_frameR_13.png')

    # Estimate depth and colorize it
    disparity_map = depth_estimator(left_img, right_img)
    color_disparity = depth_estimator.draw_disparity()
    combined_img = np.hstack((left_img, color_disparity))

    print(time.time()-start)
    cv2.namedWindow("Estimated disparity", cv2.WINDOW_NORMAL)
    cv2.imshow("Estimated disparity", combined_img)
    cv2.waitKey(0)
