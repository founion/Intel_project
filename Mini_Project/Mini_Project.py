%pip install -q "openvino>=2023.1.0"
%pip install -q --extra-index-url https://download.pytorch.org/whl/cpu torch opencv-python matplotlib
%pip install -q "gdown<4.6.4"

import collections
import sys
import time
import cv2
import os
import torch
import numpy as np
import openvino as ov
import notebook_utils as utils
import matplotlib.pyplot as plt
import ipywidgets as widgets

from IPython import display
from numpy.lib.stride_tricks import as_strided
from decoder import OpenPoseDecoder
from pathlib import Path
from collections import namedtuple
from IPython.display import HTML, FileLink, display
from notebook_utils import load_image
from model.u2net import U2NET, U2NETP

sys.path.append("../utils")

# A directory where the model will be downloaded.
base_model_dir = Path("model")

# The name of the model from Open Model Zoo.
model_name = "human-pose-estimation-0001"
# Selected precision (FP32, FP16, FP16-INT8).
precision = "FP16-INT8"

model_path = base_model_dir / "intel" / \
    model_name / precision / f"{model_name}.xml"

if not model_path.exists():
    model_url_dir = f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/{model_name}/{precision}/"
    utils.download_file(model_url_dir + model_name + '.xml',
                        model_path.name, model_path.parent)
    utils.download_file(model_url_dir + model_name + '.bin',
                        model_path.with_suffix('.bin').name, model_path.parent)

core = ov.Core()

device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value='AUTO',
    description='Device:',
    disabled=False,
)

# Initialize OpenVINO Runtime
core = ov.Core()
# Read the network from a file.
model = core.read_model(model_path)
# Let the AUTO device decide where to load the model (you can use CPU, GPU as well).
compiled_model = core.compile_model(model=model, device_name=device.value, config={"PERFORMANCE_HINT": "LATENCY"})

# Get the input and output names of nodes.
input_layer = compiled_model.input(0)
output_layers = compiled_model.outputs

# Get the input size.
height, width = list(input_layer.shape)[2:]
input_layer.any_name, [o.any_name for o in output_layers]
decoder = OpenPoseDecoder()

# Import local modules
utils_file_path = Path("../utils/notebook_utils.py")
notebook_directory_path = Path(".")

if not utils_file_path.exists():
    # Clone the repository if the notebook_utils.py file does not exist
    # !git clone --depth 1 https://github.com/openvinotoolkit/openvino_notebooks.git
    utils_file_path = Path("./openvino_notebooks/notebooks/utils/notebook_utils.py")
    notebook_directory_path = Path("./openvino_notebooks/notebooks/205-vision-background-removal/")

sys.path.append(str(utils_file_path.parent))
sys.path.append(str(notebook_directory_path))

# Define the model configurations
model_config = namedtuple("ModelConfig", ["name", "url", "model", "model_args"])

u2net_lite = model_config(
    name="u2net_lite",
    url="https://drive.google.com/uc?id=1W8E4FHIlTVstfRkYmNOjbr0VDXTZm0jD",
    model=U2NETP,
    model_args=(),
)

# Set the chosen model configuration
u2net_model = u2net_lite

# Load the downloaded model
MODEL_DIR = "model"
model_path = Path(MODEL_DIR) / u2net_model.name / Path(u2net_model.name).with_suffix(".pth")

# Load the model
net = u2net_model.model(*u2net_model.model_args)
net.eval()

# Load the weights
print(f"Loading model weights from: '{model_path}'")
net.load_state_dict(state_dict=torch.load(model_path, map_location="cpu"))

# Convert the PyTorch model to OpenVINO IR
model_ir = ov.convert_model(net, example_input=torch.zeros((1,3,512,512)), input=([1, 3, 512, 512]))

# Choose the device for inference
core = ov.Core()
device = "CPU"

# Load the network to OpenVINO Runtime
compiled_model_ir = core.compile_model(model=model_ir, device_name=device)

# Resize the background image to 480x640
background_image1 = cv2.imread("universe.jpg")
background_image1 = cv2.resize(background_image1, (640, 480))

background_image2 = cv2.imread("central_park.jpeg")
background_image2 = cv2.resize(background_image2, (640, 480))

background_image3 = cv2.imread("beach.jpg")
background_image3 = cv2.resize(background_image3, (640, 480))

# 2D pooling in numpy (from: https://stackoverflow.com/a/54966908/1624463)
def pool2d(A, kernel_size, stride, padding, pool_mode="max"):
    """
    2D Pooling

    Parameters:
        A: input 2D array
        kernel_size: int, the size of the window
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    """
    # Padding
    A = np.pad(A, padding, mode="constant")

    # Window view of A
    output_shape = (
        (A.shape[0] - kernel_size) // stride + 1,
        (A.shape[1] - kernel_size) // stride + 1,
    )
    kernel_size = (kernel_size, kernel_size)
    A_w = as_strided(
        A,
        shape=output_shape + kernel_size,
        strides=(stride * A.strides[0], stride * A.strides[1]) + A.strides
    )
    A_w = A_w.reshape(-1, *kernel_size)

    # Return the result of pooling.
    if pool_mode == "max":
        return A_w.max(axis=(1, 2)).reshape(output_shape)
    elif pool_mode == "avg":
        return A_w.mean(axis=(1, 2)).reshape(output_shape)

# non maximum suppression
def heatmap_nms(heatmaps, pooled_heatmaps):
    return heatmaps * (heatmaps == pooled_heatmaps)

# Get poses from results.
def process_results(img, pafs, heatmaps):
    # This processing comes from
    # https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/common/python/models/open_pose.py
    pooled_heatmaps = np.array(
        [[pool2d(h, kernel_size=3, stride=1, padding=1, pool_mode="max") for h in heatmaps[0]]]
    )
    nms_heatmaps = heatmap_nms(heatmaps, pooled_heatmaps)

    # Decode poses.
    poses, scores = decoder(heatmaps, nms_heatmaps, pafs)
    output_shape = list(compiled_model.output(index=0).partial_shape)
    output_scale = img.shape[1] / output_shape[3].get_length(), img.shape[0] / output_shape[2].get_length()
    # Multiply coordinates by a scaling factor.
    poses[:, :, :2] *= output_scale
    return poses, scores

colors = ((255, 0, 0), (255, 0, 255), (170, 0, 255), (255, 0, 85), (255, 0, 170), (85, 255, 0),
          (255, 170, 0), (0, 255, 0), (255, 255, 0), (0, 255, 85), (170, 255, 0), (0, 85, 255),
          (0, 255, 170), (0, 0, 255), (0, 255, 255), (85, 0, 255), (0, 170, 255))

default_skeleton = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7),
                    (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))

def draw_poses(img, poses, point_score_threshold, skeleton=default_skeleton):
    if poses.size == 0:
        return img

    img_limbs = np.copy(img)
    for pose in poses:
        points = pose[:, :2].astype(np.int32)
        points_scores = pose[:, 2]
        # Draw joints.
        for i, (p, v) in enumerate(zip(points, points_scores)):
            if v > point_score_threshold:
                cv2.circle(img, tuple(p), 1, colors[i], 2)
        # Draw limbs.
        for i, j in skeleton:
            if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
                cv2.line(img_limbs, tuple(points[i]), tuple(points[j]), color=colors[j], thickness=4)
    cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
    return img

# Main processing function to run pose estimation.
def run_pose_estimation(source=0, flip=False, use_popup=False, skip_first_frames=0):
    pafs_output_key = compiled_model.output("Mconv7_stage2_L1")
    heatmaps_output_key = compiled_model.output("Mconv7_stage2_L2")
    player = None
    try:
        # Create a video player to play with target fps.
        player = utils.VideoPlayer(source, flip=flip, fps=30, skip_first_frames=skip_first_frames)
        # Start capturing.
        player.start()
        if use_popup:
            title = "Press ESC to Exit"
            cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)

        processing_times = collections.deque()

        while True:
            # Grab the frame.
            frame = player.next()
            if frame is None:
                print("Source ended")
                break
            # If the frame is larger than full HD, reduce size to improve the performance.
            scale = 1280 / max(frame.shape)
            if scale < 1:
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            # Resize the image and change dims to fit neural network input.
            # (see https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/human-pose-estimation-0001)
            input_img = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            # Create a batch of images (size = 1).
            input_img = input_img.transpose((2,0,1))[np.newaxis, ...]

            # Measure processing time.
            start_time = time.time()
            # Get results.
            results = compiled_model([input_img])
            stop_time = time.time()

            pafs = results[pafs_output_key]
            heatmaps = results[heatmaps_output_key]
            # Get poses from network results.
            poses, scores = process_results(frame, pafs, heatmaps)

            # Use processing times from last 200 frames.
            if len(processing_times) > 200:
                processing_times.popleft()

            _, f_width = frame.shape[:2]
            # mean processing time [ms]
            processing_time = np.mean(processing_times) * 1000
            fps = 1000 / processing_time

            # Preprocess the input frame
            input_mean = np.array([123.675, 116.28 , 103.53]).reshape(1, 3, 1, 1)
            input_scale = np.array([58.395, 57.12 , 57.375]).reshape(1, 3, 1, 1)
            resized_frame = cv2.resize(src=frame, dsize=(512, 512))
            input_frame = np.expand_dims(np.transpose(resized_frame, (2, 0, 1)), 0)
            input_frame = (input_frame - input_mean) / input_scale
        
            # Do inference on the input frame
            result = compiled_model_ir([input_frame])[0]
        
            # Resize the network result to the frame shape
            resized_result = np.rint(cv2.resize(src=np.squeeze(result), dsize=(frame.shape[1], frame.shape[0]))).astype(np.uint8)
        
            # Create a copy of the frame 
            bg_removed_frame = frame.copy()
            if poses.shape[0] != 0:
                if poses[-1][10][-1] != 0 and poses[-1][9][1] != 0:
                    if poses[-1][10][1] <= poses[-1][8][1]: # Left hands up
                        background_image = background_image1
                        bg_removed_frame[resized_result == 0] = background_image[resized_result == 0]
                if poses[-1][10][-1] != 0 and poses[-1][9][1] != 0:
                    if poses[-1][9][1] <= poses[-1][7][1]: # Right hands up
                        background_image = background_image2
                        bg_removed_frame[resized_result == 0] = background_image[resized_result == 0] 
                if poses[-1][10][-1] != 0 and poses[-1][9][1] != 0:
                    if poses[-1][10][1] <= poses[-1][8][1] and poses[-1][9][1] <= poses[-1][7][1]:
                        background_image = background_image3
                        bg_removed_frame[resized_result == 0] = background_image[resized_result == 0]
            # Use this workaround if there is flickering.
            if use_popup:
                cv2.imshow(title, frame)
                cv2.imshow("Frame with New Background", bg_removed_frame)
                key = cv2.waitKey(1)

                # capture = 99('c')
                if key == 99:
                    cv2.imwrite("capture.jpg", bg_removed_frame)
                    print("image captured!")
                # escape = 27
                if key == 27:
                    break
            else:
                # Encode numpy array to jpg.
                _, encoded_img = cv2.imencode(".jpg", frame, params=[cv2.IMWRITE_JPEG_QUALITY, 90])
                # Create an IPython image.
                i = display.Image(data=encoded_img)
                # Display the image in this notebook.
                display.clear_output(wait=True)
                display.display(i)
    # ctrl-c
    except KeyboardInterrupt:
        print("Interrupted")
    # any different error
    except RuntimeError as e:
        print(e)
    finally:
        if player is not None:
            # Stop capturing.
            player.stop()
        if use_popup:
            cv2.destroyAllWindows()
         
USE_WEBCAM = True
cam_id = 5
video_file = "https://github.com/intel-iot-devkit/sample-videos/blob/master/store-aisle-detection.mp4?raw=true"
source = cam_id if USE_WEBCAM else video_file

additional_options = {"skip_first_frames": 500} if not USE_WEBCAM else {}
run_pose_estimation(source=source, flip=isinstance(source, int), use_popup=True, **additional_options)