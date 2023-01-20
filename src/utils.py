import torch
import numpy as np
import argparse
from pathlib import Path

# default classes that will be detected and tracked
DEFAULT_CLASSES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck', 9: 'traffic light', 11: 'stop sign'}

# Transform a numpy array of an image from bgr to rgb format
def numpy_brg_to_rgb(np_array):
    return np.fliplr(np_array.reshape(-1,3)).reshape(np_array.shape).copy()

def custom_from_numpy(np_array, device):
    return torch.from_numpy(np_array).to(device=device, dtype=torch.float32)

def save_or_show(options):

    return True

def parse_opt():
    parser = argparse.ArgumentParser()
    # General options
    parser.add_argument('--source', type=str, default='data/videos/CCD/000017.mp4', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--img-height', type=int, default=720, help='loaded images height')
    parser.add_argument('--img-width', type=int, default=1280, help='loaded images width')

    # Yolo options
    parser.add_argument('--yolo-weights', type=Path, default='weights/yolov5l_finetuned_best_12.pt', help='model.pt path OR the torchhub name model of ...')
    # parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3', default=list(DEFAULT_CLASSES.keys()))
    parser.add_argument('--yolo-torchhub', action='store_true', help='Specify if you want to download the yolo weights from PyTorchHub. \nOtherwise remember to specify the local path of where to find the weights you want to use (see the option --yolo-weights)')
    parser.add_argument('--yolo-model-name', type=str, default='yolov5m', help='Official yolov5 model weights: yolov5n, yolov5s, yolov5m, yolov5l, yolov5x. \nIgnored if not --yolo-torchhub')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Yolo confidence threshold')
    parser.add_argument('--max-det', type=int, default=100, help='Maximum number of detected/tracked objects')
    parser.add_argument('--yolo-img-height', type=int, default=640, help='loaded images height')
    parser.add_argument('--yolo-img-width', type=int, default=640, help='loaded images width')

    # Tracker options
    parser.add_argument('--strong-sort-weights', type=Path, default='weights/osnet_x0_25_msmt17.pt', help='Model weights for the tracker')

    # Classificator options
    parser.add_argument('--lstm-hidden-dim', type=int, default=32, help='Hidden layer dimension of the LSTM inside the collision classificator')
    

    opt = parser.parse_args()
    return vars(opt)