import torch
import numpy as np
import argparse
from pathlib import Path


# Transform a numpy array of an image from bgr to rgb format
def numpy_brg_to_rgb(np_array):
    return np.fliplr(np_array.reshape(-1,3)).reshape(np_array.shape).copy()

def custom_from_numpy(np_array, device):
    return torch.from_numpy(np_array).to(device=device, dtype=torch.float32)


def parse_opt():
    parser = argparse.ArgumentParser()
    # General options
    parser.add_argument('--source', type=str, default=DEFAULT_OPTIONS['source'], help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--img-height', type=int, default=DEFAULT_OPTIONS['img_height'], help='loaded images height')
    parser.add_argument('--img-width', type=int, default=DEFAULT_OPTIONS['img_width'], help='loaded images width')

    # Yolo options
    parser.add_argument('--yolo-weights', type=Path, default=DEFAULT_OPTIONS['yolo_weights'], help='model.pt path of yolov5')
    # parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3', default=DEFAULT_OPTIONS['classes'])
    parser.add_argument('--conf-thres', type=float, default=DEFAULT_OPTIONS['conf_thres'], help='Yolo confidence threshold')
    parser.add_argument('--max-det', type=int, default=DEFAULT_OPTIONS['max_det'], help='Maximum number of detected/tracked objects')
    parser.add_argument('--yolo-img-height', type=int, default=DEFAULT_OPTIONS['yolo_img_height'], help='loaded images height')
    parser.add_argument('--yolo-img-width', type=int, default=DEFAULT_OPTIONS['yolo_img_width'], help='loaded images width')

    # Tracker options
    parser.add_argument('--strong-sort-weights', type=Path, default=DEFAULT_OPTIONS['strong_sort_weights'], help='Model weights for the tracker')

    # Classifier options
    parser.add_argument('--lstm-hidden-dim', type=int, default=DEFAULT_OPTIONS['lstm_hidden_dim'], help='Hidden layer dimension of the LSTM inside the collision classifier')
    parser.add_argument('--classifier-weights', type=Path, default=DEFAULT_OPTIONS['classifier_weights'], help='path for the model.pt of the clasifier')
    

    opt = parser.parse_args()
    return vars(opt)

# default classes that will be detected and tracked
# DEFAULT_CLASSES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck', 9: 'traffic light', 11: 'stop sign'}

DEFAULT_OPTIONS = {'source': 'data/videos/CCD/100096.mp4', 
                   'show_vid': False, 
                   'save_vid': False, 
                   'save_txt': False, 
                   'img_height': 720, 
                   'img_width': 1280, 
                   # Yolo options:
                   'yolo_weights': Path('weights/yolov5l_finetuned_best_12.pt'), 
                   'classes': [0, 1, 2, 3, 5, 7, 9, 11], 
                   'conf_thres': 0.25, 
                   'max_det': 150, 
                   'yolo_img_height': 640, 
                   'yolo_img_width': 640, 
                   # Tracker options:
                   'strong_sort_weights': Path('weights/osnet_x0_25_msmt17.pt'), 
                   # Classifier options:
                   'lstm_hidden_dim': 8,
                   'classifier_weights': Path('weights/Crash_Classfier_weights.pt')
                   }
