import os
import sys
import argparse
from pathlib import Path

import torch

sys.path.append('external/yolov5')
from external.yolov5.models.common import DetectMultiBackend


# sys.path.append('external/Yolov5_StrongSORT_OSNet')
sys.path.append('external/Yolov5_StrongSORT_OSNet/strong_sort/deep/reid')
from external.Yolov5_StrongSORT_OSNet.track import *
# WEIGHTS = './weights/'

# these are the only classes that will be detected and tracked
DEFAULT_CLASSES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck', 9: 'traffic light', 11: 'stop sign'}

def load_yolov5_model_from_torchhub(model_name = 'yolov5n'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    yolov5_model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)#, device=device)  # or yolov5n - yolov5x6, custom
    
    yolov5_model.eval() 
       
    return yolov5_model

def run(
        yolo_weights = None,
        classes = None,
        device = '',
        
        weights_from_torchhub = False
        ):
    
    if(not yolo_weights):
        print('Error: YOLOv5 weights not specified')
        return -1
    torch.no_grad()
    
    if(weights_from_torchhub):
        yolov5_model = load_yolov5_model_from_torchhub(yolo_weights) 
    else:
        # print(yolo_weights)
        # return
        yolov5_model = DetectMultiBackend(yolo_weights, device=device, dnn=False, data=None, fp16=False)
    
    # Images
    # img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list
    img = './data/traffico.jpg'
    # Inference
    results = yolov5_model(img)

    # Results
    results.show()  # or .show(), .save(), .crop(), .pandas(), etc.
    
    
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=Path, default='weights/yolov5l_finetuned_best_12.pt', help='model.pt path OR the torchhub name model of ...')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3', default=list(DEFAULT_CLASSES.keys()))
    parser.add_argument('--device', default='', help='')
    # TODO: add the option to download the yolo weights from torch hub
    opt = parser.parse_args()
    return opt


def main(opt):
    # TODO: check requirements before run
    run(**vars(opt))
    

if __name__ == "__main__":
    opt = parse_opt()
    
    # print('\n\n')
    # print(type(opt), '\n', opt, '\n\n')
    # print(type(vars(opt)), '\n', vars(opt), '\n\n')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ', str(device), '\n')
    
    main(opt)
