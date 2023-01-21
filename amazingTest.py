# Computer Vision Project 🚀
'''
Code based on:
https://github.com/ultralytics/yolov5
https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet
'''

import os
import sys
# import argparse
# from pathlib import Path
import numpy as np
import torch

# imports from yolov5
sys.path.append('external/yolov5')
from external.yolov5.utils.dataloaders import LoadImages, LoadStreams, VID_FORMATS, IMG_FORMATS
# from external.yolov5.utils.general import (LOGGER, Profile, check_img_size, non_max_suppression, scale_coords, check_requirements, cv2, check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
# from external.yolov5.utils.plots import Annotator, colors, save_one_box
# from external.yolov5.utils.torch_utils import time_sync

# imports from Yolov5_StrongSORT_OSNet
sys.path.append('external/Yolov5_StrongSORT_OSNet')
sys.path.append('external/Yolov5_StrongSORT_OSNet/strong_sort/deep/reid')
# from external.Yolov5_StrongSORT_OSNet.strong_sort.strong_sort import StrongSORT

from src.utils import parse_opt, custom_from_numpy
# from src.hparams import *
from src.build import build


@torch.no_grad()
def run(device, options, yolo, tracker, classifier):
    source = str(options['source'])

    img_size = (options['yolo_img_height'], options['yolo_img_width'])
    dataset = LoadImages(source, img_size=img_size, stride=yolo.model.stride, auto=yolo.model.pt)

    curr_frame, prev_frame = None, None
    h = None
    for frame_idx, (path, img_scaled, img, vid_cap, s) in enumerate(dataset):
        print('### ', str(frame_idx), ' ###')
        curr_frame = img.copy()
        # curr_frame_scaled = custom_from_numpy(img_scaled, device).unsqueeze(0)
        curr_frame_scaled = torch.from_numpy(img_scaled).to(device, dtype=torch.float32).unsqueeze(0)
        
        yolo_detections = yolo.detect(curr_frame_scaled)
        tracking_output = tracker.track(curr_frame, prev_frame, yolo_detections, curr_frame_scaled.shape[2:])
        # print(type(yolo_detections), yolo_detections.shape, yolo_detections)
        # print(type(tracking_output), tracking_output.shape, tracking_output)
        
        if(frame_idx > 0): 
            collision, h = classifier(curr_frame, prev_frame, tracking_output, h)
            print(collision)

        # ToDo: Display output
        
        prev_frame = curr_frame
    return('Done')


def main(options):
    print(options)

    device = torch.device('cpu') # torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    print('Device: ', str(device), '\n')

    yolo, tracker, classifier = build(device, options)
    print(type(yolo))
    print(type(tracker))
    print(type(classifier))
    
    out = run(device, options, yolo, tracker, classifier)
    print(out)


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)