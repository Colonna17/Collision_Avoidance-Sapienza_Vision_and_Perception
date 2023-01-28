# Computer Vision Project 🚀
'''
Code based on:
https://github.com/ultralytics/yolov5
https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet
'''

import sys
import numpy as np
import torch
import cv2

# imports from yolov5
sys.path.append('external/yolov5')
from external.yolov5.utils.dataloaders import LoadImages

# imports from Yolov5_StrongSORT_OSNet
sys.path.append('external/Yolov5_StrongSORT_OSNet')
sys.path.append('external/Yolov5_StrongSORT_OSNet/strong_sort/deep/reid')

from src.utils import parse_opt, custom_from_numpy
from src.build import build


@torch.no_grad()
def run(device, options, yolo, tracker, classifier):
    source = str(options['source'])
    img_size = (options['yolo_img_height'], options['yolo_img_width'])
    dataset = LoadImages(source, img_size=img_size, stride=yolo.model.stride, auto=yolo.model.pt)

    # For visualization
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    label_position = (50, 50)
    label_thickness = 3   
    safe_color = (0, 255, 0)    
    collision_color = (255, 0, 0) 
    output_path = 'results/final_classification/' + source[source.rindex('/')+1:]
    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (options['img_width'], options['img_height']))
    
    print('\nClassification output: ')
    curr_frame, prev_frame = None, None
    h = None
    for frame_idx, (path, img_scaled, img, vid_cap, s) in enumerate(dataset):
        curr_frame = img.copy()
        curr_frame_scaled = custom_from_numpy(img_scaled, device).unsqueeze(0)
        yolo_detections = yolo.detect(curr_frame_scaled)
        tracking_output = tracker.track(curr_frame, prev_frame, yolo_detections, curr_frame_scaled.shape[2:])
        
        if(frame_idx > 0): 
            collision, h = classifier(curr_frame, prev_frame, tracking_output, h)
            print('Frame ', frame_idx, ': ', collision.item())
            
            prediction = round(collision.item())
            text = 'Collision' if prediction == 1 else 'Safe'
            color = collision_color if prediction == 1 else safe_color
            image = cv2.putText(curr_frame, text, label_position, font, fontScale, color, label_thickness)
            video.write(image)
            
        prev_frame = curr_frame
    
    video.release()
    print('\nVideo saved in: ', output_path)
    
    return('Done')


def main(options):
    print(options)

    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    print('Device: ', str(device), '\n')

    yolo, tracker, classifier = build(device, options)
    classifier.load_state_dict(torch.load(options['classifier_weights']))
    print(type(yolo))
    print(type(tracker))
    print(type(classifier))
    
    out = run(device, options, yolo, tracker, classifier)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)