# Computer Vision Project 🚀
'''
Code based on:
https://github.com/ultralytics/yolov5
https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet
'''

import os
import sys
import argparse
from pathlib import Path
import numpy
import torch
import torch.backends.cudnn as cudnn
import logging

from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose', color_scheme='Linux', call_pdb=False)

# imports from yolov5
sys.path.append('external/yolov5')
from external.yolov5.models.common import DetectMultiBackend
from external.yolov5.utils.dataloaders import LoadImages, LoadStreams, VID_FORMATS, IMG_FORMATS
from external.yolov5.utils.general import (LOGGER, Profile, check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
from external.yolov5.utils.plots import Annotator, colors, save_one_box
# from external.yolov5.utils.torch_utils import time_sync

# imports from Yolov5_StrongSORT_OSNet
sys.path.append('external/Yolov5_StrongSORT_OSNet')
sys.path.append('external/Yolov5_StrongSORT_OSNet/strong_sort/deep/reid')
from external.Yolov5_StrongSORT_OSNet.strong_sort.strong_sort import StrongSORT

# remove dulicated logging
nr_loggers = len(logging.getLogger().handlers)
while nr_loggers > 1:
    logging.getLogger().removeHandler(logging.getLogger().handlers[0]) 
    nr_loggers -=1 

# default classes that will be detected and tracked
DEFAULT_CLASSES = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck', 9: 'traffic light', 11: 'stop sign'}

class params_strongSort():  
    ecc = True              # activate camera motion compensation
    mc_lambda = 0.995       # matching with both appearance (1 - MC_LAMBDA) and motion cost
    ema_alpha = 0.9         # updates  appearance  state in  an exponential moving average manner
    max_dist = 0.2           # The matching threshold. Samples with larger distance are considered an invalid match
    max_iou_distance = 0.7   # Gating threshold. Associations with cost larger than this value are disregarded.
    max_age = 30             # Maximum number of missed misses before a track is deleted
    n_init = 3               # Number of frames that a track remains in initialization phase
    nn_budget = 100          # Maximum size of the appearance descriptors gallery    


@torch.no_grad()
def run(
        source = '0',
        yolo_weights = Path('weights/yolov5l_finetuned_best_12.pt').resolve(),
        classes = list(DEFAULT_CLASSES.keys()),
        yolo_torchhub = False,
        yolo_model_name = '',
        strong_sort_weights=Path('weights/osnet_x0_25_msmt17.pt').resolve(),
        # TODO: review the following arguments
        imgsz = (640, 640),
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        agnostic_nms=False,  # class-agnostic NMS
        max_det=1000,  # maximum detections per image
        line_thickness = 2,
        augment = False,  # augmented inference
        half = False,
        hide_labels = False,
        hide_conf = False,
        hide_class = False,
        show_vid = False,
        project = Path('results').resolve(),
        exp_name = 'exp', # experiment name
        save_vid = False,
        save_txt = False,
        
        ):
    source = str(source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    stream = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)
    exp_name = exp_name if exp_name else yolo_weights.stem + '_' + strong_sort_weights.stem
    save_dir = increment_path(Path(project) / exp_name)
    makedir = save_vid or save_txt
    if makedir: (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    
    # Select device
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    print('Device: ', str(device), '\n')
    
    # Load yolov5 model
    if yolo_torchhub:
        yolov5_model = torch.hub.load('ultralytics/yolov5', yolo_model_name, pretrained=True, device=device).model #, device=device)  # or yolov5n - yolov5x6, custom
    else:
        yolov5_model = DetectMultiBackend(yolo_weights, device=device, dnn=False, data=None, fp16=False)
    yolov5_model.eval()
    stride, names, pt = yolov5_model.stride, yolov5_model.names, yolov5_model.pt
    # Dataloader
    if stream: # webcam, txt file or url of a stream
        show_vid = check_imshow() # Check if environment supports image displays
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = len(dataset)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1
    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # Initialize StrongSORT
    cfg = params_strongSort()
    
    # Create as many strong sort instances as there are video sources
    strongsort_list = []
    for i in range(nr_sources):
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                half,
                max_dist = cfg.max_dist,
                max_iou_distance = cfg.max_iou_distance,
                max_age = cfg.max_age,
                n_init = cfg.n_init,
                nn_budget = cfg.nn_budget,
                mc_lambda = cfg.mc_lambda,
                ema_alpha = cfg.ema_alpha                
            )
        )
        strongsort_list[i].model.warmup()
    outputs = [None] * nr_sources
    
    # Tracking
    yolov5_model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    seen = 0
    dt = [0.0, 0.0, 0.0, 0.0] # for SPEED evaluation # TODO add one element for our classificator
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        # TODO t1 = time_sync() # for SPEED evaluation
        # cv2.imshow('test', im[0])
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        im = torch.from_numpy(im).to(device)
        im = im.float() / 255.0 # casting and normalization
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # TODO t2 = time_sync() # for SPEED evaluation
        # TODO dt[0] += t2 - t1 # dt[0] = total amount of time spent on loading frames
        pred = yolov5_model(im) # YOLOv5 model in validation model, output = (inference_out, loss_out)
        # TODO t3 = time_sync() # for SPEED evaluation
        # TODO dt[1] += t3 - t2 # dt[1] = total amount of time spent on object detection (YOLOv5)
        
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        # TODO dt[2] += time_sync() - t3 # dt[2] = total amount of time spent on NMS
        # Process detections
        for i, detection in enumerate(pred):
            seen += 1
            if stream:  # nr_sources >= 1
                p, im0 = path[i], im0s[i].copy()
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0 = path, im0s.copy()
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            
            annotator = Annotator(im0, line_width=line_thickness, pil=not ascii)
            if cfg.ecc:  # camera motion compensation
                strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])
                
            if detection is not None and len(detection):
                # Rescale boxes from img_size to im0 size
                detection[:, :4] = scale_coords(im.shape[2:], detection[:, :4], im0.shape).round()
                            
                # Print results
                for c in detection[:, -1].unique():
                    n = (detection[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(detection[:, 0:4]) # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
                confs = detection[:, 4]
                clss = detection[:, 5]
                
                # pass detections to strongsort
                # TODO t4 = time_sync() # for SPEED evaluation
                outputs[i] = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                # TODO dt[3] += time_sync() - t4 # dt[3] = total amount of time spent on tracking (StrongSORT)
                
                # TODO our amazing classificato
                # TODO draw something on the video based on the output of the classificator
                
                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    for j, (output, conf) in enumerate(zip(outputs[i], confs)):
    
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]

                        if save_txt:
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left, bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                        if save_vid or show_vid:  # Add bbox to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                            annotator.box_label(bboxes, label, color=colors(c, True))                            

            else:
                strongsort_list[i].increment_ages()
            
            # Stream results
            im0 = annotator.result()
            if show_vid:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)
                
            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]
            
    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=Path, default='weights/yolov5l_finetuned_best_12.pt', help='model.pt path OR the torchhub name model of ...')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3', default=list(DEFAULT_CLASSES.keys()))
    # parser.add_argument('--device', default='0', help='Cuda device index')
    parser.add_argument('--yolo-torchhub', action='store_true', help='Specify if you want to download the yolo weights from PyTorchHub. \nOtherwise remember to specify the local path of where to find the weights you want to use (see the option --yolo-weights)')
    
    # TODO: do not set the default if not yolo-tprchhub
    parser.add_argument('--yolo-model-name', type=str, default='yolov5m', help='Official yolov5 model weights: yolov5n, yolov5s, yolov5m, yolov5l, yolov5x. \nIgnored if not --yolo-torchhub')
    
    parser.add_argument('--source', type=str, default='data/videos/CCD/000017.mp4', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--show-vid', action='store_true', help='shows the result')
    opt = parser.parse_args()
    return opt


def main(opt):
    # TODO: check requirements before run
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
