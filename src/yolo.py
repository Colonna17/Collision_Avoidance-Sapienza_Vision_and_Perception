
import sys
import torch

sys.path.append('external/yolov5')
from external.yolov5.models.common import DetectMultiBackend
from external.yolov5.utils.general import non_max_suppression

class Yolo():
    def __init__(self, device, options):
        self.device = device
        if options['yolo_torchhub']:
            self.model = torch.hub.load('ultralytics/yolov5', yolo_model_name, pretrained=True, device=device).model #, device=device)  # or yolov5n - yolov5x6, custom
        else:
            self.model = DetectMultiBackend(options['yolo_weights'], device=device, dnn=False, data=None, fp16=False)
        # self.stride, self.names, self.pt= self.model.stride, self.model.names, self.model.pt
        self.conf_thres = options['conf_thres']
        self.classes = options['classes']
        self.max_det = options['max_det']
        self.model.warmup((1,3,options['yolo_img_height'], options['yolo_img_width']))

    def detect(self, image):
        preds = self.model(image)

        # Apply Non-Max Suppression to the predictions
        iou_thres = 0.45 # (not considered so far)
        agnostic_nms = False # (not considered so far)
        preds = non_max_suppression(preds, self.conf_thres, iou_thres, self.classes, agnostic_nms, self.max_det)[0]
        
        return preds

