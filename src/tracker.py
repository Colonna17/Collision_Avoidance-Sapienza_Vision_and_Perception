import sys
import torch, numpy

# imports from Yolov5_StrongSORT_OSNet
sys.path.append('external/Yolov5_StrongSORT_OSNet')
sys.path.append('external/Yolov5_StrongSORT_OSNet/strong_sort/deep/reid')
from external.Yolov5_StrongSORT_OSNet.strong_sort.strong_sort import StrongSORT

from external.yolov5.utils.general import scale_coords, xyxy2xywh
# (LOGGER, Profile, check_img_size, non_max_suppression, check_requirements, cv2, check_imshow, increment_path, strip_optimizer, colorstr, print_args, check_file) 

class params_strongSort():  
    ecc = True               # activate camera motion compensation
    mc_lambda = 0.995        # matching with both appearance (1 - MC_LAMBDA) and motion cost
    ema_alpha = 0.9          # updates  appearance  state in  an exponential moving average manner
    max_dist = 0.2           # The matching threshold. Samples with larger distance are considered an invalid match
    max_iou_distance = 0.7   # Gating threshold. Associations with cost larger than this value are disregarded.
    max_age = 30             # Maximum number of missed misses before a track is deleted
    n_init = 3               # Number of frames that a track remains in initialization phase
    nn_budget = 100          # Maximum size of the appearance descriptors gallery    

class Tracker():
    def __init__(self, device, options):
        self.device = device
        self.cfg = params_strongSort()
        half = False # half precision (not considered so far)
        self.strong_sort = StrongSORT(
                options['strong_sort_weights'],
                device,
                half,
                max_dist = self.cfg.max_dist,
                max_iou_distance = self.cfg.max_iou_distance,
                max_age = self.cfg.max_age,
                n_init = self.cfg.n_init,
                nn_budget = self.cfg.nn_budget,
                mc_lambda = self.cfg.mc_lambda,
                ema_alpha = self.cfg.ema_alpha                
            )
        self.strong_sort.model.warmup()

    def track(self, curr_frame, prev_frame, detections, scaled_img_size):
        if(self.cfg.ecc):  # Camera motion compensation
            self.strong_sort.tracker.camera_update(prev_frame, curr_frame)
        
        detections[:, :4] = scale_coords(scaled_img_size, detections[:, :4], curr_frame.shape).round()
        xywhs = xyxy2xywh(detections[:, 0:4]) # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
        confs = detections[:, 4]
        clss = detections[:, 5]
        
        try: # ToDO: Fast and weak solution. Find a final solution for this problem
            outputs = self.strong_sort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), curr_frame)
        except:
            outputs = []

        if(len(outputs) == 0):
            # return torch.empty(0, device=self.device, dtype=torch.float32)
            return numpy.empty(0)
        # return torch.from_numpy(outputs).to(self.device)
        return outputs
