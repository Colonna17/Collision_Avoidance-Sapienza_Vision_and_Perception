
from torchpack.utils.config import configs
from mmcv import Config
from mmcv.runner import load_checkpoint

from external.bevfusion.mmdet3d.models import build_model
from external.bevfusion.mmdet3d.utils import recursive_eval


def main():
    config_filename = 'external/bevfusion/configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml'
    checkpoint_filename = 'external/bevfusion/pretrained/bevfusion-det.pth'
    
    configs.load(config_filename, recursive=True)
    cfg = Config(recursive_eval(configs), filename=config_filename)
    
    model = build_model(cfg.model)
    checkpoint = load_checkpoint(model, checkpoint_filename, map_location="cpu")
    
    print(checkpoint.keys())
    
    print('## Ok bye ##')
    

if __name__ == "__main__":
    main()
