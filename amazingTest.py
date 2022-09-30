
import torch
import pandas as pd

def load_yolov5_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model_args = {}
    # model_args['device'] = device
    # Model
    yolov5_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')#, device=device)  # or yolov5n - yolov5x6, custom
    yolov5_model.eval() 
       
    return yolov5_model

def main():

    model = load_yolov5_model() 
    
    # Images
    # img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list
    img = './data/images/traffic.jpg'
    # Inference
    results = model(img)

    # Results
    print(results)
    print(results.pd().xyxy[0])
    print(results.names)
    results.show()  # or .show(), .save(), .crop(), .pandas(), etc.


if __name__ == "__main__":
    main()
    
