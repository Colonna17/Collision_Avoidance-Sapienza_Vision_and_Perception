
import torch

def load_yolov5_model():
    # Model
    yolov5_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom
    return yolov5_model

def main():

    model = load_yolov5_model() 
    
    # Images
    img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

    # Inference
    results = model(img)

    # Results
    results.show()  # or .show(), .save(), .crop(), .pandas(), etc.
    
    print('## Ok bye ##')
    

if __name__ == "__main__":
    main()
    
