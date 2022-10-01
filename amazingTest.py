
import torch

def load_yolov5_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Model loading
    yolov5_model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)#, device=device)  # or yolov5n - yolov5x6, custom
    yolov5_model.eval() 
       
    return yolov5_model

def main():

    model = load_yolov5_model() 
    
    # Input
    # img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list
    img = './data/images/traffic.jpg'
    
    # Inference
    results = model(img)

    # Results
    print(results)
    print(results.pandas().xyxy[0])
    print(results.names)
    results.show()  # or .show(), .save(), .crop(), .pandas(), etc.
    

if __name__ == "__main__":
    main()
    
