from src.yolo import Yolo
from src.tracker import Tracker
from src.classifier import Classifier

def build(device, options):
    
    yolo = Yolo(device, options)

    tracker = Tracker(device, options)

    classifier = Classifier(device, options)

    return (yolo, tracker, classifier)
