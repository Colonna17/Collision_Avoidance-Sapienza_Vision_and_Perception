from src.yolo import Yolo
from src.tracker import Tracker
from src.classificator import Classificator

def build(device, options):
    
    yolo = Yolo(device, options)

    tracker = Tracker(device, options)

    classificator = Classificator(device, options)

    return (yolo, tracker, classificator)
