#!/bin/bash

echo 'Installing requirements...'
echo 'Installing YOLOv5 requirements'
pip install -qr external/yolov5/requirements.txt # --force-reinstall
echo 'Installing Yolov5_StrongSORT_OSNet requirements'
pip install -qr external/Yolov5_StrongSORT_OSNet/requirements.txt
echo 'Installing our custom requirements'
pip install -qr requirements.txt 
echo 'Done!'