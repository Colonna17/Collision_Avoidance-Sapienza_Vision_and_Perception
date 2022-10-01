#!/bin/bash

echo 'Installing requirements...'
echo 'Installing YOLOv5 requirements'
pip3 install -qr external/yolov5/requirements.txt
pip3 install -qr external/Yolov5_StrongSORT_OSNet/requirements.txt
echo 'Installing Yolov5_StrongSORT_OSNet requirements'
pip3 install -qr requirements.txt
echo 'Installing our custom requirements'
echo 'Done!'
