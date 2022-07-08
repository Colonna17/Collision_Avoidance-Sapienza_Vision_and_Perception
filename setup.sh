#!/bin/bash

conda create -n vision python=3.8
conda activate vision
pip install -r requirements.txt

# pip install openmim
# mim install mmcv-full==1.4.8
# mim install mmdet
mim install mmsegmentation
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d/
pip install -e .

rm -r  mmdetection3d