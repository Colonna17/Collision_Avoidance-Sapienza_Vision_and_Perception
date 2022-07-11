#!/bin/bash

# conda create -y -n vision -f environment.yml

cd external/bevfusion/pretrained
wget https://bevfusion.mit.edu/files/pretrained/bevfusion-det.pth

# cd ..
# python3 setup.py develop