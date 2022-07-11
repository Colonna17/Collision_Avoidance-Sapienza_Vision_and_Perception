#!/bin/bash

conda env create -n vision -f environment.yml

mkdir external/bevfusion/pretrained
cd external/bevfusion/pretrained
wget https://bevfusion.mit.edu/files/pretrained/bevfusion-det.pth

cd ..
source ~/anaconda3/etc/profile.d/conda.sh
conda activate vision
python3 setup.py develop