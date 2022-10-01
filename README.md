## Collision_Avoidance-Sapienza_Vision_and_Perception
Sapienza University project for Vision and Perception class 2022. Students: Edoardo Colonna, Lapo Carrieri and Silverio Manganaro
# Abstract
Nowadays one of the hottest topics in the AI area is autonomous driving. It heavily relies on new technologies from computer vision to achieve results that otherwise would not make it possible to use self-driving cars. However, many improvements are still needed in this field. Our project aims to develop a method for 2D/3D object detection in autonomous driving during adverse weather conditions. We'll start by comparing our results with actual benchmarks by using pre-trained models. Then, by using outputs from the first part of the work, we'll focus on estimating object orientation and velocity in order to generate an original collision avoidance method, in line with current literature on the subject.
# Dataset
We'll try to obtain a model having similar performance with respect to the following dataset:

-DENSE (https://www.uni-ulm.de/en/in/driveu/projects/dense-datasets#c8116699), a dataset robust to "all weather conditions"; it has 100k Objects labelled with accurate 2D and 3D bounding boxes

-CCD (Car Crash Dataset)(https://github.com/Cogito2012/CarCrashDataset) given its specialisation in traffic accident scenes; has more than 1500 accident videos with labels and accident reasons descriptions

# Environment setup

Run the following command only the first time:
```bash
./setup.sh
```
It will install all the requirements needed by our code and by all submodules.
<!-- It will create a conda environment called **vision**. -->

# Test
<!-- Activate the environment:
```bash
conda activate vision
``` -->
You can run our amazing test and check if everything works:
```python
python3 amazingTest.py
```

It is also available a full tutorial here: <b> [bottone con il notebook da creare] </b>