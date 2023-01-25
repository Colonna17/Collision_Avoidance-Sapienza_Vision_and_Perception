## Collision_Avoidance-Sapienza_Vision_and_Perception
Sapienza University project for Vision and Perception class 2022. Students: Edoardo Colonna, Lapo Carrieri and Silverio Manganaro
# Abstract
Nowadays one of the hottest topics in the AI area is autonomous driving. It heavily relies on new technologies from computer vision to achieve results that otherwise would not make it possible to use self-driving cars. However, many improvements are still needed in this field. Our project aims to develop a method for 3D object detection in autonomous driving. We'll start by comparing our results with actual benchmarks by using pre-trained models. Then, by using outputs from the first part of the work, we will train a model to track the vehicles frame by frame. Finally, we will focus on estimating object orientation and velocity in order to generate a collision avoidance method, in line with current literature on the subject.
# Dataset
We'll try to obtain a model having similar performance with respect to the following dataset:

-DENSE ([https://universe.roboflow.com/eddyprojects/v-p-2022/dataset/9]), a dataset created by ourself that contains "all weather conditions"; it has 1000 images with thousands of Objects labelled with accurate 2D bounding boxes using 7 classes 

-CCD (Car Crash Dataset)(https://github.com/Cogito2012/CarCrashDataset) given its specialisation in traffic accident scenes; has more than 1500 accident videos with labels and accident reasons descriptions

# Project
The project is divided in three main steps:
  ## Yolo custom training
  We assemble a dataset and train a custom YOLOv5 model to recognize the objects in our dataset. To do so we will take the following steps:
- Gather a dataset of images and label our dataset (using Roboflow)
- Export our dataset to YOLOv5
- Train YOLOv5 to recognize the objects in our dataset
- Evaluate our YOLOv5 model's performance 
- Run test inference to view our model at work
- Train many models with different hyperparameters and compare the obtained performance
- Download the weights corresponding to the best model obtained
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1BqAma0E9KdxTgOLPOhJ-2C7YkzzAeIHc)
## Tracking
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Colonna17/Collision_Avoidance-Sapienza_Vision_and_Perception/blob/main/notebooks/tracking.ipynb)
The second step to obtain a classifier is the use of Strongsort in order to keep track of the bounding boxes and mantain the indexes of them. This improve significally the performance of the model in respect to other computer vision models.
DeepSORT is a computer vision tracking algorithm for tracking objects while assigning an ID to each object. It uses
•	Detection: 
•	Estimation: 
Data association: 
•	Creation and Deletion of Track Identities:


## Car crash prevision
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Colonna17/Collision_Avoidance-Sapienza_Vision_and_Perception/blob/main/notebooks/CCD_Classification.ipynb)

The final step is the connection of all the models explained before, all goes directly into the classifier that create an output between 0 and 1 that represent the probability of an accident.

