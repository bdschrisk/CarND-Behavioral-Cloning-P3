# Behavioural Cloning Project
### "Project Karl"
*__"My first customer was a lunatic. My second had a death wish."__*

## Overview
This project is intended to apply behavioural cloning using deep learning to drive a car around a track within a simulated environment.

*If you're looking for the reinforcement version of this project see the CarND-Behavioral-Cloning-P3-RL repo.*

#### Control Surface Approximation Function
For solving the control surface problem, I've designed a network loosely based on the LeNet architecture. The input feeds through 
input processing layers where it crops, normalises and resizes the input.  After preprocessing, repeating convolutional layers with 64, 
32, 16 and 8 filters of 5x5, 3x3 and 2x2 detectors, consecutively, extract meaningful features from the input.  An average pooling layer 
downsamples each convolution into rectifiers, for non-linearity, condensing the input into a final average pooling layer. The final 
pooling layer is responsible for extracting spatial respresentations from each convolution into a compressed output.  Hidden layers of 
sizes 256 and 128 with parametric rectifiers are computed, along with a dropout rate of 20%, for providing latent features for the final
linear Q layer, where each unit is a steering action.

##### KaNet Model

Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
input_1 (InputLayer)             (None, 160, 320, 3)   0
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 90, 320, 3)    0           input_1[0][0]
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 90, 320, 3)    0           cropping2d_1[0][0]
____________________________________________________________________________________________________
resize_1 (Resize)                (None, 45, 160, 3)    0           lambda_1[0][0]
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 41, 156, 64)   4864        resize_1[0][0]
____________________________________________________________________________________________________
averagepooling2d_1 (AveragePooli (None, 19, 77, 64)    0           convolution2d_1[0][0]
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 19, 77, 64)    0           averagepooling2d_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 17, 75, 32)    18464       activation_1[0][0]
____________________________________________________________________________________________________
averagepooling2d_2 (AveragePooli (None, 16, 74, 32)    0           convolution2d_2[0][0]
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 16, 74, 32)    0           averagepooling2d_2[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 15, 73, 16)    2064        activation_2[0][0]
____________________________________________________________________________________________________
averagepooling2d_3 (AveragePooli (None, 14, 72, 16)    0           convolution2d_3[0][0]
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 14, 72, 16)    0           averagepooling2d_3[0][0]
____________________________________________________________________________________________________
averagepooling2d_4 (AveragePooli (None, 7, 36, 16)     0           activation_3[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 4032)          0           averagepooling2d_4[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 256)           1032448     flatten_1[0][0]
____________________________________________________________________________________________________
prelu_1 (PReLU)                  (None, 256)           256         dense_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 256)           0           prelu_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 128)           32896       dropout_1[0][0]
____________________________________________________________________________________________________
prelu_2 (PReLU)                  (None, 128)           128         dense_2[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 128)           0           prelu_2[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 13)            1677        dropout_2[0][0]
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 13)            0           dense_4[0][0]
====================================================================================================
Total params: 1,092,797
Trainable params: 1,092,797
Non-trainable params: 0

### Training

### Files in this repo:
 - Behavioural-Cloning.ipynb (Jupyter notebook with project writeup)
 - extensions/* (custom keras modules used in model building)
 - callbacks.py (Custom callback to prevent overfitting)
 - densenet.py (DenseNet architecture)
 - kanet.py (KaNet architecture)
 - generator.py (module for generating data during training)
 - modelutils.py (module with model training helper functions)
 - model.py (model training script)
 - modelconfig.py (configuration file for persisting model building for saving/loading)
 - video.py (Used to create videos of driving data)

### Running the Code:



