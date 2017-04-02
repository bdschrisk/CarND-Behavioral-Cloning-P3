# Behavioural Cloning Project
### "Project Karl"
*__"My first customer was a lunatic. My second had a death wish."__*

## Overview
This project is intended to apply behavioural cloning using deep learning to drive a car around a track within a simulated environment.

*If you're looking for the reinforcement version of this project see the CarND-Behavioral-Cloning-P3-RL repo.*

#### Model Architecture
*To learn the steering approximation function I've designed a deep neural network which is loosely based on the LeNet architecture, which I named "KaNet".*

In order to learn decent conceptual features of proper steering behaviour, an adequately deep neural network is required. However, if the network is too shallow, it cannot learn higher order latent features. This model does that, and is also not too deep as to cause common gradient issues found in deeper neural networks nor does it overfit the data.

The raw input image data is fed into a series of Input processing layers. Sequentially, the input is first cropped to exclude the skyscape and then normalised, following this the input is resized to 50% across the width and height dimensions.

After the preprocessing layers, repeating convolutional layers with 64, 32, 16 and 8 filters of 5x5, 3x3 and 2x2 detectors, consecutively, extract meaningful features from the input. An average pooling layer downsamples each convolution into rectifiers for non-linearity, condensing the input into a final average pooling layer. The final pooling layer is responsible for extracting spatial respresentations from each convolution into a compressed output. A global hidden layer of size 256 with parametric rectifiers provide latent features for extracting angles and throttle vectors. Each output node is connected to its own hidden layer of size 128 with parametric rectifiers for identifying useful representations of its output.

Dropout layers were applied after each hidden activation layer, with a drop out rate of 20%. I've also used L2 loss normalisation for weights and biases for lower layers. Dropout was not applied to the convolutional layers as overfitting is generally caused by the dense upper layers in deep networks.

##### KaNet Model
____________________________________________________________________________________________________
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
convolution2d_4 (Convolution2D)  (None, 13, 71, 8)     520         activation_3[0][0]
____________________________________________________________________________________________________
averagepooling2d_4 (AveragePooli (None, 12, 70, 8)     0           convolution2d_4[0][0]
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 12, 70, 8)     0           averagepooling2d_4[0][0]
____________________________________________________________________________________________________
averagepooling2d_5 (AveragePooli (None, 6, 35, 8)      0           activation_4[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1680)          0           averagepooling2d_5[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 256)           430336      flatten_1[0][0]
____________________________________________________________________________________________________
prelu_1 (PReLU)                  (None, 256)           256         dense_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 256)           0           prelu_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 128)           32896       dropout_1[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 128)           32896       dropout_1[0][0]
____________________________________________________________________________________________________
prelu_2 (PReLU)                  (None, 128)           128         dense_2[0][0]
____________________________________________________________________________________________________
prelu_3 (PReLU)                  (None, 128)           128         dense_4[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 128)           0           prelu_2[0][0]
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 128)           0           prelu_3[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 1)             129         dropout_2[0][0]
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             129         dropout_3[0][0]
____________________________________________________________________________________________________
merge_1 (Merge)                  (None, 2)             0           dense_3[0][0]
                                                                   dense_5[0][0]
____________________________________________________________________________________________________
Total params: 522,810
Trainable params: 522,810
Non-trainable params: 0

### Training
Training the network was performed using the Adam optimiser with MSE loss, along with an initial learning rate of 0.001. As part of training the network, a custom callback was used which prevents overfitting by monitoring the divergence between the training and validation loss metrics.

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

Usage of `drive.py` requires you have save the trained model as an h5 weights file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save_weights(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument `run1` is the directory to save the images seen by the agent to. If the directory already exists it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp when the image image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Create a video based on images found in the `run1` directory. The name of the video will be name of the directory following by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

The video will run at 48 FPS. The default FPS is 60.
