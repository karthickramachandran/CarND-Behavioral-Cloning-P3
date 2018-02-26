# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains starting files for the Behavioral Cloning Project.  

This repository consists of the following files,  

- model.py
- model.h5
- drive.py
- video.mp4
- report.md

## Execution

To train the network, navigate to the Behaviour cloning project folder, the copy the images and the csv file into the reordings folder.  

Execute the following command,  
> python model.py

When the training is completed, the model is saved as _model.h5_ in the same folder.  

To test the model,  

> Run the simulator, click Autonomous mode,  
> Execute python drive.py model.h5  

## Save the video

To save the video,  

- Save image sequnces
- Create video from the image sequences

Execute,  
> python drive.py model.h5 video

Image sequence gets saved in the video folder.  

> python video.py video

The images sequnce is used to create a video.mp4 file.  


## Network architecture

![netwrok_architecture] (network.png)

Nvidia convolutional neural network architecture is used to train and predict the steering angles.  

The network comprises of,  

- 5 -Convolution layers
- 3 Fully connected layers

> RELU is used as activation layer with max pooling and drop out is used to avoid overfitting. Adam optimizer is used to tune the parameters.  

## Report in detail

- Read the CSV file
- Split data for training and validation
- Create a generator
- Create the Network model
- Feed the data for training and validation
- Save the model

While collecting the data, the images are saved in IMG folder along with a CSV file. Read all the line in the CSV file. The first column is the location/name of the images from the center camera, followed by left and right cameras and then the steering angle.  

sklearn 'train_test_split' is used to split the dataset into training and validation data. The data is fed to a generator, which allows us to augument the data in real-time parallell to the model.  

Images have been flipped and noises are introducted to increase the stability of the prediction as well as to extrapolate the available dataset.  

A batch size of 32 is used for training and network is trained for 10 epochs with a learning rate of 0.001.  

## Data collection

Training a monotonous data resulted in failure of keeping the vehicle on the road, thus the dataset is created in a way that the vehicle always tres to go-away(towards center) from the road edges. This has helped to teach the Neural network to drive away from the road edges towards center.   

Few images from the dataset are displayed below,  


![dataset_images_1](/home/rmc/UDACITY/CarND-Behavioral-Cloning-P3/report_images/center_2018_02_23_15_59_57_380.jpg)  

![dataset_images_2](/home/rmc/UDACITY/CarND-Behavioral-Cloning-P3/report_images/center_2018_02_23_16_01_22_622.jpg)  

![dataset_images_3](/home/rmc/UDACITY/CarND-Behavioral-Cloning-P3/report_images/center_2018_02_23_16_01_57_449.jpg)  

![dataset_images_4](/home/rmc/UDACITY/CarND-Behavioral-Cloning-P3/report_images/center_2018_02_23_16_02_14_432.jpg)  

![dataset_images_5](/home/rmc/UDACITY/CarND-Behavioral-Cloning-P3/report_images/center_2018_02_23_16_02_21_196.jpg)



