# Keras includes
import tensorflow as tf
import keras

config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)

#from keras.utils import multi_gpu_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Convolution2D, Lambda
from keras.layers.convolutional import Cropping2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback

import os
# Numpy includes
import numpy as np

import csv
import cv2
import matplotlib.pyplot as plt
from keras.backend import tf

#Other  includes
import random
from scipy import misc

from sklearn.model_selection import train_test_split


xs=[]
ys=[]
def sp_noise(image,prob):
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def resize(image):
    return  tf.image.resize_images(image, [66, 200])
def build_predict_steering_model():

    model = Sequential()
    model.add(Cropping2D(cropping=((60, 20), (0, 0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda pix: (pix / 255.0) - 0.5))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(24, 5, activation="relu"))
    model.add(Conv2D(36, 5, activation="relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(48, 5, activation="relu"))
    model.add(Conv2D(64, 3, activation="relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, 3, activation="relu"))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(100, activation="relu"))
    model.add(Dropout(0.15))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(0.15))
    model.add(Dense(10, activation="relu"))
    model.add(Dropout(0.15))
    model.add(Dense(1))

    model.summary()

    return model



    #print("--")
def read_file():
    data_dir = os.path.join(os.getcwd(), 'recordings/')
    csv_file = data_dir+"driving_log.csv"
    driving_logs= [csv_file,]
    lines = []

    for driving_log in driving_logs:
        print(driving_log)
        for line in open(driving_log):
            row = line.split()
            #print(row)
            lines.append(row)

    return lines



def train_generate(train_samples, batch_size):
    while 1:
        for offset in range(0,len(train_samples), batch_size):
            samples = train_samples[offset:offset+batch_size]
            images = []
            steering_angles = []

            for sample in samples:
                # read center image
                image = sample[0].split(',')[0]
                source_dir = os.path.join(os.getcwd(), 'recordings/')
                #image_name_location = source_dir+center_image
                center_image = misc.imread(image)

                # Read steering_angle
                angle = float (sample[0].split(',')[3])

                # add
                images.append(center_image)
                steering_angles.append(angle)

            Xs = images
            ys = steering_angles

            for i in range(int(batch_size/batch_size)):#batch_size
                image_flipped = np.fliplr(images[i])
                #print(image_flipped.shape)
                image_normalized = image_flipped/255.0
                salt_pepper_img = sp_noise(images[i], 0.05)
                Xs.append(image_flipped)
                Xs.append(salt_pepper_img)
                ys.append(-steering_angles[i])
                ys.append(steering_angles[i])

            image_train = np.array(Xs)
            angle_train = np.array(ys)
            yield (image_train, angle_train)#random.shuffle


def validation_generate(validation_samples, batch_size):
    while 1:
        for offset in range(0,len(validation_samples), batch_size):
            samples = validation_samples[offset:offset+batch_size]
            images = []
            steering_angles = []
            for sample in samples:
                image = sample[0].split(',')[0]
                source_dir = os.path.join(os.getcwd(), 'recordings/')
                #image_name_location = source_dir+center_image
                center_image = misc.imread(image)

                # Read steering_angle
                angle = (sample[0].split(',')[3])
                angle=angle
                    # add
                images.append(center_image)
                steering_angles.append(angle)

            image_validation = np.array(images)
            angle_validation = np.array(steering_angles)
            yield (image_validation, angle_validation) #random.shuffle

def main():
    batch_size = 32
    epochs = 10
    model_name = 'model.h5'

    print('Training Batch size: ', batch_size, '\nNo of epochs: ', epochs)
    print('Model is saved as: ', model_name)

    print('MAIN FUNCTION')
    print('\nPREPARING KERAS MODELS .... ')

    lines = read_file()

    #print('lines ', lines)
    train_samples, validation_samples = train_test_split(lines, test_size=0.2) #train_test_split(lines, test_size=0.3)

    train_gen = train_generate(train_samples, batch_size)
    val_gen = validation_generate(validation_samples, batch_size)

    model = build_predict_steering_model()
    model.compile(loss="mse", optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8))

    checkpoint = ModelCheckpoint('model/model{epoch:02d}.h5')

    history_object = model.fit_generator(train_gen,
        steps_per_epoch= len(train_samples)/batch_size,
        validation_data=val_gen,
        validation_steps=len(validation_samples)/batch_size,
        epochs=epochs,
        verbose = 1,
callbacks=[checkpoint])

    model.save(model_name)

if __name__ == "__main__":
    main()
