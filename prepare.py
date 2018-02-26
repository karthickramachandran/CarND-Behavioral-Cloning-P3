# Keras includes
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Cropping2D

#Other  includes
import random
import os
# Numpy includes
import numpy as np
import pandas as pd
from scipy import misc
#OPenCV includes
#import cv2

import model

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
def preprocess():
    batch_size = 32
    epochs = 100
    model_name = 'model.h5'
    left_correction = 0.2
    right_correction = -0.2


    print('Training Batch size: ', batch_size, '\nNo of epochs: ', epochs)
    print('Model is saved as: ', model_name)

    source_dir = os.path.join(os.getcwd(), 'recordings/IMG/')
    data_dir = os.path.join(os.getcwd(), 'recordings/')
    print('recorded data is available in: ', source_dir)
    preprocessed_data_dir = os.path.join(os.getcwd(), 'processed_data/IMG/')

    pics_location = os.listdir(source_dir)
    print('Preprocessed data is stored in: ', preprocessed_data_dir)

    new_pics=pics_location[0:20000]

    columns = ['center', 'left', 'right', 'steering_angle', 'throttle', 'brake', 'speed']
    df = pd.read_csv(data_dir+"driving_log.csv", names=columns)
    print(df.shape)

    xs = df['center']
    ys= df['steering_angle']
    xs_r = df['right']
    ys_r= df['steering_angle']+right_correction

    xs_l = df['left']
    ys_l= df['steering_angle']+left_correction

    xs = pd.concat([xs, xs_l], axis=0)
    xs = pd.concat([xs, xs_r], axis=0)
    ys = pd.concat([ys, ys_l], axis=0)
    ys = pd.concat([ys, ys_r], axis=0)
    num_images, cols = df.shape

    print('len of xs', len(xs))
    print('len of ys', len(ys))
    #shuffle list of images
    c = list(zip(xs, ys))
    random.shuffle(c)
    xs, ys = zip(*c)

    train_xs = xs[:int(len(xs) * 0.6)]
    train_ys = ys[:int(len(xs) * 0.6)]

    val_xs = xs[-int(len(xs) * 0.4):]
    val_ys = ys[-int(len(xs) * 0.4):]

    num_train_images = len(train_xs)
    num_val_images = len(val_xs)

    print('No of training images: ', num_train_images)
    print('No of validation images: ', num_val_images)

    # Images are read in the order center, left and right

    k=0
    images = []
    steering_angles = []

    for i in new_pics:
        img = misc.imread(source_dir+i)
        image_flipped = np.fliplr(img)
        image_normalized = image_flipped/255.0
        salt_pepper_img = sp_noise(img, 0.05)

        images.append(img)
        images.append(image_flipped)
        images.append(salt_pepper_img)
        steering_angles.append(-ys[k])# FLIPPED IMAGE
        steering_angles.append(ys[k]) #Salt and pepper image

        k=k+1

    model = model.build_predict_steering_model()
    model.compile(loss="mse", optimizer=Adam)

    ckpt = modelCheckpoint('model.h5')


    # for i in new_pics:
    #      #print(i)
    #      image = misc.imread(source_dir+i)
    #      image_flipped = np.fliplr(image)
    #      image_normalized = image_flipped/255.0
    #      salt_pepper_img = sp_noise(image, 0.05)
    #
    #      misc.imsave(preprocessed_data_dir+'img_'+i+'_Orig'+'.jpg', image)
    #      img_loc=pd.DataFrame([preprocessed_data_dir+'img_'+i+'_Orig'+'.jpg'])
    #
    #      print(len(xs))
    #      print(ys[k])
    #      ys=pd.concat([ys, ys[k]], axis=0)
    #
    #      misc.imsave(preprocessed_data_dir+'img_'+i+'_F'+'.jpg', image_flipped)
    #      img_loc=preprocessed_data_dir+'img_'+i+'_F'+'.jpg'
    #      ar= np.array(img_loc)
    #      xs= pd.concat([xs, img_loc], axis=0)
    #      ys.append(ys[i])
    #
    #      misc.imsave(preprocessed_data_dir+'img_'+i+'_N'+'.jpg', image_normalized)
    #      img_loc=preprocessed_data_dir+'img_'+i+'_N'+'.jpg'
    #      ar= np.array(img_loc)
    #      xs= pd.concat([xs, img_loc], axis=0)
    #      ys.append(ys[i])
    #
    #      misc.imsave(preprocessed_data_dir+'img_'+i+'_SP'+'.jpg', salt_pepper_img)
    #      img_loc=preprocessed_data_dir+'img_'+i+'_SP'+'.jpg'
    #      ar= np.array(img_loc)
    #      xs= pd.concat([xs, img_loc], axis=0)
    #      ys.append(ys[i])

    print('preprocessing is sucessfully done')

def main():
    preprocess()
