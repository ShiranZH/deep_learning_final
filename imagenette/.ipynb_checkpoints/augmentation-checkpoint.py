#!/usr/bin/env python
#-*- coding: utf-8 -*-

# import packages
# data augmentation for subset of imagenet
import os
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img

datagen = ImageDataGenerator(
    rotation_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

dir = '../data/imagenette2/train'

for d in os.listdir(dir):
    newdir = os.path.join(dir, d)
    for f in os.listdir(newdir):
        newfile = os.path.join(newdir, f)

        img = load_img(newfile)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        i = 0
        for batch in datagen.flow(x, batch_size=1, save_to_dir=newdir, save_prefix=f, save_format='JPEG'):
            i += 1
            if i > 20:
                break  # otherwise the generator would loop indefinitely
