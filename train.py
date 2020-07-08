import numpy as np
import h5py
import os
import cv2
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.models import Model
from dataset import DataGenerator

# NOTE: batch_size > 1 does not work, errors at line:
# class1 = Dense(1024, activation='relu')(flat1)
# Because of this, we cannot use huge datasets yet, I recommend < 1000 image/action pairs
#if you find a fix lmk, think we need to add input_dims argument or something...
params = {"batch_size":1,
	  "action_dim":6, # (pull_x, pull_y, drop_x, drop_y, hold_x, hold_y)
          "img_dim": (640,480,3)}
# It will look for images at /datasets/chord_single_knot/images and actions at  /datasets/chord_single_knot/actions
dataset_name = 'chord_single_knot' 
train_generator = DataGenerator(dataset_name, **params)
nb_epoch = 100 # Train for 100 epochs

model = VGG16(include_top=False, input_shape=params["img_dim"])
flat1 = Flatten()(model.outputs)
class1 = Dense(1024, activation='relu')(flat1) # This is the line that will error for batch_size > 1
output = Dense(params["action_dim"], activation='linear')(class1)
model = Model(inputs=model.inputs, outputs=output)

model.compile(loss='mean_squared_error',
              optimizer=Adam(lr=1e-4),
              metrics=['mean_squared_error']) # Using L2 loss for regression

model.fit_generator(generator=train_generator, nb_epoch=nb_epoch)
model.save("bc_model") # Note, only saving at the end of training currently
