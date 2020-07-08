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

params = {"batch_size":1,
	  "action_dim":6,
          "img_dim": (640,480,3)}
dataset_name = 'chord_single_knot'
train_generator = DataGenerator(dataset_name, **params)
nb_epoch = 1

model = VGG16(include_top=False, input_shape=params["img_dim"])
flat1 = Flatten()(model.outputs)
class1 = Dense(1024, activation='relu')(flat1)
output = Dense(params["action_dim"], activation='linear')(class1)
model = Model(inputs=model.inputs, outputs=output)

model.compile(loss='mean_squared_error',
              optimizer=Adam(lr=1e-4),
              metrics=['mean_squared_error'])

model.fit_generator(generator=train_generator, nb_epoch=nb_epoch)
model.save("bc_model")
