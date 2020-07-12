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
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint
from dataset import DataGenerator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="0"

# NOTE: batch_size > 1 does not work, errors at line:
# class1 = Dense(1024, activation='relu')(flat1)
# Because of this, we cannot use huge datasets yet, I recommend < 1000 image/action pairs
#if you find a fix lmk, think we need to add input_dims argument or something...
params = {"batch_size":1,
	  "action_dim":6, # (pull_x, pull_y, drop_x, drop_y, hold_x, hold_y)
          "img_dim": (640,480,3)}
# It will look for images at /datasets/chord_single_knot/images and actions at  /datasets/chord_single_knot/actions
dataset_name = 'undo_reid_train' 
val_dataset_name = 'undo_reid_val'
train_generator = DataGenerator(dataset_name, **params)
val_generator = DataGenerator(val_dataset_name, **params)
nb_epoch = 100 # Train for 100 epochs

model = VGG16(include_top=False, input_shape=params["img_dim"])
flat1 = Flatten()(model.outputs)
class1 = Dense(1024, activation='relu')(flat1) # This is the line that will error for batch_size > 1
class1 = Dropout(0.3)(class1)
output = Dense(params["action_dim"], activation='linear')(class1)
model = Model(inputs=model.inputs, outputs=output)

model.compile(loss='mean_squared_error',
              optimizer=Adam(lr=1e-4),
              metrics=['mean_squared_error']) # Using L2 loss for regression

checkpoints_dir = 'checkpoints'
log_dir = os.path.join(checkpoints_dir, dataset_name)
if not os.path.exists(checkpoints_dir):
    os.mkdir(checkpoints_dir)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

filepath = "%s/saved-model-{epoch:02d}-{val_loss:.3f}.hdf5"%(log_dir)
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
    save_best_only=False, mode='auto', period=1)

history = model.fit_generator(generator=train_generator, validation_data=val_generator, nb_epoch=nb_epoch, callbacks=[checkpoint])
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('%s/loss.png'%(log_dir))
#model.save("bc_model_undo_reid") # Note, only saving at the end of training currently
