import numpy as np
import h5py
import os
import cv2

img_dim = [640,480,3]
action_dim = 6
batch_size = 1
nb_epoch = 100

def img_reshape(input_img):
    _img = np.transpose(input_img, (1, 2, 0))
    _img = np.flipud(_img)
    _img = np.reshape(_img, (1, img_dim[0], img_dim[1], img_dim[2]))
    return _img

images_all = np.zeros((0, img_dim[0], img_dim[1], img_dim[2]))
actions_all = np.zeros((0,action_dim))

dataset_name = 'chord_single_knot'
dataset_dir = 'datasets/%s'%(dataset_name)
image_dir = os.path.join(dataset_dir, 'images')
action_dir = os.path.join(dataset_dir, 'actions')
num_train = len(os.listdir(image_dir))
num_train = 500

# TODO, this is very inefficient, need to make a data loader 
print('Packing data into arrays...')
for i in range(num_train):
    if (i%50 == 0):
        print(i)
    img = cv2.imread(os.path.join(image_dir, '%05d.jpg'%i))
    act = np.load(os.path.join(action_dir, '%05d.npy'%i))
    images_all = np.concatenate([images_all, img_reshape(img)], axis=0)
    actions_all = np.concatenate([actions_all, np.reshape(act, [1,action_dim])], axis=0)

from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.models import Model

model = VGG16(include_top=False, input_shape=tuple(img_dim))
flat1 = Flatten()(model.outputs)
class1 = Dense(1024, activation='relu')(flat1)
output = Dense(action_dim, activation='linear')(class1)
model = Model(inputs=model.inputs, outputs=output)

model.compile(loss='mean_squared_error',
              optimizer=Adam(lr=1e-4),
              metrics=['mean_squared_error'])

model.fit(images_all, actions_all,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              shuffle=True)

model.save("bc_test_model")
