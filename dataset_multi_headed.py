import os
import cv2
import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    def __init__(self, dataset_name, batch_size=1, action_dim=6, img_dim=(640,480,3), path_to_datasets='datasets'):
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.img_dim = img_dim
        self.image_dir = os.path.join(path_to_datasets, dataset_name, 'images')
        self.action_dir = os.path.join(path_to_datasets, dataset_name, 'actions')
        self.num_examples = len(os.listdir(self.image_dir))
        self.on_epoch_end()

    def __len__(self):
      return int(np.floor(self.num_examples / self.batch_size))

    def __getitem__(self, index):
      'Get next batch'
      indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
      imgs, actions = self.__data_generation(indexes)
      return imgs, actions

    def on_epoch_end(self):
      'Updates indexes after each epoch'
      self.indexes = np.arange(self.num_examples)
      np.random.shuffle(self.indexes)

    def img_reshape(self, input_img):
        _img = np.transpose(input_img, (1, 2, 0))
        _img = np.flipud(_img)
        _img = np.reshape(_img, (1, self.img_dim[0], self.img_dim[1], self.img_dim[2]))
        return _img

    def __data_generation(self, indexes):
      'Generates data containing batch_size samples'
      imgs = np.empty((self.batch_size, self.img_dim[0], self.img_dim[1], self.img_dim[2]))
      actions = np.empty((self.batch_size, self.action_dim))
      undone_indicator = np.empty((self.batch_size, 1))
      terminate_indicator = np.empty((self.batch_size, 1))
      for i, idx in enumerate(indexes):
          img = cv2.imread(os.path.join(self.image_dir, '%05d.jpg'%idx))
          img = self.img_reshape(img)
          data = np.load(os.path.join(self.action_dir, '%05d.npy'%idx))
          action = data[:self.action_dim]
          undone = data[-2]
          terminate = data[-1]
          action = np.reshape(action, [1,self.action_dim])
          imgs[i,:] = img
          actions[i,:] = action
          undone_indicator[i,:] = undone
          terminate_indicator[i,:] = terminate
      labels = {"keypoints_output": actions, \
		"undone_cls_output": keras.utils.to_categorical(undone_indicator, num_classes=2), \
	"termination_cls_output": keras.utils.to_categorical(terminate_indicator, num_classes=2)}
      return imgs, labels
