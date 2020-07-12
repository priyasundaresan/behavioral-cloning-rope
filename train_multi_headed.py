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
from dataset_multi_headed import DataGenerator
from keras.layers import Input
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"]="0"

params = {"batch_size":1,
	  	  "action_dim":8, # (endLx, endLy, pull_x, pull_y, hold_x, hold_y, endRx, endRy)
          "img_dim": (640,480,3)}

dataset_name = 'undo_reid_term_train' 
train_generator = DataGenerator(dataset_name, **params)
val_dataset_name = 'undo_reid_term_val'
val_generator = DataGenerator(val_dataset_name, **params)
nb_epoch = 5 # Train for 100 epochs
INIT_LR = 1e-4

class UntangleNet:
	@staticmethod
	def build_keypoint_branch(inputs, action_dim):
		output = Dense(action_dim, activation='linear', name="keypoints_output")(inputs)
		return output
	@staticmethod
	def build_reid_cls_branch(inputs):
		output = Dense(2, activation='softmax', name="reid_cls_output")(inputs)
		return output
	@staticmethod
	def build_terminate_cls_branch(inputs):
		output = Dense(2, activation='softmax', name="termination_cls_output")(inputs)
		return output
	def build(img_shape, action_dim):
		model = VGG16(include_top=False, input_shape=img_shape)
		flat1 = Flatten()(model.outputs)
		branch_inputs = Dense(1024, activation='relu')(flat1) 
		kpt_branch = UntangleNet.build_keypoint_branch(branch_inputs, action_dim)
		reid_branch = UntangleNet.build_reid_cls_branch(branch_inputs)
		terminate_branch = UntangleNet.build_terminate_cls_branch(branch_inputs)
		final_model = Model(inputs=model.inputs, outputs=[kpt_branch, reid_branch, terminate_branch])
		return final_model

model = UntangleNet.build(params["img_dim"], params["action_dim"])
losses = {"keypoints_output": "mean_squared_error", 
		  "reid_cls_output": "categorical_crossentropy",
		  "termination_cls_output": "categorical_crossentropy"}
lossWeights = {"keypoints_output": 1.0,
		  "reid_cls_output": 1.0,
		  "termination_cls_output": 1.0}
opt = Adam(lr=INIT_LR, decay=INIT_LR / nb_epoch)

model.compile(loss=losses,
              optimizer=opt,
              metrics=['accuracy']) 

checkpoints_dir = 'checkpoints'
log_dir = os.path.join(checkpoints_dir, dataset_name)
if not os.path.exists(checkpoints_dir):
    os.mkdir(checkpoints_dir)
if not os.path.exists(log_dir):
    os.mkdir(log_dir)

#filepath = "%s/saved-model-{epoch:02d}-{val_loss:.3f}.hdf5"%(log_dir)
filepath = "%s/saved-model-{epoch:02d}-{loss:.3f}.hdf5"%(log_dir)
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1,
    save_best_only=False, mode='auto', period=1)

# TRAINING
H = model.fit_generator(generator=train_generator, validation_data=val_generator, nb_epoch=nb_epoch, callbacks=[checkpoint])

# PLOTTING
lossNames = ["loss", "keypoints_output_loss", "reid_cls_output_loss", "termination_cls_output_loss"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(4, 1, figsize=(13, 13))
# loop over the loss names
for (i, l) in enumerate(lossNames):
	# plot the loss for both the training and validation data
	title = "Loss for {}".format(l) if l != "loss" else "Total loss"
	ax[i].set_title(title)
	ax[i].set_xlabel("Epoch #")
	ax[i].set_ylabel("Loss")
	ax[i].plot(np.arange(0, nb_epoch), H.history[l], label=l)
	ax[i].plot(np.arange(0, nb_epoch), H.history["val_" + l],
		label="val_" + l)
	ax[i].legend()
# save the losses figure
plt.tight_layout()
plt.savefig("{}/losses.png".format(log_dir))
plt.close()
accuracyNames = ["keypoints_output_acc", "reid_cls_output_acc", "termination_cls_output_acc"]
plt.style.use("ggplot")
(fig, ax) = plt.subplots(3, 1, figsize=(8, 8))
# loop over the accuracy names
for (i, l) in enumerate(accuracyNames):
	# plot the loss for both the training and validation data
	ax[i].set_title("Accuracy for {}".format(l))
	ax[i].set_xlabel("Epoch #")
	ax[i].set_ylabel("Accuracy")
	ax[i].plot(np.arange(0, nb_epoch), H.history[l], label=l)
	ax[i].plot(np.arange(0, nb_epoch), H.history["val_" + l],
		label="val_" + l)
	ax[i].legend()
# save the accuracies figure
plt.tight_layout()
plt.savefig("{}/accs.png".format(log_dir))
plt.close()
