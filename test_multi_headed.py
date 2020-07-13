from keras.models import Sequential
import tensorflow as tf
import colorsys
import cv2
import numpy as np
import os
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.models import Model, load_model

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def img_reshape(input_img):
    _img = np.transpose(input_img, (1, 2, 0))
    _img = np.flipud(_img)
    _img = np.reshape(_img, (1, img_dim[0], img_dim[1], img_dim[2]))
    return _img

def plot_predictions(img, pred, gt, plot_gt=False):
    vis = img.copy()
    #(keypoints, reid, terminate) = pred
    (keypoints, terminate) = pred
    keypoints = keypoints.reshape((4,2)).astype(int)
    #reid = reid[0].argmax()
    terminate = terminate[0].argmax()
#    if reid:
#        cv2.putText(vis, "Take Reid", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
#	0.7, (255, 255, 255), 2)
    if terminate:
        cv2.putText(vis, "Undone", (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (255, 255, 255), 2)
    for i, (u, v) in enumerate(keypoints):
        (r, g, b) = colorsys.hsv_to_rgb(float(i)/len(keypoints), 1.0, 1.0)
        R, G, B = int(255 * r), int(255 * g), int(255 * b)
        cv2.circle(vis,(int(u), int(v)), 4, (R, G, B), -1)
    return vis

if __name__ == '__main__':
    dataset_name = 'undo_reid_term_val'
    dataset_dir = 'datasets/%s'%(dataset_name)
    image_dir = os.path.join(dataset_dir, 'images')
    action_dir = os.path.join(dataset_dir, 'actions')
    num_test = len(os.listdir(image_dir))

    img_dim = [640,480,3]
    action_dim = 6

    model = load_model("checkpoints/undo_reid_term_train/saved-model-20-380.131.hdf5", {'tf': tf})

    output_dir = 'preds'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    for i in range(num_test):
        print(i)
        img = cv2.imread(os.path.join(image_dir, '%05d.jpg'%i))
        input_img = img_reshape(img)
        pred = model.predict(input_img)
        gt = np.load(os.path.join(action_dir, '%05d.npy'%i)).squeeze()
        vis = plot_predictions(img, pred, gt)
        cv2.imwrite(os.path.join(output_dir, '%05d.jpg'%i), vis)
