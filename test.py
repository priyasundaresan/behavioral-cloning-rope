from keras.models import Sequential
import colorsys
import cv2
import numpy as np
import os
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.models import Model, load_model

model = load_model("bc_test_model")

img_dim = [640,480,3]
action_dim = 6

def img_reshape(input_img):
    _img = np.transpose(input_img, (1, 2, 0))
    _img = np.flipud(_img)
    _img = np.reshape(_img, (1, img_dim[0], img_dim[1], img_dim[2]))
    return _img

def plot_pred_actions(img, act_pred, act, plot_gt=False):
    pull_loc, drop_loc, hold_loc = act_pred.astype(int)
    cv2.circle(img, tuple(hold_loc), 3, (255,0,0), -1)
    cv2.arrowedLine(img, tuple(pull_loc), tuple(drop_loc), (0,255,0), 2)
    return img
    #for i, (u,v) in enumerate(act_pred):
    #    (r,g,b) = colorsys.hsv_to_rgb(float(i)/len(act_pred), 1.0, 1.0)
    #    R,G,B = int(255*r), int(255*g), int(255*b)
    #    cv2.circle(img,(int(u),int(v)),3,(R,G,B),-1)
    if plot_gt:
        for i, (u,v) in enumerate(act):
            (r,g,b) = colorsys.hsv_to_rgb(float(i)/len(act_pred), 1.0, 1.0)
            R,G,B = int(255*r), int(255*g), int(255*b)
            cv2.circle(img,(int(u),int(v)),3,(R,G,B),-1)
            cv2.circle(img,(int(u),int(v)),2,(255,255,255),-1)
    return img

if __name__ == '__main__':
    dataset_name = 'chord_single_knot'
    dataset_dir = 'datasets/%s'%(dataset_name)
    image_dir = os.path.join(dataset_dir, 'images')
    action_dir = os.path.join(dataset_dir, 'actions')
    num_train = len(os.listdir(image_dir))
    num_train = 500

    output_dir = 'preds'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    print('Packing data into arrays...')
    #for img, act in zip(img_list, action_list):
    for i in range(num_train):
        print(i)
        img = cv2.imread(os.path.join(image_dir, '%05d.jpg'%i))
        input_img = img_reshape(img)
        act_pred = model.predict(input_img).reshape((3,2))
        act = np.load(os.path.join(action_dir, '%05d.npy'%i)).squeeze()
        vis = plot_pred_actions(img, act_pred, act)
        cv2.imwrite(os.path.join(output_dir, '%05d.jpg'%i), vis)
