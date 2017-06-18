import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.io import imread, imshow
import cv2

#matplotlib inline
import plotly.offline as py
#py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from __future__ import division, print_function, absolute_import
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation




def transform_image(img, rescaled_dim, to_gray=False):
    resized = cv2.resize(img, (rescaled_dim, rescaled_dim), cv2.INTER_LINEAR)
    if to_gray:
        resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY).astype('float')
    else:
        resized = resized.astype('float')
    normalized = cv2.normalize(resized, None, 0.0, 1.0, cv2.NORM_MINMAX)
    return normalized

rescaled_dim = 32


X=np.load('/home/u3749/code/new/all_img.npy')
Y=np.load('/home/u3749/code/new/all_type.npy')
X,Y=shuffle(X,Y)

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)

# Convolutional network building
network = input_data(shape=[None, 32, 32, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
#network = input_data(shape=[None, 32, 32, 3])
network = conv_2d(network, 16, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 16, 3, activation='relu')
network = max_pool_2d(network, 2)

network = fully_connected(network, 256, activation='relu')
network = dropout(network, 0.25)

network = fully_connected(network, 3, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

model = tflearn.DNN(network, tensorboard_verbose=0)

model.fit(X, Y, n_epoch=300, shuffle=True, validation_set=0.25,show_metric=True, batch_size=96, run_id='cvix_cnn')

currently trying 1 layer converlution
L2 is not good 


try x2:
network = conv_2d(network, 1, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 1, 3, activation='relu')



























best:


 network = input_data(shape=[None, 32, 32, 3],
...                      data_preprocessing=img_prep,
...                      data_augmentation=img_aug)
>>> #network = input_data(shape=[None, 32, 32, 3])
... network = conv_2d(network, 16, 3, activation='relu')
>>> network = max_pool_2d(network, 2)
>>> network = conv_2d(network, 16, 3, activation='relu')
>>> network = max_pool_2d(network, 2)
>>> network = fully_connected(network, 256, activation='relu')
network = dropout(network, 0.5)
network = fully_connected(network, 3, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

model = tflearn.DNN(network, tensorboard_verbose=0)

model.fit(X, Y, n_epoch=200, shuffle=True, validation_set=0.25,show_metric=True, batch_size=96, run_id='cvi)

Training Step: 13000  | total loss: 0.24950 | time: 20.714s
| Adam | epoch: 200 | loss: 0.24950 - acc: 0.9006 | val_loss: 1.24218 - val_acc: 0.6795 -- iter: 6156/6156

network = max_pool_2d(network, 2)
network = conv_2d(network, 1, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 1, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 1, 3, activation='relu')
network = max_pool_2d(network, 2)

network = fully_connected(network, 128, activation='relu',regularizer='L1'



best2: 
 network = conv_2d(network, 16, 3, activation='relu')
>>> network = max_pool_2d(network, 2)
>>> network = conv_2d(network, 16, 3, activation='relu')
>>> network = max_pool_2d(network, 2)
>>> network = fully_connected(network, 512, activation='relu')
>>> network = dropout(network, 0.25)

| Adam | epoch: 200 | loss: 0.35586 - acc: 0.8555 | val_loss: 1.04542 - val_acc: 0.6727 -- iter: 6156/6156


best3:
>>> network = fully_connected(network, 256, activation='relu')
>>> network = dropout(network, 0.25)
| Adam | epoch: 200 | loss: 0.46258 - acc: 0.8059 | val_loss: 0.95803 - val_acc: 0.6634 -- iter: 6156/6156


