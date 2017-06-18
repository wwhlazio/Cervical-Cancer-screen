import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.io import imread, imshow
import cv2

import plotly.offline as py
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



from subprocess import check_output
print(check_output(["ls", "/data/kaggle_3.27/additional"]).decode("utf8"))

from glob import glob
basepath = '/data/kaggle_3.27/additional/'

all_cervix_images = []

for path in sorted(glob(basepath + "*")):
    cervix_type = path.split("/")[-1]
    cervix_images = sorted(glob(basepath + cervix_type + "/*"))
    all_cervix_images = all_cervix_images + cervix_images

all_cervix_images = pd.DataFrame({'imagepath': all_cervix_images})
all_cervix_images['filetype'] = all_cervix_images.apply(lambda row: row.imagepath.split(".")[-1], axis=1)
all_cervix_images['type'] = all_cervix_images.apply(lambda row: row.imagepath.split("/")[-2], axis=1)
print (all_cervix_images.head())


from collections import defaultdict

def transform_image(img, rescaled_dim, to_gray=False):
    resized = cv2.resize(img, (rescaled_dim, rescaled_dim), cv2.INTER_LINEAR)
    if to_gray:
        resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY).astype('float')
    else:
        resized = resized.astype('float')
    normalized = cv2.normalize(resized, None, 0.0, 1.0, cv2.NORM_MINMAX)
    return normalized


rescaled_dim = 32


for t in all_cervix_images['type'].unique():
    images = defaultdict(list)    	
    sample_counter = 0
    for _, row in all_cervix_images[all_cervix_images['type'] == t].iterrows():
        try:
            img = imread(row.imagepath)
            sample_counter +=1
            images[t].append(img)
        except:
            print('image read failed for {}'.format(row.imagepath))
    dfs = []
    t_ = pd.DataFrame(
        {
            'nrows': list(map(lambda i: i.shape[0], images[t])),
            'ncols': list(map(lambda i: i.shape[1], images[t])),
            'nchans': list(map(lambda i: i.shape[2], images[t])),
            'type': t
        }
    )
    dfs.append(t_)
    shapes_df = pd.concat(dfs, axis=0)
    shapes_df_grouped = shapes_df.groupby(by=['nchans', 'ncols', 'nrows', 'type']).size().reset_index().sort_values(['type', 0], ascending=False)
    print(shapes_df_grouped)
    shapes_df_grouped['size_with_type'] = shapes_df_grouped.apply(lambda row: '{}-{}-{}'.format(row.ncols, row.nrows, row.type), axis=1)
    shapes_df_grouped = shapes_df_grouped.set_index(shapes_df_grouped['size_with_type'].values)
    shapes_df_grouped['count'] = shapes_df_grouped[[0]]
    all_images = []
    all_image_types = []
    all_images = all_images + images[t]
    all_image_types = all_image_types + len(images[t])*[t]
    gray_all_images_as_vecs = [transform_image(img, rescaled_dim) for img in all_images]
    gray_imgs_mat = np.array(gray_all_images_as_vecs).squeeze()
    all_image_types = np.array(all_image_types)
    gray_imgs_mat.shape, all_image_types.shape
    for i in range(len(all_image_types)):
        if all_image_types[i] == 'Type_1_v2':
                all_image_types[i] = 0
        if all_image_types[i] == 'Type_2_v2':
                all_image_types[i] = 1
        if all_image_types[i] == 'Type_3_v2':
                all_image_types[i] = 2
    #save saparately

X=np.load('/home/u3749/code/64_rescale_X.npy')
Y=np.load('/home/u3749/code/64_rescale_Y.npy')
X_test=np.load('/home/u3749/code/64_rescale_X_test.npy')
Y_test=np.load('/home/u3749/code/64_rescale_Y_test.npy')

X1=np.load('/home/u3749/code/64_rescale_X_type_1_add.npy')
X2=np.load('/home/u3749/code/64_rescale_X_type_2_add.npy')
X3=np.load('/home/u3749/code/64_rescale_X_type_3_add.npy')

Y1=np.load('/home/u3749/code/64_rescale_Y_type_1_add.npy')
Y2=np.load('/home/u3749/code/64_rescale_Y_type_2_add.npy')
Y3=np.load('/home/u3749/code/64_rescale_Y_type_3_add.npy')

XA=np.concatenate((X,X1),axis=0)
XA=np.concatenate((XA,X2),axis=0)
XA=np.concatenate((XA,X3),axis=0)
XA=np.concatenate((XA,X_test),axis=0)

YA=np.concatenate((Y,to_categorical(Y1, 3)),axis=0)
YA=np.concatenate((YA,to_categorical(Y2, 3)),axis=0)
YA=np.concatenate((YA,to_categorical(Y3, 3)),axis=0)
YA=np.concatenate((YA,Y_test),axis=0)

np.save('/home/u3749/code/64_rescale_all_X.npy',XA)
np.save('/home/u3749/code/64_rescale_all_Y.npy',YA)

csv = np.genfromtxt('/home/u3749/code/solution_stg1_release.csv',delimiter=",")
testlab=csv[0:513,1:4]
testlabx=testlab[1:513,]

np.save('/home/u3749/code/test_sub_lab.npy',testlabx)
