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


#types of figure
from subprocess import check_output
print(check_output(["ls", "/data/kaggle/train"]).decode("utf8"))


#list of file and type
from glob import glob
basepath = '/data/kaggle/train/'

all_cervix_images = []

for path in sorted(glob(basepath + "*")):
    cervix_type = path.split("/")[-1]
    cervix_images = sorted(glob(basepath + cervix_type + "/*"))
    all_cervix_images = all_cervix_images + cervix_images

all_cervix_images = pd.DataFrame({'imagepath': all_cervix_images})
all_cervix_images['filetype'] = all_cervix_images.apply(lambda row: row.imagepath.split(".")[-1], axis=1)
all_cervix_images['type'] = all_cervix_images.apply(lambda row: row.imagepath.split("/")[-2], axis=1)
print (all_cervix_images.head())
#print (all_cervix_images.imagepath[1:5])


#print('We have a total of {} images in the whole dataset'.format(all_cervix_images.shape[0]))
#type_aggregation = all_cervix_images.groupby(['type', 'filetype']).agg('count')
#type_aggregation_p = type_aggregation.apply(lambda row: 1.0*row['imagepath']/all_cervix_images.shape[0], axis=1)

#fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

#type_aggregation.plot.barh(ax=axes[0])
#axes[0].set_xlabel("image count")
#type_aggregation_p.plot.barh(ax=axes[1])
#axes[1].set_xlabel("training size fraction")
#fig.savefig('/home/u3749/result/myfig.png')
#plt.close(fig)

#fig = plt.figure(figsize=(12,8))

#i = 1
#for t in all_cervix_images['type'].unique():
#    ax = fig.add_subplot(1,3,i)
#    i+=1
#    f = all_cervix_images[all_cervix_images['type'] == t]['imagepath'].values[0]
#    plt.imshow(plt.imread(f))
#    plt.title('sample for cervix {}'.format(t))
#fig.savefig('/home/u3749/result/myfig1.png')
#plt.close(fig)

from collections import defaultdict

images = defaultdict(list)

for t in all_cervix_images['type'].unique():
    sample_counter = 0
    for _, row in all_cervix_images[all_cervix_images['type'] == t].iterrows():
       #print('reading image {}'.format(row.imagepath))
        try:
            img = imread(row.imagepath)
            sample_counter +=1
            images[t].append(img)
        except:
            print('image read failed for {}'.format(row.imagepath))
#        if sample_counter > 35:    #load part of images
#            break	

dfs = []
for t in all_cervix_images['type'].unique():
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

#plt.figure(figsize=(10,8))
#shapes_df_grouped['count'].plot.barh(figsize=(10,8))
#fig=sns.barplot(x="count", y="size_with_type", data=shapes_df_grouped)
#figx=fig.get_figure()
#figx.savefig('/home/u3749/result/subtype_count.png')


def transform_image(img, rescaled_dim, to_gray=False):
    resized = cv2.resize(img, (rescaled_dim, rescaled_dim), cv2.INTER_LINEAR)
    if to_gray:
        resized = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY).astype('float')
    else:
        resized = resized.astype('float')
    normalized = cv2.normalize(resized, None, 0.0, 1.0, cv2.NORM_MINMAX)
    return normalized
#    timg = normalized.reshape(1, np.prod(normalized.shape))
#    return timg/np.linalg.norm(timg)

rescaled_dim = 64

all_images = []
all_image_types = []

for t in all_cervix_images['type'].unique():
    all_images = all_images + images[t]
    all_image_types = all_image_types + len(images[t])*[t]

# - normalize each uint8 image to the value interval [0, 1] as float image
# - rgb to gray
# - downsample image to rescaled_dim X rescaled_dim
# - L2 norm of each sample = 1
gray_all_images_as_vecs = [transform_image(img, rescaled_dim) for img in all_images]

gray_imgs_mat = np.array(gray_all_images_as_vecs).squeeze()
all_image_types = np.array(all_image_types)
gray_imgs_mat.shape, all_image_types.shape

for i in range(len(all_image_types)):
	if all_image_types[i] == 'Type_1':
		all_image_types[i] = 0
	if all_image_types[i] == 'Type_2':
		all_image_types[i] = 1
	if all_image_types[i] == 'Type_3':
		all_image_types[i] = 2


#load test:
basepath1 = '/data/kaggle/test/'

all_cervix_images_test = []
all_cervix_images_test = sorted(glob(basepath1 + "/*"))
#all_cervix_images_test = all_cervix_images_test + cervix_images

all_cervix_images_test = pd.DataFrame({'imagepath': all_cervix_images_test})


images_test = []
sample_counter = 0
for _, row in all_cervix_images_test.iterrows():
	try:
        	img = imread(row.imagepath)
		sample_counter +=1
		images_test.append(img)
	except:
		print('image read failed for {}'.format(row.imagepath))

dfs = []
t_ = pd.DataFrame(
	{
	'nrows': list(map(lambda i: i.shape[0], images_test)),
        'ncols': list(map(lambda i: i.shape[1], images_test)),
        'nchans': list(map(lambda i: i.shape[2], images_test)),
        }
)
dfs.append(t_)

shapes_df = pd.concat(dfs, axis=0)
shapes_df_grouped = shapes_df.groupby(by=['nchans', 'ncols', 'nrows']).size().reset_index()
shapes_df_grouped

all_images_test = []
all_images_test = images_test

gray_all_images_as_vecs_test = [transform_image(img, rescaled_dim) for img in all_images_test]
gray_imgs_mat_test = np.array(gray_all_images_as_vecs_test).squeeze()


#from __future__ import division, print_function, absolute_import
#import tflearn
#from tflearn.data_utils import shuffle, to_categorical
#from tflearn.layers.core import input_data, dropout, fully_connected
#from tflearn.layers.conv import conv_2d, max_pool_2d
#from tflearn.layers.estimator import regression
#from tflearn.data_preprocessing import ImagePreprocessing
#from tflearn.data_augmentation import ImageAugmentation


gray_all_images_as_vecsx, all_image_typesx = shuffle(gray_all_images_as_vecs, all_image_types)
t0=len(all_image_typesx[all_image_typesx=='0'])
t1=len(all_image_typesx[all_image_typesx=='1'])
t2=len(all_image_typesx[all_image_typesx=='2'])
Y=[]
tmp=all_image_typesx[all_image_typesx=='0']
Y=tmp[0:int(t0/5*4)]
tmp=all_image_typesx[all_image_typesx=='1']
Y=np.concatenate((Y,tmp[0:int(t1/5*4)]),axis=0)
tmp=all_image_typesx[all_image_typesx=='2']
Y=np.concatenate((Y,tmp[0:int(t2/5*4)]),axis=0)

X=[]
tmp=gray_all_images_as_vecsx[all_image_typesx=='0']
X=tmp[0:int(t0/5*4)]
tmp=gray_all_images_as_vecsx[all_image_typesx=='1']
X=np.concatenate((X,tmp[0:int(t1/5*4)]),axis=0)
tmp=gray_all_images_as_vecsx[all_image_typesx=='2']
X=np.concatenate((X,tmp[0:int(t2/5*4)]),axis=0)


Y_test=[]
tmp=all_image_typesx[all_image_typesx=='0']
Y_test=tmp[int(t0/5*4):t0]
tmp=all_image_typesx[all_image_typesx=='1']
Y_test=np.concatenate((Y_test,tmp[int(t1/5*4):t1]),axis=0)
tmp=all_image_typesx[all_image_typesx=='2']
Y_test=np.concatenate((Y_test,tmp[int(t2/5*4):t2]),axis=0)

X_test=[]
tmp=gray_all_images_as_vecsx[all_image_typesx=='0']
X_test=tmp[int(t0/5*4):t0]
tmp=gray_all_images_as_vecsx[all_image_typesx=='1']
X_test=np.concatenate((X_test,tmp[int(t1/5*4):t1]),axis=0)
tmp=gray_all_images_as_vecsx[all_image_typesx=='2']
X_test=np.concatenate((X_test,tmp[int(t2/5*4):t2]),axis=0)

Y = to_categorical(Y, 3)
Y_test = to_categorical(Y_test, 3)

np.save('/home/u3749/code/64_rescale_X.npy',X)
np.save('/home/u3749/code/64_rescale_Y.npy',Y)
np.save('/home/u3749/code/64_rescale_X_test.npy',X_test)
np.save('/home/u3749/code/64_rescale_Y_test.npy',Y_test)

np.save('/home/u3749/code/64_rescale_gray_all_images_as_vecs.npy',gray_all_images_as_vecs)
np.save('/home/u3749/code/64_rescale_all_image_types.npy',to_categorical(all_image_types,3))

X=np.load('/home/u3749/code/64_rescale_X.npy')
Y=np.load('/home/u3749/code/64_rescale_Y.npy')
X_test=np.load('/home/u3749/code/64_rescale_X_test.npy')
Y_test=np.load('/home/u3749/code/64_rescale_Y_test.npy')


XA=np.load('/home/u3749/code/64_rescale_gray_all_images_as_vecs.npy')
YA=np.load('/home/u3749/code/64_rescale_all_image_types.npy')


# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)
img_aug.add_random_flip_updown()


# Convolutional network building
network = input_data(shape=[None, 64, 64, 3],
                     data_preprocessing=img_prep,
                     data_augmentation=img_aug)
#network = input_data(shape=[None, 32, 32, 3])
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 512, activation='relu',regularizer='L2')
network = dropout(network, 0.5)
network = fully_connected(network, 64, activation='relu',regularizer='L2')
network = dropout(network, 0.5)
network = fully_connected(network, 64, activation='relu',regularizer='L2')
network = dropout(network, 0.5)
network = fully_connected(network, 3, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(X, Y, n_epoch=10, shuffle=True, validation_set=(X_test, Y_test),show_metric=True, batch_size=96, run_id='cvix_cnn')

model.fit(XA, YA, n_epoch=10, shuffle=True, validation_set=(test, testlabx),show_metric=True, batch_size=96, run_id='cvix_cnn')

model.fit(X, Y, n_epoch=100, shuffle=True, validation_set=0.3,show_metric=True, batch_size=96, run_id='cvix_cnn')

model.fit(gray_imgs_mat, to_categorical(all_image_types,3), n_epoch=30, shuffle=True, validation_set=0.1,show_metric=True, batch_size=96, run_id='cvix_cnn')

model.load('/home/u3749/code/tflearn_cvx_default.tflearn')
pv=model.predict(gray_all_images_as_vecs_test)

all_file=np.array(sorted(glob("*")))
all_file=pd.DataFrame(all_file)
pvx=pd.DataFrame(pv)
res=np.concatenate((all_file,pvx),axis=1)

res=np.concatenate((all_cervix_images_test,pv),axis=1)

import csv
with open('/home/u3749/code/new/output_32_stage2.csv','w') as f:
       writer = csv.writer(f,delimiter=',')
       writer.writerows(res)


find the failed example in the test

pv=model.predict(X_test)
for i in range(1,297):
    print(np.sum(X_test[i]))
for i in range(1,1183):
    print(np.sum(X[i]))


for i in range(1,1480):
    print(np.sum(gray_all_images_as_vecs[i]))

for i in range(1,512):
    print(np.sum(gray_imgs_mat_test[i]))

from sklearn.manifold import TSNE
tsne = TSNE(
    n_components=3,
    init='random', # pca
    random_state=101,
    method='barnes_hut',
    n_iter=500,
    verbose=2
).fit_transform(gray_imgs_mat)

from sklearn import preprocessing

trace1 = go.Scatter3d(
    x=tsne[:,0],
    y=tsne[:,1],
    z=tsne[:,2],
    mode='markers',
    marker=dict(
        sizemode='diameter',
        color = preprocessing.LabelEncoder().fit_transform(all_image_types),
        colorscale = 'Portland',
        colorbar = dict(title = 'cervix types'),
        line=dict(color='rgb(255, 255, 255)'),
        opacity=0.9
    )
)

data=[trace1]
layout=dict(height=800, width=800, title='3D embedding of images')
fig=dict(data=data, layout=layout)
py.plot(fig, filename='3DBubble',image='png')

plt.figure(figsize=(10,8))
plt.hold(True)
for t in all_cervix_images['type'].unique():
    tsne_t = tsne[np.where(all_image_types == t), :][0]
    plt.scatter(tsne_t[:, 0], tsne_t[:, 1])
#    plt.hold(True)
plt.legend(all_cervix_images['type'].unique())
plt.savefig('/home/u3749/result/heteroge.png')
plt.close(fig)

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
def imscatter(x, y, images, ax=None, zoom=0.01):
    ax = plt.gca()
    images = [OffsetImage(image, zoom=zoom) for image in images]
    artists = []
    for x0, y0, im0 in zip(x, y, images):
        ab = AnnotationBbox(im0, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    #return artists

nimgs = 60
plt.figure(figsize=(10,8))
imscatter(tsne[0:nimgs,0], tsne[0:nimgs,1], all_images[0:nimgs])

mask = np.zeros_like(sq_dists, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(12,12))
sns.heatmap(sq_dists, cmap=plt.get_cmap('viridis'), square=True, mask=mask)

from scipy.spatial.distance import pdist, squareform

sq_dists = squareform(pdist(gray_imgs_mat))

all_image_types = list(all_image_types)

d = {
    'Type_1': pal[0],
    'Type_2': pal[1],
    'Type_3': pal[2]
}

# translate each sample to its color
colors = list(map(lambda t: d[t], all_image_types))

sns.clustermap(
    sq_dists,
    figsize=(12,12),
    row_colors=colors, col_colors=colors,
    cmap=plt.get_cmap('viridis')
)

plt.savefig('/home/u3749/result/heatmap.png')

mask = np.zeros_like(sq_dists, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(12,12))
fig=sns.heatmap(sq_dists, cmap=plt.get_cmap('viridis'), square=True, mask=mask)
figx=fig.get_figrue()
figx.savefig('/home/u3749/result/matrix.png')

# upper triangle of matrix set to np.nan
sq_dists[np.triu_indices_from(mask)] = np.nan
sq_dists[0, 0] = np.nan

fig = plt.figure(figsize=(12,8))
# maximally dissimilar image
ax = fig.add_subplot(1,3,1)
maximally_dissimilar_image_idx = np.nanargmax(np.nanmean(sq_dists, axis=1))
plt.imshow(all_images[maximally_dissimilar_image_idx])
plt.title('maximally dissimilar')

# maximally similar image
ax = fig.add_subplot(1,3,2)
maximally_similar_image_idx = np.nanargmin(np.nanmean(sq_dists, axis=1))
plt.imshow(all_images[maximally_similar_image_idx])
plt.title('maximally similar')

# now compute the mean image
ax = fig.add_subplot(1,3,3)
mean_img = gray_imgs_mat.mean(axis=0).reshape(rescaled_dim, rescaled_dim, 3)
plt.imshow(cv2.normalize(mean_img, None, 0.0, 1.0, cv2.NORM_MINMAX))
plt.title('mean image')

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
y = LabelEncoder().fit_transform(all_image_types).reshape(-1)
X = gray_imgs_mat # no need for normalizing, we already did this earlier Normalizer().fit_transform(gray_imgs_mat)
X.shape, y.shape

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

