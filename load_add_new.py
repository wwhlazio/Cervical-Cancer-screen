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

t=all_cervix_images['type'].unique()[0]
t=all_cervix_images['type'].unique()[1]
t=all_cervix_images['type'].unique()[2]

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
	for i in range(len(all_image_types)):
		if all_image_types[i] == 'Type_1_v2':
		all_image_types[i] = 0
		if all_image_types[i] == 'Type_2_v2':
		all_image_types[i] = 1
		if all_image_types[i] == 'Type_3_v2':
		all_image_types[i] = 2

		image read failed for /data/kaggle_3.27/additional/Type_1_v2/3068.jpg
		image read failed for /data/kaggle_3.27/additional/Type_1_v2/5893.jpg
		file=all_cervix_images[all_cervix_images['type']==t]

filex=np.array(file)

	np.where(filex[0:1191,0]=='/data/kaggle_3.27/additional/Type_1_v2/3068.jpg')
filex=np.delete(filex,(471),axis=0)
	np.where(filex[0:1190,0]=='/data/kaggle_3.27/additional/Type_1_v2/5893.jpg')
filex=np.delete(filex,(895),axis=0)




	from numpy import genfromtxt
	rem = genfromtxt('/home/u3749/data/removed_filesx.csv',delimiter=',',dtype='unicode',skip_header=1)


	ind=[]
	u=0
	for j in range(len(filex)):
		for i in range(len(rem)):
			tmp=basepath+t+'/'+rem[i,0]	
			if(filex[j,0] == tmp):
	print(tmp)
	ind.append(u)
	u=u+1

	file0=np.delete(filex,ind,axis=0)
	all_image_types0=np.delete(all_image_types,ind,axis=0)
gray_imgs_mat0=np.delete(gray_imgs_mat,ind,axis=0)







	for type2:
	image read failed for /data/kaggle_3.27/additional/Type_2_v2/2845.jpg
	image read failed for /data/kaggle_3.27/additional/Type_2_v2/5892.jpg
	image read failed for /data/kaggle_3.27/additional/Type_2_v2/7.jpg

	file=all_cervix_images[all_cervix_images['type']==t]
filex=np.array(file)

	np.where(filex[0:3567,0]=='/data/kaggle_3.27/additional/Type_2_v2/2845.jpg')
filex=np.delete(filex,(1122),axis=0)
	np.where(filex[0:3566,0]=='/data/kaggle_3.27/additional/Type_2_v2/5892.jpg')
filex=np.delete(filex,(2812),axis=0)
	np.where(filex[0:3565,0]=='/data/kaggle_3.27/additional/Type_2_v2/7.jpg')
filex=np.delete(filex,(3382),axis=0)

	from numpy import genfromtxt
	rem = genfromtxt('/home/u3749/data/removed_filesx.csv',delimiter=',',dtype='unicode',skip_header=1)

	ind=[]
	u=0
	for j in range(len(filex)):
		for i in range(len(rem)):
			tmp=basepath+t+'/'+rem[i,0]
			if(filex[j,0] == tmp):
	print(tmp)
	ind.append(u)
	u=u+1


	file1=np.delete(filex,ind,axis=0)
	all_image_types1=np.delete(all_image_types,ind,axis=0)
gray_imgs_mat1=np.delete(gray_imgs_mat,ind,axis=0)




	for Type_3:
	file=all_cervix_images[all_cervix_images['type']==t]
filex=np.array(file)


	ind=[]
	u=0
	for j in range(len(filex)):
		for i in range(len(rem)):
			tmp=basepath+t+'/'+rem[i,0]
			if(filex[j,0] == tmp):
	print(tmp)
	ind.append(u)
	u=u+1

	file2=np.delete(filex,ind,axis=0)
	all_image_types2=np.delete(all_image_types,ind,axis=0)
gray_imgs_mat2=np.delete(gray_imgs_mat,ind,axis=0)


	del images
	del all_images

	from subprocess import check_output
	print(check_output(["ls", "/data/kaggle/train"]).decode("utf8"))

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

	t=all_cervix_images['type'].unique()[0]
	t=all_cervix_images['type'].unique()[1]
	t=all_cervix_images['type'].unique()[2]

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
	for i in range(len(all_image_types)):
		if all_image_types[i] == 'Type_1':
		all_image_types[i] = 0
		if all_image_types[i] == 'Type_2':
		all_image_types[i] = 1
		if all_image_types[i] == 'Type_3':
		all_image_types[i] = 2

		for type1
		image read failed for /data/kaggle/train/Type_1/1339.jpg
	file=all_cervix_images[all_cervix_images['type']==t]
filex=np.array(file)

	np.where(filex[0:250,0]=='/data/kaggle/train/Type_1/1339.jpg')
filex=np.delete(filex,(63),axis=0)

	filex1=np.delete(filex,ind,axis=0)
	all_image_typesx=np.delete(all_image_types,ind,axis=0)
gray_imgs_matx = np.delete(gray_imgs_mat,ind,axis=0)


	f1=np.concatenate((filex1,file0),axis=0)
	t1=np.concatenate((all_image_typesx,all_image_types0),axis=0)
im1=np.concatenate((gray_imgs_matx,gray_imgs_mat0),axis=0)

	np.save('/home/u3749/code/new/file_type1.npy',f1)
	np.save('/home/u3749/code/new/type_type1.npy',t1)
	np.save('/home/u3749/code/new/img_type1.npy',im1)

np.save('/home/u3749/code/new/train_file_type1.npy',filex1)
np.save('/home/u3749/code/new/add_file_type1.npy',file0)
np.save('/home/u3749/code/new/train_type_type1.npy',all_image_typesx)
np.save('/home/u3749/code/new/add_type_type1.npy',all_image_types0)
np.save('/home/u3749/code/new/train_img_type1.npy',gray_imgs_matx)
np.save('/home/u3749/code/new/add_img_type1.npy',gray_imgs_mat0)


	for type2

filex2=np.delete(filex,ind,axis=0)

	f2=np.concatenate((filex2,file1),axis=0)
	t2=np.concatenate((all_image_typesx,all_image_types1),axis=0)
im2=np.concatenate((gray_imgs_matx,gray_imgs_mat1),axis=0)

	np.save('/home/u3749/code/new/file_type2.npy',f2)
	np.save('/home/u3749/code/new/type_type2.npy',t2)
	np.save('/home/u3749/code/new/img_type2.npy',im2)

np.save('/home/u3749/code/new/train_file_type2.npy',filex2)
np.save('/home/u3749/code/new/add_file_type2.npy',file1)
np.save('/home/u3749/code/new/train_type_type2.npy',all_image_typesx)
np.save('/home/u3749/code/new/add_type_type2.npy',all_image_types1)
np.save('/home/u3749/code/new/train_img_type2.npy',gray_imgs_matx)
np.save('/home/u3749/code/new/add_img_type2.npy',gray_imgs_mat1)


	for type3
	file=all_cervix_images[all_cervix_images['type']==t]
filex=np.array(file)


	ind=[]
	u=0
	for j in range(len(filex)):
		for i in range(len(rem)):
			tmp=basepath+t+'/'+rem[i,0]
			if(filex[j,0] == tmp):
	print(tmp)
	ind.append(u)
	u=u+1

	filex3=np.delete(filex,ind,axis=0)
	all_image_typesx=np.delete(all_image_types,ind,axis=0)
gray_imgs_matx = np.delete(gray_imgs_mat,ind,axis=0)

	f3=np.concatenate((filex3,file2),axis=0)
	t3=np.concatenate((all_image_typesx,all_image_types2),axis=0)
im3=np.concatenate((gray_imgs_matx,gray_imgs_mat2),axis=0)

	np.save('/home/u3749/code/new/file_type3.npy',f3)
	np.save('/home/u3749/code/new/type_type3.npy',t3)
	np.save('/home/u3749/code/new/img_type3.npy',im3)

np.save('/home/u3749/code/new/train_file_type3.npy',filex3)
np.save('/home/u3749/code/new/add_file_type3.npy',file2)
np.save('/home/u3749/code/new/train_type_type3.npy',all_image_typesx)
np.save('/home/u3749/code/new/add_type_type3.npy',all_image_types2)
np.save('/home/u3749/code/new/train_img_type3.npy',gray_imgs_matx)
np.save('/home/u3749/code/new/add_img_type3.npy',gray_imgs_mat2)





ext = genfromtxt('/home/u3749/data/fixed_labels_v2.csv',delimiter=',',dtype='unicode',skip_header=1) 
ext1 = ext
for i in range(len(ext)):
    if ext[i,1] == 'Type_1':
        ext[i,1] = 0
    if ext[i,1] == 'Type_2':
        ext[i,1] = 1
    if ext[i,1] == 'Type_3':
        ext[i,1] = 2
    if ext[i,2] == 'Type_1':
        ext[i,2] = 0
    if ext[i,2] == 'Type_2':
        ext[i,2] = 1
    if ext[i,2] == 'Type_3':
        ext[i,2] = 2

f1=np.load('/home/u3749/code/new/add_file_type1.npy')
f2=np.load('/home/u3749/code/new/add_file_type2.npy')
f3=np.load('/home/u3749/code/new/add_file_type3.npy')
t1=np.load('/home/u3749/code/new/add_type_type1.npy')
t2=np.load('/home/u3749/code/new/add_type_type2.npy')
t3=np.load('/home/u3749/code/new/add_type_type3.npy')
im1=np.load('/home/u3749/code/new/add_img_type1.npy')
im2=np.load('/home/u3749/code/new/add_img_type2.npy')
im3=np.load('/home/u3749/code/new/add_img_type3.npy')


f=np.concatenate((f1,f2),axis=0)
f=np.concatenate((f,f3),axis=0)
t=np.concatenate((t1,t2),axis=0)
t=np.concatenate((t,t3),axis=0)
im=np.concatenate((im1,im2),axis=0)
im=np.concatenate((im,im3),axis=0)

np.save('/home/u3749/code/new/add_file',f)
np.save('/home/u3749/code/new/add_type',t)
np.save('/home/u3749/code/new/add_img',im)



fx=[]
for i in range(len(f)):
    tmp=f[i,0].split('/')
    fx.append(tmp[len(tmp)-1])

fx=np.array(fx)

tmp=np.intersect1d(ext[0:len(ext),0],fx)
for i in range(len(tmp)):
    indx1=np.where(fx==tmp[i])
    indx2=np.where(ext[0:len(ext),0]==tmp[i])
    t[indx1[0]]=ext[indx2[0],2]	

fx1=np.load('/home/u3749/code/new/train_file_type1.npy')
fx2=np.load('/home/u3749/code/new/train_file_type2.npy')
fx3=np.load('/home/u3749/code/new/train_file_type3.npy')
tx1=np.load('/home/u3749/code/new/train_type_type1.npy')
tx2=np.load('/home/u3749/code/new/train_type_type2.npy')
tx3=np.load('/home/u3749/code/new/train_type_type3.npy')
imx1=np.load('/home/u3749/code/new/train_img_type1.npy')
imx2=np.load('/home/u3749/code/new/train_img_type2.npy')
imx3=np.load('/home/u3749/code/new/train_img_type3.npy')


fc=np.concatenate((fx1,fx2),axis=0)
fc=np.concatenate((fc,fx3),axis=0)
tc=np.concatenate((tx1,tx2),axis=0)
tc=np.concatenate((tc,tx3),axis=0)
imc=np.concatenate((imx1,imx2),axis=0)
imc=np.concatenate((imc,imx3),axis=0)


fcc=[]
for i in range(len(fc)):
    tmp=fc[i,0].split('/')
    fcc.append(tmp[len(tmp)-1])
fcc=np.array(fcc)
    
train/Type_1/80 - type 3
train/Type_3/968 - type 1
train/Type_3/1120 - type 1

tc[np.where(fcc=='80.jpg')[0]]=2
tc[np.where(fcc=='968.jpg')[0]]=0
tc[np.where(fx3=='1120.jpg')[0]]=0


np.save('/home/u3749/code/new/train_file',fc)
np.save('/home/u3749/code/new/train_type',tc)
np.save('/home/u3749/code/new/train_img',imc)

fa=np.concatenate((f,fc),axis=0)
ta=np.concatenate((t,tc),axis=0)
ima=np.concatenate((im,imc),axis=0)

np.save('/home/u3749/code/new/all_file',fa)
np.save('/home/u3749/code/new/all_type',ta)
np.save('/home/u3749/code/new/all_img',ima)

basepath1 = '/data/test_stg2/'

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

