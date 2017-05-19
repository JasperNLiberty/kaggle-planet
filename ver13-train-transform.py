# 128x128 jpg

# 7 more layers added to vgg16 --> 512 x 2 x 2

# arbitrary numbers to train (N), validate (V < M) and test (M)

# THEANO_FLAGS=mode=FAST_RUN,device=cuda,floatX=float32 python

# cd /Users/user/Documents/learning_materials/kaggle-planet-gpu
train_dir_name = '/Users/user/Documents/learning_materials/kaggle_planet/train-jpg/'
test_dir_name = '/Users/user/Documents/learning_materials/kaggle_planet/test-jpg/'
N = 40479 #40479
V = 200 # < M
M = 61191 #61191


## setup
import os
import pickle
from keras import backend as K
import cv2
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
from tqdm import tqdm, tqdm_notebook
import sys
import pandas as pd
from sklearn.metrics import fbeta_score
sys.setrecursionlimit(3000)
K.set_image_dim_ordering('th')
pd.set_option('max_colwidth', 400)
np.core.arrayprint._line_width = 999
np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)

# path to the model weights file.
weights_path = 'vgg16_weights.h5'
top_model_weights_path = 'bottleneck_fc_model.h5'


## create model using weights
model_load = Sequential()
model_load.add(ZeroPadding2D((1, 1), input_shape=(3, 128, 128)))
model_load.add(Convolution2D(64, (3, 3), activation='relu', name='conv1_1'))
model_load.add(ZeroPadding2D((1, 1)))
model_load.add(Convolution2D(64, (3, 3), activation='relu', name='conv1_2'))
model_load.add(MaxPooling2D((2, 2), strides=(2, 2)))

model_load.add(ZeroPadding2D((1, 1)))
model_load.add(Convolution2D(128, (3, 3), activation='relu', name='conv2_1'))
model_load.add(ZeroPadding2D((1, 1)))
model_load.add(Convolution2D(128, (3, 3), activation='relu', name='conv2_2'))
model_load.add(MaxPooling2D((2, 2), strides=(2, 2)))

model_load.add(ZeroPadding2D((1, 1)))
model_load.add(Convolution2D(256, (3, 3), activation='relu', name='conv3_1'))
model_load.add(ZeroPadding2D((1, 1)))
model_load.add(Convolution2D(256, (3, 3), activation='relu', name='conv3_2'))
model_load.add(ZeroPadding2D((1, 1)))
model_load.add(Convolution2D(256, (3, 3), activation='relu', name='conv3_3'))
model_load.add(MaxPooling2D((2, 2), strides=(2, 2)))

model_load.add(ZeroPadding2D((1, 1)))
model_load.add(Convolution2D(512, (3, 3), activation='relu', name='conv4_1'))
model_load.add(ZeroPadding2D((1, 1)))
model_load.add(Convolution2D(512, (3, 3), activation='relu', name='conv4_2'))
model_load.add(ZeroPadding2D((1, 1)))
model_load.add(Convolution2D(512, (3, 3), activation='relu', name='conv4_3'))
model_load.add(MaxPooling2D((2, 2), strides=(2, 2)))

model_load.add(ZeroPadding2D((1, 1)))
model_load.add(Convolution2D(512, (3, 3), activation='relu', name='conv5_1'))
model_load.add(ZeroPadding2D((1, 1)))
model_load.add(Convolution2D(512, (3, 3), activation='relu', name='conv5_2'))
model_load.add(ZeroPadding2D((1, 1)))
model_load.add(Convolution2D(512, (3, 3), activation='relu', name='conv5_3'))
model_load.add(MaxPooling2D((2, 2), strides=(2, 2)))

model_load.add(ZeroPadding2D((1, 1)))
model_load.add(Convolution2D(512, (3, 3), activation='relu', name='conv6_1'))
model_load.add(ZeroPadding2D((1, 1)))
model_load.add(Convolution2D(512, (3, 3), activation='relu', name='conv6_2'))
model_load.add(ZeroPadding2D((1, 1)))
model_load.add(Convolution2D(512, (3, 3), activation='relu', name='conv6_3'))
model_load.add(MaxPooling2D((2, 2), strides=(2, 2)))

# load the weights of the VGG16 networks
# (trained on ImageNet, won the ILSVRC competition in 2014)
# note: when there is a complete match between your model definition
# and your weight savefile, you can simply call model.load_weights(filename)
assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
f = h5py.File(weights_path)
for k in range(f.attrs['nb_layers']):
    if k >= len(model_load.layers) - 7:
        # we don't look at the last (fully-connected) layers in the savefile
        break
    g = f['layer_{}'.format(k)]
#     weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    weights = []
    for p in range(g.attrs['nb_params']):
        if len(g['param_{}'.format(p)].shape) > 1:
            weights.append(np.transpose(g['param_{}'.format(p)], (2,3,1,0)))
        else:
            weights.append(g['param_{}'.format(p)])
    
    model_load.layers[k].set_weights(weights)
f.close()
print('Model loaded.')



# df_train
df_train = pd.read_csv('./train.csv')
flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))
label_map = {l: i for i, l in enumerate(labels)}
inv_label_map = {i: l for l, i in label_map.items()}
n_label = len(label_map)


# df_test
testdir = os.listdir(test_dir_name)
jpgdir = [test for test in testdir if '.jpg' in test]
df_test = pd.DataFrame()
df_test['image_name'] = [jpg.replace('.jpg','') for jpg in jpgdir]
df_test['tags'] = 0

# y_train
y_train = []
for f, tags in tqdm(df_train.values, miniters=10):
    targets = np.zeros(n_label)
    for t in tags.split(' '):
        targets[label_map[t]] = 1 
    y_train.append(targets)
y_train = np.array(y_train, np.uint8)[0:N]

# # x_train
# x_train = []
# for f, tags in tqdm(df_train.values[0:N], miniters=10):
#     img = cv2.imread(train_dir_name+'{}.jpg'.format(f))
#     targets = np.zeros(n_label)
#     for t in tags.split(' '):
#         targets[label_map[t]] = 1 
#     x_train.append(cv2.resize(img, (128, 128)))
# x_train = np.array(x_train, np.float16) / 255.   

# x_train.shape

# # x_transform
# x_train_transform = np.zeros((N,512,2,2))
# for xn in tqdm(range(N)):
#     im = x_train[xn]
#     im = im.transpose((2,0,1))
#     im = np.expand_dims(im, axis=0)
#     im_transform = model_load.predict(im)
#     x_train_transform[xn] = im_transform

# with open('x_train_transform', 'w') as f:
#     np.save(f, x_train_transform)

# start CNN training
with open('x_train_transform', 'r') as f:
    x_train_transform = np.load(f)



# split validation off train data
split = N - V
split2 = N - V
x_train0, x_valid0, y_train0, y_valid0 = x_train_transform[:split], x_train_transform[split2:], y_train[:split], y_train[split2:]

## train and save weights
model_train = Sequential()
model_train.add(Flatten(input_shape=x_train_transform.shape[1:]))
model_train.add(Dense(256, activation='relu'))
model_train.add(Dropout(0.25))
model_train.add(Dense(n_label, activation='sigmoid'))
model_train.compile(loss='binary_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
              optimizer='adam',
              metrics=['accuracy'])
model_train.fit(x_train0, y_train0,
          batch_size=32,
          epochs=50,
          verbose=1,
          validation_data=(x_valid0, y_valid0))




# # x_test
# x_test = []
# for f, tags in tqdm(df_test.values[0:M], miniters=10):
#     img = cv2.imread(test_dir_name+'{}.jpg'.format(f))
#     targets = np.zeros(n_label)
#     x_test.append(cv2.resize(img, (128, 128)))    
# x_test = np.array(x_test, np.float16) / 255.



# # x_test_transform
# x_test_transform = np.zeros((M,512,2,2))
# for xn in tqdm(range(M)):
#     im = x_test[xn]
#     im = im.transpose((2,0,1))
#     im = np.expand_dims(im, axis=0)
#     im_transform = model_load.predict(im)
#     x_test_transform[xn] = im_transform

    
# with open('x_test_transform', 'w') as f:
#     np.save(f, x_test_transform)

with open('x_test_transform', 'r') as f:
    x_test_transform = np.load(f)




# make predictions
data_to_predict = x_test_transform
p_test = model_train.predict(data_to_predict, batch_size=128)


# create label list from probablity list
final_p = p_test
preds = [' '.join([inv_label_map[y_pred_pos] for y_pred_pos, y_pred in enumerate(
    (y_pred_row > 0.18).astype(int))  if y_pred==1]) for y_pred_row in final_p]

# create submission file
subm = pd.DataFrame()
subm['image_name'] = df_test.image_name.values[0:M]
subm['tags'] = preds
subm.to_csv('submission.csv', index=False)