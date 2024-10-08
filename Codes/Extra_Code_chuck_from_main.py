#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 01:10:44 2022

@author: zahid
"""
#%% Prepare dataset for training

# 40 frames considered to to equivalent to 85 samples in PPg
import random
import numpy as np
random.seed(1)
rv = np.arange(0,6000, 1)+500
np.random.shuffle(rv)


# rv = [randint(0, 5300) for _ in range(5000)] ## random removal 
rv =  np.array(rv)
pulR = np.reshape(ppgnp_align, [ppgnp_align.shape[0],1]) # take 45 frames together
#      #%%
if 'trainX' in locals():
    print("already exists")
else:
    trainX = []
    trainY = []

data_align = data_align[:,:,:,np.newaxis]
frame_cons = 40 # how many frame to consider at a time

# Prepare the training instanses
for j,i in enumerate(rv):
    img = np.reshape(data_align[i:i+frame_cons,:,:,0], [frame_cons, *im_size])
    img = np.moveaxis(img, 0,-1)
    trainX.append(img)
    p_point = np.int(np.round(i*64/30))
    ppg = pulR[p_point: p_point+85, 0]
    trainY.append(ppg)
    
#%%    Normalize and split 
# data preprocessing step for person 1


trainY = np.array(trainY, dtype = np.float32)
trainY = trainY - trainY.min(axis = 1)[:, np.newaxis]
trainY = (trainY/(trainY.max(axis = 1)[:, np.newaxis]+ 10**-5))*2-1

trainX = np.array(trainX, dtype = np.float32)
trainX = (trainX-trainX.min())
trainX = trainX/ trainX.max()

trX, teX, trY, teY = train_test_split(trainX , trainY, 
                                      test_size = .1, random_state = 42)

#%% Some parameter definition

num_classes = 85
num_features = 100*100*40 

# Training parameters. Sunday, May 24, 2020 
learning_rate = 0.0008 # start with 0.001
training_steps = 50000
batch_size = 16
display_step = 100
#%% Data preparation
import pdb

def get_data():
    i  = randint(40, 8900)
    # print(i)
    q1 =  np.transpose(data[i:i+40, :,:,1], [1,2,0]) # for now green channel only
    k_pos1 = rand_crop_tf(q1)
    k_pos11 = tf.image.resize(np.transpose(data[(i+26):i+66, :,:,1], [1,2,0]), size = im_size)
    ll = list([k_pos1, k_pos11])
    # pdb.set_trace()
    k_pos = ll[randint(0, len(ll)-1)]    
    
    i1 = i+np.random.choice([-1,1])*randint(13,18)
    q2 =  np.transpose(data[i1:i1+40, :,:,1], [1,2,0]) # for now green channel only
    k_pos2 = rand_crop_tf(q2)
    
    q1 = tf.image.resize(q1, size = im_size)
    q2 = tf.image.resize(q2, size = im_size)
    
    k_neg1 = rand_frame_shuf(q1.numpy())
    k_neg2 = frame_repeat(q1.numpy())
    
    k_neg3 = rand_frame_shuf(q2.numpy())
    k_neg5 = rand_frame_shuf(q2.numpy())
    k_neg4 = frame_repeat(q2.numpy())
    
    x_train = tf.stack([q1, k_pos, q2, k_pos2, k_neg1, k_neg2, k_neg3, k_neg4, k_neg5])/255.0
    
    return x_train


#%% tensorflow dataload
# Run only for person 1

train_data = tf.data.Dataset.from_tensor_slices((trX, trY))
# train_data = train_data.repeat().shuffle(buffer_size=100,
#                                          seed= 8).batch(batch_size).prefetch(1)


train_data = train_data.shuffle(buffer_size=100,
                                         seed= 8).batch(batch_size).prefetch(1)
#%% Training the actual network
# single net
# inarg = (neural_net, train_data)
# multi-task net

inarg = (neural_net1, neural_net2, train_data)

with tf.device('gpu:0/'):
    train_nn(*inarg)


#%% Keras Model Replicate

import keras

from keras.models import Model
from keras.layers import Conv2D, MaxPool2D,  \
    Dropout, Dense, Input, concatenate,      \
    GlobalAveragePooling2D, AveragePooling2D,\
    Flatten, BatchNormalization, ReLU, AveragePooling2D

kernel_init ='he_uniform'

bias_init ='he_uniform'

def inception_module(x,
                     filters_1x1,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5 ,
                     filters_pool_proj,
                     strides =1,
                     name=None):
    
    conv_1x1 = Conv2D(filters_1x1, kernel_size=1, padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init, strides= strides)(x)
    
    conv_3x3 = Conv2D(filters_3x3, kernel_size = 3, padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init, strides =strides)(x)

    conv_5x5 = Conv2D(filters_5x5_reduce, kernel_size= 5, padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init, strides =strides)(x)
    conv_5x5 = Conv2D(filters_5x5, kernel_size = 3, padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init, strides =strides)(conv_5x5)

    pool_proj = MaxPool2D(3, strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init, strides =strides)(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)
    
    return output

def convBnRelu(x, 
               ch, 
               kernel_size = 3, 
               strides = 1,
               padding = 'same'):
    
    output = Conv2D(ch, kernel_size =  kernel_size, strides=strides, padding=padding,
                          kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    output = BatchNormalization()(output)
    output = ReLU()(output)
    return output


input_layer = Input(shape=(100, 100, 40))

x = convBnRelu(x= input_layer, ch = 32)

x = convBnRelu(x= x, ch = 64)


x = MaxPool2D(2, strides=2)(x)

x = convBnRelu(x= x, ch = 64)

x = convBnRelu(x= x, ch = 96)


x = MaxPool2D(2, strides=2)(x)



x = inception_module(x,
                     filters_1x1=16,
                     filters_3x3=16,
                     filters_5x5_reduce=16,
                     filters_5x5=16,
                     filters_pool_proj=16,
                     strides =1,
                     name='inception_3a')

x = AveragePooling2D(2, strides = 2)(x)

x = inception_module(x,
                     filters_1x1=16,
                     filters_3x3=16,
                     filters_5x5_reduce=16,
                     filters_5x5=16,
                     filters_pool_proj=16,
                     strides =1,
                     name='inception_3b')

x = AveragePooling2D(2, strides = 2)(x)

x = AveragePooling2D(2, strides = 2)(x)

x = Flatten()(x)



x = Dense(512, activation='relu')(x)


x = Dense(128, activation='relu')(x)


x = Dense(85, activation='tanh')(x)



model = Model(input_layer, x, name='inception_v1')

model.summary()

#%% keras optimizer

optimizer  = tf.optimizers.SGD(learning_rate)

model.compile(optimizer=optimizer,
              loss=RootMeanSquareLoss1,
              metrics=['mse'])

#%% keras Training loop [well not in per with current setting]
# may require this https://towardsdatascience.com/write-a-custom-training-routine-for-your-keras-model-a9139312a8a9
with tf.device('gpu:0/'): 
    model.fit(
        # trainX, trainY,
        train_data,
        batch_size = 16,
        # validation_split=0.1,
        epochs = 2
        )

#%% Log file for saving parameters and accuracy [grid search experiments]

import logging

# logging.info("Learning rate = {learning_rate}".format(**kwargs))
# logging.warning('Learning Rate: {learning_rate}, te_arr: {Accuracy}'.format(**kwargs))


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler('hello.log')
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the file handler to the logger
logger.addHandler(handler)

#%% Saving infomation in the hello.log file

kwargs = {'learning_rate':learning_rate, 'te_err': val_loss[-1].numpy(),
          'tr_err':train_loss[-1], 'Person':'Rini'}

logger.info(
    '''Learning Rate: {learning_rate}, val_accurcy: {te_err}, 
    train_loss: {tr_err}, Subject: {Person}'''.format(**kwargs))

#%% Random test from different starting point

# modification in network 

# performance measurement. 

# peak penalize (except mse)

# Don't forget to align HERE!! 

i = 811

fig=plt.figure(figsize=(8, 8))
columns = 3
rows = 3
for j in range( 1, columns*rows +1 ):
    
    i =randint( 50, 5100)
    i=  7500+j*40
    print(i)
    tX = np.reshape(data_align[i:i+40,:,:,:], [40,100,100])
    tX = np.moveaxis(tX, 0,-1) # very important line in axis changeing 
    
    p_point = np.int(np.round(i*64/30))
    
    gt = pulR[p_point: p_point+85, 0]

    gt = (gt-gt.min())/(gt.max()-gt.min())
    
    # i  = 5+j +j
    # tX = teX1[i]    
    # gt = 0.5*(teY1[i]+1)    
    # tX = teX[i]    
    # gt = 0.5*(teY[i]+1)
    
    fig.add_subplot(rows, columns, j)
    tX1 = np.reshape(tX, [-1, 100,100,40])
    plt.plot(gt*2-1)
    
    tX1 = (tX1 - tX1.min())/(tX1.max() - tX1.min())
    

    # predd = neural_net(trX1) 
    
    
    predd = model(tX1) 
    plt.plot(predd[0])

    plt.legend(["Ground Truth", "Predicted"])
    plt.xlabel('time sample \n (60 samples = 1 second)', fontsize =12)
    plt.ylabel('magnitude \n (Normalized voltage)', fontsize = 12)
    from matplotlib import rcParams
    rcParams['lines.linewidth'] = 2
    rcParams['lines.color'] = 'r'
   
# plt.savefig('sub4goodres.eps', format = 'eps', dpi= 600)
plt.show()

#%% Saving results

pred_train = []
gt_train = []


for j in range( 1,2000):
    
    #i =randint( 50, 5100)
    i=  5050+j

    tX = np.reshape(data_align[i:i+40,:,:,:], [40,100,100])
    tX = np.moveaxis(tX, 0,-1) # very important line in axis changeing 
    
    p_point = np.int(np.round(i*64/30))
    
    gt = pulR[p_point: p_point+85, 0]

    gt = (gt-gt.min())/(gt.max()-gt.min())
    
    gt = gt.reshape([1,85])

    tX1 = np.reshape(tX, [-1, 100,100,40])
    
    tX1 = (tX1 - tX1.min())/(tX1.max() - tX1.min())
    

    # predd = neural_net(trX1) 
    
    
    predd = model(tX1) 
    
    gt_train.append(gt)
    pred_train.append(predd)

# plt.savefig('sub4goodres.eps', format = 'eps', dpi= 600)
plt.show()



#%% Seeing inside the network
in1 = neural_net1.layers[0].layers[0](tX1).numpy() # plt.plot(in1[0,:,:,1])
in2 = neural_net1.layers[0].layers[1](in1).numpy() # plt.plot(in2[0,:,:,1])  
in3 = neural_net1.layers[0].layers[2](in2).numpy()
in4 = neural_net1.layers[0].layers[3](in3).numpy()
in5 = neural_net1.layers[0].layers[4](in4).numpy()
in6 = neural_net1.layers[0].layers[5](in5).numpy()
in7 = neural_net1.layers[0].layers[6](in6).numpy()
in8 = neural_net1.layers[0].layers[7](in7).numpy()
in9 = neural_net1.layers[0].layers[8](in8).numpy()


# alternative neural_net1.layers[0].layers[0].model.layers[2]  or use the name like conv1, ..
# in3 = neural_net.layers[2](in2).numpy()

# neural_net1.layers[0].layers[0].model.layers[0].weights or .bias

# ##we can also select the model inside the inside layer

#neural_net.layers[0].layers[0].layers[1]

# Think of tensorflow 2 very flexible as long as we define the model very clearly

# Think of each of the layer and apply independently even when there is shared network inside...
# in2=neural_net.layers[0](trX1[:,:,:,10:20]).numpy()
# well we can go on

# ways to get the weights and biases. the layers should be in our mind in the firsthand 
#neural_net.layers[0].layers[0].layers[0].weights
# Keep track what you did in call and what you have in model layer definition

#%%  Video Sequence check

for i in range(40):
    print(i)
    plt.imshow(trainX[1000,:,:,i])
    plt.show()# create target from video itself. 

#%%    Normalize and split
trainX = np.array(trainX, dtype = np.float32)
trainY = np.array(trainY, dtype = np.float32)


trainY = trainY - trainY.min(axis = 1)[:, np.newaxis]
# trainY = trainY/(trainY.max(axis = 1)[:,np.newaxis]
    
    
#%%    Get_weights and Set_weights from trained network

# neural_net.layers[7].set_weights(Basenet.layers[7].get_weights())
#weightss = np.array(neural_net.layers[0].layers[0].layers[0].weights)

#%% Plotting learning curves

tr_l = np.array(train_loss)


val_l = np.array(val_loss)

fig = plt.figure(figsize=(19.20,10.80))
# For suject 2 go till 6230
# For subject 3 go till 7100
plt.plot(tr_l, 'r', val_l, 'g')

plt.xlabel( "training step" )
plt.ylabel( "Errors in MSE" )
plt.title( "Sample Learning Curve" )

lst = ["Training", 'Validation']

plt.legend(lst)

plt.savefig('learning_curve.eps', format = 'eps', dpi= 1000)

#%%  Better visualization

fig=plt.figure(figsize=(8, 8))
columns = 3
rows = 3
for i in range(1, columns*rows + 1):
    img = in5[0, :,:, 40+i]
    fig.add_subplot(rows, columns, i)
    plt.imshow(img)

# plt.savefig('inter_filt', dpi= 1000)
# plt.show()



#%% PPG visulization

plt.plot(pulR[500:4000])
plt.xlabel('time')
plt.ylabel('PPG magnitude')
plt.title("PPG magnitude changes")

#%% Signal Reconstruction in the test time 

divVec = np.ones([85])
divVec1 = np.zeros([85]) 
gtV = np.zeros([85])
recPPG = np.zeros([85])

for j in range(5):
    
    olap = 40
    i = 2000 +j*olap
    print(i)
    tX = np.reshape(data_align[i:i+40,:,:,:], [40,100,100])
    tX = np.moveaxis(tX, 0,-1) # very important line in axis changeing 
     
    p_point = np.int(np.round(i*64/30))
    
    gt = pulR[p_point: p_point+85, 0]
    gt = (gt-gt.min())/(gt.max()-gt.min())
    
    # i  = 5+j +j
    # tX = teX1[i]    
    # gt = 0.5*(teY1[i]+1)    
    # tX = teX[i]    
    # gt = 0.5*(teY[i]+1)    

    tX1 = np.reshape(tX, [-1, 100,100,40])
    tX1 = (tX1 - tX1.min())/(tX1.max() - tX1.min())
    
    olap =  np.int(olap*64/30)
    
    # predd = neural_net(trX1) 
    predd = neural_net1(tX1)
    
    recPPG[-85:] = recPPG[-85:] + predd
    
    recPPG = np.concatenate((recPPG, np.zeros([olap])))
    
    
    gtV[-85:] = gtV[-85:] + np.squeeze(gt*2-1)
    gtV = np.concatenate((gtV, np.zeros([olap])))
    
    
    divVec1[-85:] = divVec1[-85:]+divVec
    divVec1 = np.concatenate((divVec1, np.zeros([olap])))
    

fig = plt.figure(figsize=(19.20,10.80))
plt.plot(gtV[:-80], 'C2')
plt.plot(recPPG[:-80], 'C3')
plt.legend(["Ground Truth", "Predicted"], fontsize = 42, loc = "upper right", ncol = 2) 
plt.xlabel('time sample (60 samples = 1 second)', fontsize =40, fontweight = 'bold')
plt.ylabel('PPG magnitude \n (Normalized voltage)', fontsize = 40, fontweight= 'bold')

plt.title("PPG plot for the collected data", fontsize = 40, fontweight = 'bold')

plt.margins(x =0, y =0.17)
# from matplotlib import rcParams
# rcParams['lines.linewidth'] = 4
# rcParams['lines.color'] = 'r'
plt.xticks(fontsize = 35, fontweight = 'bold')
plt.yticks(fontsize = 35, fontweight =  'bold')

import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 5
mpl.rcParams['lines.color'] = 'r'
mpl.rcParams['font.weight'] = 200
plt.style.use('seaborn-whitegrid')

mpl.rcParams['axes.labelsize'] = 30
mpl.rcParams['axes.linewidth'] = 5
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20
mpl.rcParams['axes.edgecolor'] = 'black'
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams["legend.frameon"] = True

# plt.savefig('cd_sample_res.eps', format = 'eps', dpi= 500)

plt.show()

#%% positive augmentation 

# croping
# # non-tensor 
# def rand_crop_array(img): # input size (l,w, channel)
#     return cv2.resize(img[100:400, 100:400,:], (100, 100))

# tensor 
im_size = (100, 100)
def rand_crop_tf(img):
    [temp1, temp2] =[randint(200,400),randint(200,400)]
    return tf.image.resize(tf.image.random_crop(img, size=(temp1, temp2, 40)), size = im_size)

#%% negative augmentation 
# we can use multiple shuffling
def rand_frame_shuf(img):
    temp = np.arange(40)
    np.random.shuffle(temp)
    return img[:,:, temp]

def frame_repeat(img):
    temp = randint(0,39)
    img[:,:,0:40] = img[:,:, temp:temp+1]
    return img
#%% 

fig = plt.figure(figsize=(19.20,10.80))
ax = fig.add_subplot()
ax.scatter(X_emb[0:1000,0], X_emb[0:1000,1], s = 100, cmap=1, marker='v', label = "High Frequency")
ax.scatter(X_emb[1000:2000,0], X_emb[1000:2000,1], s= 100, cmap = 0, marker='1', label = 'Low Frequency')
ax.legend(fontsize=25, loc = "lower right", ncol = 1)
ax.set_title('Frequency Learning by Pretraining', fontsize =30, fontweight = 'bold')
ax.set_xlabel('Embedding Dimension 1', fontsize =28, fontweight = 'bold')
ax.set_ylabel('Embedding Dimension 2', fontsize = 28, fontweight= 'bold')


fig.savefig('Freqs.svg', format = 'svg', dpi= 500)

#%%

fig = plt.figure(figsize=(19.20,10.80))
plt.plot(ppgs[2], marker = '<',  linewidth = 2, markersize = 15)
plt.plot(ppgs[3], marker = 'o',  linewidth = 2, markersize = 15)
plt.plot(ppgs[6], marker = 'x',  linewidth = 2, markersize = 15)
plt.legend(["Sample PPG", "Half-Cycle Shifted PPG", "Full-Cycle Shifted PPG"], fontsize = 25, loc = "upper left", ncol = 3, fancybox = True) 


plt.xlabel('Time sample (60 sample = 1 second)', fontsize =28, fontweight = 'bold')
plt.ylabel('PPG magnitude \n (Normalized voltage)', fontsize = 28, fontweight= 'bold')

plt.title("Time Shift can generate similar and dissimilar PPG", fontsize = 30, fontweight = 'bold')
plt.savefig('shift_.svg', format = 'svg', dpi = 500)