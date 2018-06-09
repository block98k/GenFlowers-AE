from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input,Lambda
from keras import backend as K
from keras.models import Model
from keras.layers import Dense,Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv2DTranspose,Reshape,concatenate,Reshape
from keras.layers.normalization import BatchNormalization
from keras import losses
from keras.optimizers import Adam
import cv2
import os
from tqdm import tqdm

input_size = 384
latent_dim = 128
num_data=8189
cwd='/home/meirtz/explore/CuoNet/Data/flowers/jpg/'


def processing(image):
    new_image=cv2.resize(image/255., (input_size,input_size))
    return new_image
def dataprepare():
    for i in tqdm(range(1,num_data+1)):
        if i<10:
            name='image_0000'+str(i)+'.jpg'
        elif i<100:
            name='image_000'+str(i)+'.jpg'
        elif i<1000:
            name='image_00'+str(i)+'.jpg'
        else:
            name='image_0'+str(i)+'.jpg'
        image=plt.imread(cwd+name)
        x[i-1]=processing(image)
        
x=np.empty((num_data,input_size,input_size,3),dtype='float32')
dataprepare()

def kk_block(X,num_filter):
    #3*3+3*3
    X1 = Conv2D(num_filter,(3, 3), padding='same')(X)
    X1 = BatchNormalization()(X1)
    X1 = Activation("relu")(X1)
    X1 = Conv2D(num_filter,(3, 3), padding='same',strides=(2,2))(X1)
    X1 = BatchNormalization()(X1)
    X1 = Activation("relu")(X1)
    
    #1*1+3*3
    X2 = Conv2D(num_filter,(1, 1), padding='same')(X)
    X2 = BatchNormalization()(X2)
    X2 = Activation("relu")(X2)
    X2 = Conv2D(num_filter,(3, 3), padding='same',strides=(2,2))(X2)
    X2 = BatchNormalization()(X2)
    X2 = Activation("relu")(X2)
    
    out= concatenate([X1,X2],axis=-1)
    return out
def aver(arg):
    x1,x2=arg
    return (x1+x2)/2
def half_1(x):
    half=x.shape[3]//2
    return x[:,:,:,0:half]
def half_2(x):
    half=x.shape[3]//2
    return x[:,:,:,half:x.shape[3]]
def kk_block_T(X,num_filter):
    X1 = Lambda(half_1)(X)
    X1 = Conv2DTranspose(num_filter,(3, 3), padding='same')(X1)
    X1 = BatchNormalization()(X1)
    X1 = Activation("relu")(X1)
    X1 = Conv2DTranspose(num_filter,(3, 3), strides=(2, 2),padding='same')(X1)
    X1 = BatchNormalization()(X1)
    X1 = Activation("relu")(X1)
    
    
    X2 = Lambda(half_2)(X)
    X2 = Conv2DTranspose(num_filter,(1, 1), padding='same')(X2)
    X2 = BatchNormalization()(X2)
    X2 = Activation("relu")(X2)
    X2 = Conv2DTranspose(num_filter,(3, 3), strides=(2, 2),padding='same')(X2)
    X2 = BatchNormalization()(X2)
    X2 = Activation("relu")(X2)
    
    out= Lambda(aver)([X1,X2])
    return out

net_input = Input(shape=(input_size,input_size,3))
X = Conv2D(16,(3, 3), padding='same',activation='relu')(net_input)
X = Conv2D(16,(3, 3), padding='same',activation='relu')(X)
X = MaxPooling2D((2, 2))(X)
X = kk_block(X,32)
X = kk_block(X,64)
X = kk_block(X,128)
X = kk_block(X,256)
X = kk_block(X,512)
X = kk_block(X,1024)
hidden=Conv2D(latent_dim,(3, 3), padding='valid')(X)
hidden=BatchNormalization()(hidden)
decoder_input = Conv2DTranspose(2048,(3, 3),padding='valid',activation='relu')(hidden)
X = kk_block_T(decoder_input,1024)
X = kk_block_T(X,512)
X = kk_block_T(X,256)
X = kk_block_T(X,128)
X = kk_block_T(X,64)
X = kk_block_T(X,32)
X = Conv2DTranspose(16,(3, 3), strides=(2, 2),padding='same')(X)
X = Conv2DTranspose(16,(3, 3), padding='same',activation='relu')(X)
out = Conv2DTranspose(3,(3, 3), padding='same',activation='relu')(X)
net = Model(net_input, out)

def GenFlower_loss(x_origin,x_out):
    x_origin=K.flatten(x_origin)
    x_out=K.flatten(x_out)
    loss = losses.binary_crossentropy(x_origin, x_out)
    return loss
optimizers=Adam(lr=0.00005)
net.compile(optimizer=optimizers, loss=GenFlower_loss)

epochs = 200
batch_size = 48
if os.path.exists('my_model_weights.h5'):
    net.load_weights('my_model_weights.h5')
net.fit(x,x,shuffle=True,
        epochs=epochs,
        batch_size=batch_size)

net.save_weights('my_model_weights.h5')