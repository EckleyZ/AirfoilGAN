# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 11:43:23 2022
@author: Zachary Eckley
"""

import numpy as np
import tensorflow as tf
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import ELU
from tensorflow.keras.layers import Dropout
import matplotlib.pyplot as plt
from playsound import playsound

# TensorFlow Testing

#%% Define model function

def define_generator(latent_dim):
    model = Sequential()
    # foundation for 7x7 image
    N = 128
    
    n_nodes = N * 121 * 201 * 10 # N * Re tests * AOA tests * data type
    model.add(Dense(n_nodes,input_dim=latent_dim))
    model.add(Elu(alpha=1))
    model.reshape((121,201,10,N))
    # downsample to 40x200x1
    model.add(Conv3D(N,(4, 2, 10),(3, 1, 1), padding='valid'))
    model.add(Elu(alpha=1))
    #downsample to 10x200x1
    model.add(Conv3D(N,(13, 1, 1),(3, 1, 1), padding='valid'))
    model.add(Elu(alpha=1))
    #downsample to 2x200
    model.add(Conv3D(N,(5, 1, 1),(5, 1, 1), padding='valid'))
    model.add(Elu(alpha=1))
    #model.add(Conv3D(1, (2,50), activation='sigmoid', padding='same'))
      
      #output needs to be 2x200 or something similar
    '''
    n_nodes = N * 2 * 50
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(Elu(alpha=1))
    model.add(Reshape((2, 50, N)))
    # upsample to 2x100
    model.add(Conv2DTranspose(N, (2,4), strides=(1,2), padding='same'))
    model.add(Elu(alpha=0.2))
    # upsample to 2x200
    model.add(Conv2DTranspose(N, (2,4), strides=(1,2), padding='same'))
    model.add(Elu(alpha=0.2))
    model.add(Conv2D(1, (2,50), activation='sigmoid', padding='same'))
    '''
    return model

# converting eager tensor to array 
# Array = EagerTensor.numpy();

#%% Load the dataset
P_Data = np.load("D:\\Non Windows Stuff\\HPC Research\\Airfoil_Performance_Data.npy")
C_Data = np.load("D:\\Non Windows Stuff\\HPC Research\\Airfoil_Coordinate_Data.npy")

#%% Testing the model
s = 3
skip = s*np.array(list(range(0,int(120/s+1))))
Data = PerformanceData[0, skip, :, 0]
DataNum = np.size(Data)
Data = Data.reshape(1,DataNum)
noise = tf.convert_to_tensor(Data)
G = define_generator(DataNum)
Coords = G(noise)
arrayCoords = Coords.numpy()
x = arrayCoords[:,0,:].reshape([200,1])
y = arrayCoords[:,1,:].reshape([200,1])
plt.plot(x,y)
plt.show()
print('Like a toddler scribbling with crayons')
plt.plot(x)
plt.plot(y)
plt.legend(["x","y"])
plt.show()
print("Data in, noise out... This is bad")

#%% Define discriminator
def define_discriminator(in_shape=(1,2,200,1)):
    model = Sequential()
    #down sample to 14x14
    model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
      #down sample to 7x7
    model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)     #dont know what an adam optimizer is
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

#%% create the combined model
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    #connect them
    model = Sequential()
    # add generator
    model.add(g_model)
    # add discriminator
    model.add(d_model)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)   #dont know what an adam optimizer is
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

#%% randomly select airfoils to be in real and fake dataset
def SplitDataset(AirfoilNum,batchSize):
    batchNum = AirfoilNum/batchSize
    halfBatch = batchSize/2
    Num = list(range(0,AirfoilNum))
    batch = np.zeros([1, AirfoilNum])
    b = 0
    for b in range(0,AirfoilNum):
        selected = randint(0, len(Num))
        batch[0,b] = Num[selected]
        Num.pop(selected)
        
    batch = batch.reshape(batchNum,batchSize)
    real = batch[:,0:halfBatch]
    fake = batch[:,halfBatch:batchSize]

    return real, fake

#%% creating a dataset from MNIST images with class label as 1
def generate_real_samples(dataset, realSet):
    # choose random instances
    X = np.zeros([len(realSet), 2, 200])
    for r in range(len(realSet)):
        X[r, :, :] = CoordinateData[realSet[r], :, :]
    
    # generate 'real' class labels (1)
    y = ones((len(realSet), 1))
    return X, y

#%% create input for Generator (generate points in latent space for G)
def generate_latent_points(dataset, fakeSet, dataNum):
    # generate points in the latent space
    x_input = np.zeros([len(fakeSet), dataNum])
    for f in range(len(fakeSet)):
        x_input[f, :, :] = PerformanceData[fakeSet[f], :, :, 0].reshape(1,dataNum)
    
    return x_input

#%% creating fake dataset for discriminator (use G to make fake examples)
def generate_fake_samples(g_model, dataset, fakeSet, dataNum):    
    #generate points in latent space
    x_input = generate_latent_points(dataset, fakeSet, dataNum)
    #predict outputs
    X = g_model.predict(x_input)
    # create 'fake' class labels (0)
    y = zeros((len(fakeSet), 1))
    return X, y

#%% Training method GAN
def train(g_model, d_model, gan_model, CoordinateData, PerformanceData, dataNum, n_epochs=25, n_batch=48):
    bat_per_epo = int(CoordinateData.shape[0]/n_batch)
    #manually enumerate epochs
    for i in range(n_epochs):
        #enumerate batches over the training set
        reals, fakes = SplitDataset(CoordinateData.shape[0], n_batch)
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(CoordinateData, reals[j,:])
            # generate fake samples
            X_fake, y_fake = generate_fake_samples(g_model, PerformanceData, fakes[j,:], dataNum)
            # create training set for the discriminator
            X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
            print(y.size[0])
            # update discriminator model weights
            d_loss, _ = d_model.train_on_batch(X, y)
            #prepare points in latent space as input for the generator
            X_gan = generate_latent_points(PerformanceData, n_batch, dataNum)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))

#%% Train the model
DataNum = int((120/s+1)*201)
D = define_discriminator()          # create the disriminator
G = define_generator(DataNum)       # create the generator
GAN = define_gan(G, D)              # create the gan
dataset = SplitDataset()            # separate data into real and fake
train(G, D, GAN, CoordinateData, PerformanceData)   # train model
playsound('D:\\MyWork\\Independent\\HPC Research\\SalamiLid.wav')

    
#%% Testing the generator after training
noise_dim = 100
num_examples = 10

#Seed will be reused overtime to visualize progress in the animated GIF
seed = tf.random.normal([num_examples, noise_dim])
predictions = G(seed, training=False)

fig = plt.figure(figsize=(4,4))
for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i+1)
    plt.imshow(predictions[i, :, :, 0], cmap='gray')
    plt.axis('off')
    
G.save('\kaggle\output')            
© 2022 GitHub, Inc.
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About
