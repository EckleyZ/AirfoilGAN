# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 11:43:23 2022

@author: Zachary Eckley
"""

import numpy as np
import tensorflow as tf
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy import hstack
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow_addons.layers import SpectralNormalization
import matplotlib.pyplot as plt

# TensorFlow Testing

#%% Define model function

def define_generator(latent_dim):
    model = Sequential()
    # foundation for 2x50 points
    N = 128
    n_nodes = N * 2 * 50
    
    model.add(SpectralNormalization(Dense(n_nodes, input_dim=latent_dim)))
    model.add(LeakyReLU(alpha=0.2))
    #model.add(BatchNormalization(axis=-1))
    model.add(Reshape((2, 50, N)))
    
    # upsample to 2x100
    model.add(SpectralNormalization(Conv2DTranspose(N/2, (2,4), strides=(1,2), padding='same')))
    model.add(LeakyReLU(alpha=0.2))
    #model.add(BatchNormalization(axis=-1))
    
    # upsample to 2x200
    model.add(SpectralNormalization(Conv2DTranspose(N/4, (2,4), strides=(1,2), padding='same')))
    model.add(LeakyReLU(alpha=0.2))
    #model.add(BatchNormalization(axis=-1))
    model.add(SpectralNormalization(Conv2D(1, (2,50), activation='relu', padding='same')))  
    
    model.trainable = True
    
    return model

#%% Load the dataset
P_data = np.load("D:\\Non Windows Stuff\\HPC Research\\Encoded_Airfoil_Performance_Data.npy")
C_data = np.load("D:\\Non Windows Stuff\\HPC Research\\Airfoil_Coordinate_Data.npy")
C_data[:,1,:] = C_data[:,1,:]+1

#%% Testing the model
DataNum = int(P_data.shape[1])+100
G = define_generator(DataNum)
inputData = hstack((P_data[0,:].reshape(1,100),(tf.random.normal([1,100])).numpy()))
Coords = G(inputData, training=False)
arrayCoords = Coords.numpy()
x = arrayCoords[:,0,:].reshape([200,1])
y = arrayCoords[:,1,:].reshape([200,1])
plt.scatter(x,y)
plt.show()
plt.scatter(np.linspace(0,199,200),x)
plt.scatter(np.linspace(0,199,200),y)
plt.legend(["x","y"])
plt.show()

#%% Define discriminator
def define_discriminator(in_shape=(2,200,1)):
    model = Sequential()
    N = 256
    model.add(Conv2D(N, (2,4), strides=(2, 2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Conv2D(N, (2,4), strides=(2, 2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    # compile model
    opt = Adam(learning_rate=0.0001, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    #model.trainable = True
    
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
    #opt = Adam(learning_rate=0.0002, beta_1=0.5)
    opt = Adam(learning_rate=0.0001, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

#%% randomly select airfoils to be in real and fake dataset
def SplitDataset(AirfoilNum,batchSize,n_epochs):
    batchNum = int(AirfoilNum/batchSize)
    halfBatch = int(batchSize/2)
    batch = np.zeros([batchNum, batchSize, n_epochs])
    Num = np.array(list(range(0,AirfoilNum)))
    for b in range(0,n_epochs):
        np.random.shuffle(Num)
        batch[:,:,b] = Num.reshape(batchNum,batchSize)
        
        
    real = batch[:,0:halfBatch,:]
    fake = batch[:,halfBatch:batchSize,:]

    return real.astype(int), fake.astype(int)

#%% creating a dataset from MNIST images with class label as 1
def generate_real_samples(dataset, realSet):
    
    # choose random instances
    X = np.zeros([len(realSet), 2, 200,1])
    for r in range(len(realSet)):
        X[r, :, :,:] = dataset[realSet[r], :, :].reshape([1, 2, 200, 1])
    
    # generate 'real' class labels (1)
    y = ones((len(realSet), 1))
    return X, y

#%% create input for Generator (generate points in latent space for G)
def generate_latent_points(dataset, fakeSet, dataNum):
    
    # generate points in the latent space
    x_input = np.zeros([len(fakeSet), dataNum])
    for f in range(len(fakeSet)):
        dataPoints = dataset[fakeSet[f], :]
        dataPoints = dataPoints.reshape(1,dataPoints.size)
        noise = (tf.random.normal([1,100])).numpy()
        x_input[f,:] = hstack((dataPoints,noise))
    
    return x_input

#%% creating fake dataset for discriminator (use G to make fake examples)
def generate_fake_samples(g_model, dataset, fakeSet, dataNum):    
    #generate points in latent space
    x_input = generate_latent_points(dataset, fakeSet, dataNum)
    
    #predict outputs
    X = g_model.predict(x_input)
    half_batch = int(len(fakeSet))
    
    # create 'fake' class labels (0)
    y = zeros((len(fakeSet), 1))
    return X.reshape([half_batch,2,200,1]), y

#%% Training method GAN
def train(g_model, d_model, gan_model, C_data, P_data, dataNum, n_epochs=1000, n_batch=int(16)):
    
    #define batch per epoch and split dataset for all epochs
    bat_per_epo = int(C_data.shape[0]/n_batch)
    reals, fakes = SplitDataset(C_data.shape[0], n_batch, n_epochs)
    
    # Create a seed for plotting airfoil after each epoch
    inputData = hstack((P_data[0,:].reshape(1,100),(tf.random.normal([1,100])).numpy()))
    seed = tf.convert_to_tensor(inputData)
    
    #initialize arrays for loss
    d_loss = np.zeros([bat_per_epo,1])  # Loss throughout each epoch
    g_loss = np.zeros([bat_per_epo,1])
    g_lossLog = []                      # Average loss for each epoch
    d_lossLog = []
    
    # setup for ouputs
    print('   Epoch  | D loss avg | G loss avg')
    print('----------+------------+------------')
    
    for i in range(n_epochs):
        #enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(C_data, reals[j,:,i])
            # generate fake samples
            X_fake, y_fake = generate_fake_samples(g_model, P_data, fakes[j,:,i], dataNum)
            # create training set for the discriminator
            X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
            # update discriminator model weights
            d_loss[j], _ = d_model.train_on_batch(X, y)
            #prepare points in latent space as input for the generator
            batch = np.concatenate((reals[j,:,i],fakes[j,:,i]))
            X_gan = generate_latent_points(P_data, batch, dataNum)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss[j] = gan_model.train_on_batch(X_gan, y_gan)
        
        
        # Plot G prediction
        prediction = G(seed, training=False)
        fig = plt.figure(figsize=(20,8))
        plt.plot(prediction[0,0,:,0],prediction[0,1,:,0])
        plt.xlim(0,1)
        plt.ylim(0.5,1.5)
        plt.grid(True)
        plt.show()
        
        if np.remainder(i+1,50)==1:
            SaveLoc = 'Models\\GANmodel_recent\\Model\\E'+str(i+1)
            G.save(SaveLoc)
        
        # summarize loss on this epoch
        DLavg = np.sum(d_loss)/bat_per_epo
        GLavg = np.sum(g_loss)/bat_per_epo
        print('{0:4d}/{1:4d} |   {2:1.4f}   |   {3:1.4f}'.format(i+1,n_epochs,DLavg,GLavg))
        
        # record loss over epochs
        g_lossLog.append(GLavg)
        d_lossLog.append(DLavg)
    
    # plot the loss curve over epochs
    plt.figure(figsize=(20,8))
    plt.plot(range(0,i+1),g_lossLog,'r',label='Generator Loss')
    plt.plot(range(0,i+1),d_lossLog,'b',label='Discriminator Loss')
    plt.legend(loc=0)
    plt.show()
        

#%% Train the model
AirfoilNum = C_data.shape[0]
batchSz = 16
D = define_discriminator()                          # create the discriminator
G = define_generator(DataNum)                       # create the generator
GAN = define_gan(G, D)                              # create the gan
train(G, D, GAN, C_data, P_data, DataNum)           # train model

    
#%% Testing the generator after training
noise_dim = DataNum #100
num_examples = 8

#Seed will be reused overtime to visualize progress in the animated GIF
randFoilNum = np.random.randint(0,1583,[num_examples])
randAirfoils = P_data[randFoilNum,:].reshape([num_examples,DataNum])
seed = tf.convert_to_tensor(randAirfoils)
predictions = G(seed, training=False)
predictions = predictions.numpy()

# Plot predictions
fig = plt.figure(figsize=(20,10))
for i in range(predictions.shape[0]): 
    plt.subplot(4, 2, i+1)
    plt.plot(predictions[i, 0, :],predictions[i, 1, :]-1)
    plt.xlim(0,1)
    plt.ylim(-0.4,0.4)
    plt.grid(True)
    plt.title('Airfoil #'+str(randFoilNum[i]))
    
#%% Save model 
GAN_Name = 'Models\\GANmodel_recent\\Model'    # Change name depending on model

G.save(GAN_Name)
