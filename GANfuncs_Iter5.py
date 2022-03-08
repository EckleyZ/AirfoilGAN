# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 15:35:53 2022

@author: Admin

Description:
New file to hold all of the functions for the GAN
Makes main file more easily readable

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
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from tensorflow_addons.layers import SpectralNormalization
import matplotlib.pyplot as plt

#%% Define model function

def define_generator(latent_dim):
    model = Sequential()
    # foundation for 2x50 points
    N = 1024
    n_nodes = N * 2 * 25
    
    # Layer 1 - Dense nodes
    model.add(SpectralNormalization(Dense(n_nodes, input_dim=latent_dim)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((2, 25, N)))
    
    # Layer 2 - Upsample from (2,25) --> (2,50)
    model.add(SpectralNormalization(Conv2DTranspose(N, (2,4), strides=(1,2), padding='same')))
    model.add(LeakyReLU(alpha=0.2))
    
    # Layer 3 - Upsample from (2,50) --> (2,100)
    model.add(SpectralNormalization(Conv2DTranspose(N/4, (2,4), strides=(1,2), padding='same')))
    model.add(LeakyReLU(alpha=0.2))
    
    # Layer 4 - Upsample from (2,100) --> (2,200)
    model.add(SpectralNormalization(Conv2DTranspose(N/16, (2,4), strides=(1,2), padding='same')))
    model.add(LeakyReLU(alpha=0.2))
    
    # Layer 5 - 1 filter looking all the points
    model.add(SpectralNormalization(Conv2D(1, (2,2), padding='same')))  
    model.add(LeakyReLU(alpha=0.2))
    
    model.trainable = True
    
    return model

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
    
    return model

#%% create the combined model
def define_gan(g_model, d_model):
    # make weights in the discriminator untrainable
    d_model.trainable = False
    
    #connect them
    model = Sequential()
    model.add(g_model)
    model.add(d_model)
    
    # compile model
    opt = Adam(learning_rate=0.0001, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    
    return model

#%% randomly select airfoils to be in real and fake dataset
def SplitDataset(AirfoilNum,batchSize,n_epochs,unrolling_steps):
    batchNum = int(AirfoilNum/batchSize)
    halfBatch = int(batchSize/2)
    batch = np.zeros([batchNum+unrolling_steps, batchSize, n_epochs])
    Num = np.array(list(range(0,AirfoilNum)))
    for b in range(0,n_epochs):
        np.random.shuffle(Num)
        AirfoilIDs = hstack((Num, np.random.randint(0,AirfoilNum-1,unrolling_steps*batchSize)))
        batch[:,:,b] = AirfoilIDs.reshape(batchNum+unrolling_steps,batchSize)
        
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
def train(g_model, d_model, gan_model, C_data, P_data, dataNum, unrolling_steps, n_epochs=1000, n_batch=int(16)):
    
    #define batch per epoch and split dataset for all epochs
    bat_per_epo = int(C_data.shape[0]/n_batch)
    reals, fakes = SplitDataset(C_data.shape[0], n_batch, n_epochs, unrolling_steps)
    
    # Create a seed for plotting airfoil after each epoch
    inputData = hstack((P_data[0,:].reshape(1,100),(tf.random.normal([1,100])).numpy()))
    seed = tf.convert_to_tensor(inputData)
    
    #initialize arrays for loss
    d_loss = np.zeros([bat_per_epo,1])  # Loss throughout each epoch
    g_loss = np.zeros([bat_per_epo,1])
    g_lossLog = []                      # Average loss for each epoch
    d_lossLog = []
    #initialize font for plots
    font2 = {'family' : 'Times New Roman','weight' : 'normal','size' : 14,}
    
    for i in range(n_epochs):
        for j in range(bat_per_epo):
            #unrolled training for discriminator
            for k in range(unrolling_steps):
                # create real and fake samples and then combine
                X_real, y_real = generate_real_samples(C_data, reals[j+k,:,i])
                X_fake, y_fake = generate_fake_samples(g_model, P_data, fakes[j+k,:,i], dataNum)
                X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))

                if k==0:# save weights 
                    d_loss[j], _ = d_model.train_on_batch(X, y)
                    d_weights = d_model.get_weights()
                else:
                    loss, _ = d_model.train_on_batch(X, y)
            
            # prepare points in latent space as input for the generator
            batch = np.concatenate((reals[j,:,i],fakes[j,:,i]))
            X_gan = generate_latent_points(P_data, batch, dataNum)
            y_gan = ones((n_batch, 1))  #invert fake sample label
            
            # update the generator via the discriminator's error
            g_loss[j] = gan_model.train_on_batch(X_gan, y_gan)
            
            #reset d_model weights back to one iteration of training
            d_model.set_weights(d_weights)
        
        
        '''
        # USED FOR OBSERVING SPIKES IN LOSS CURVE
        if i>=50:
            # Plot G prediction
            prediction = g_model(seed, training=False)
            plt.figure(figsize=(20,8))
            plt.plot(prediction[0,0,:,0],prediction[0,1,:,0])
            plt.xlim(0,1)
            plt.ylim(0.5,1.5)
            plt.grid(True)
            plt.xlabel('X axis (% chord)',font2)
            plt.ylabel('Y axis (% chord)',font2)
            plt.title('Epoch #'+str(i)+' Prediction',font2)
            plt.show()
            
            # Plot zoomed in version of the loss curve
            plt.figure(figsize=(20,8))
            plt.plot(range(i-50,i),g_lossLog[i-50:i],'r',label='Generator Loss')
            plt.plot(range(i-50,i),d_lossLog[i-50:i],'b',label='Discriminator Loss')
            plt.legend(loc=0)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Loss in Epochs '+str(i-50)+'-'+str(i))
            plt.show()
        
        
        if np.remainder(i+1,50)==1:
            SaveLoc = 'Models\\GANmodel_recent\\Model\\E'+str(i+1)
            g_model.build((None,1,200))
            g_model.save(SaveLoc)
        '''
        
        # summarize loss on this epoch
        if i==0:
            # setup for ouputs
            print('   Epoch  | D loss avg | G loss avg')
            print('----------+------------+------------')
            
        DLavg = np.sum(d_loss)/bat_per_epo
        GLavg = np.sum(g_loss)/bat_per_epo
        print('{0:4d}/{1:4d} |   {2:1.4f}   |   {3:1.4f}'.format(i+1,n_epochs,DLavg,GLavg))
        
        # record loss over epochs
        g_lossLog.append(GLavg)
        d_lossLog.append(DLavg)
    
    # plot the loss curve over epochs
    plt.figure(figsize=(10,4))
    plt.plot(range(0,i+1),g_lossLog,'r',label='Generator Loss')
    plt.plot(range(0,i+1),d_lossLog,'b',label='Discriminator Loss')
    plt.legend(loc=0)
    plt.xlabel('Epochs',font2)
    plt.ylabel('Loss',font2)
    plt.show()
        
