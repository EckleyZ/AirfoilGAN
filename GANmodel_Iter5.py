# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 11:43:23 2022

@author: Zachary Eckley

=============== = GANmodel Iteration 5 = ===============
Changes:
    - Changed G layer structure
    - implemented unrolled GAN
    - Moved functions to GANfuncs.py to clean up the file

changes to G layers:
    1) Added an additional layer of upsampling
    2) Hoping that additional layers will help the generator
    3) Final layer has a much smaller filter. This may help with smoothing 
       and reducing the spikiness in small regions
    
Unrolled GAN implementation
    1) Most of the changes were done to the train function in GANfuncs.py

"""

#%% import modules
import numpy as np
import tensorflow as tf
from numpy import hstack
import matplotlib.pyplot as plt

# Load the functions from extrenal file
from GANfuncs import define_generator
from GANfuncs import define_discriminator
from GANfuncs import define_gan
from GANfuncs import train


#%% Load the dataset
P_data = np.load("D:\\Non Windows Stuff\\HPC Research\\Encoded_Airfoil_Performance_Data.npy")
C_data = np.load("D:\\Non Windows Stuff\\HPC Research\\Airfoil_Coordinate_Data.npy")
C_data[:,1,:] = C_data[:,1,:]+1     # shift coords up to allow negative Y


#%% Train the model
DataNum = 200                               # size of input data
AirfoilNum = C_data.shape[0]
batchSz = 16
k = 5                                       # unrolled training steps
D = define_discriminator()                  # create the disriminator
G = define_generator(DataNum)               # create the generator
GAN = define_gan(G, D)                      # create the gan
train(G, D, GAN, C_data, P_data, DataNum, k, n_epochs=100)   # train model

    
#%% Testing the generator after training
noise_dim = 100
num_examples = 8

#Seed will be reused overtime to visualize progress in the animated GIF
randFoilNum = np.random.randint(0,AirfoilNum-1,[num_examples])
randAirfoils = P_data[randFoilNum,:].reshape([num_examples,int(DataNum/2)])
noise = np.zeros([num_examples,noise_dim])
for n in range(num_examples):
    noise[n,:] = tf.random.normal([1,100]).numpy()

testInput = hstack((randAirfoils,noise))
seed = tf.convert_to_tensor(testInput)
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


