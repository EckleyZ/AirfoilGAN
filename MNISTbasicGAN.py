# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 20:01:12 2022

@author: Zachary Eckley
"""

#fix something
import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#%% Import packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dropout
from matplotlib import pyplot
from tensorflow.keras.datasets import mnist

#%% define generator
def define_generator(latent_dim):
    model = Sequential()
    # foudnation for 7x7 image
    n_nodes = 128 * 7 * 7
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7, 7, 128)))
    # upsample to 14x14
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    # upsample to 28x28
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
    return model

#%% Testing the generator
noise = tf.random.normal([1, 100])
generator = define_generator(100)
generated_image = generator(noise, training=False)
plt.imshow(generated_image[0, :, :, 0], cmap='gray')

#%% Define discriminator
def define_discriminator(in_shape=(28,28,1)):
    model = Sequential()
    model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
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

#%% Load the data and do some pre=processing
def load_real_samples():
    # load mnist dataset
    (trainX, _), (_, _) = mnist.load_data()
    # expand to 3d, e.g. add channels dimension
    X = expand_dims(trainX, axis=-1)
    # convert from unsigned ints to floats
    X = X.astype('float32')
    # scale from [0,255] to [0,1]
    X = X/255.0
    return X

#%% creating a dataset from MNIST images with class label as 1
def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, 1))
    return X, y

#%% create input for Generator (generate points in latent space for G)
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    x_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    x_input = x_input.reshape(n_samples, latent_dim)
    return x_input

#%% creating fake dataset for discriminator (use G to make fake examples)
def generate_fake_samples(g_model, latent_dim, n_samples):    
    #generate points in latent space
    x_input = generate_latent_points(latent_dim, n_samples)
    #predict outputs
    X = g_model.predict(x_input)
    # create 'fake' class labels (0)
    y = zeros((n_samples, 1))
    return X, y

#%% Training method GAN
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=25, n_batch=256):
    bat_per_epo = int(dataset.shape[0]/n_batch)
    half_batch = int(n_batch/2)
    #manually enumerate epochs
    for i in range(n_epochs):
        #enumerate batches over the training set
        for j in range(bat_per_epo):
            # get randomly selected 'real' samples
            X_real, y_real = generate_real_samples(dataset, half_batch)
            # generate fake samples
            X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            # create training set for the discriminator
            X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
            # update discriminator model weights
            d_loss, _ = d_model.train_on_batch(X, y)
            #prepare points in latent space as input for the generator
            X_gan = generate_latent_points(latent_dim, n_batch)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = gan_model.train_on_batch(X_gan, y_gan)
            # summarize loss on this batch
            print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))
            
            
#%% Train the model
#size of the latent space
latent_dim = 100
# create the disriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
#load image data
dataset = load_real_samples()
# train model
train(g_model, d_model, gan_model, dataset, latent_dim)

    
#%% Testing the generator after training
noise_dim = 100
num_examples = 10

#Seed will be reused overtime to visualize progress in the animated GIF
seed = tf.random.normal([num_examples, noise_dim])
predictions = g_model(seed, training=False)

fig = plt.figure(figsize=(4,4))
for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i+1)
    plt.imshow(predictions[i, :, :, 0], cmap='gray')
    plt.axis('off')
    
g_model.save('\kaggle\output')
    
#%% Load model
gmdl_new = keras.models.load_model('\kaggle\output')
seed = tf.random.normal([num_examples, noise_dim])
predictions = gmdl_new(seed, training=False)

fig = plt.figure(figsize=(4,4))
for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i+1)
    plt.imshow(predictions[i, :, :, 0], cmap='gray')
    plt.axis('off')