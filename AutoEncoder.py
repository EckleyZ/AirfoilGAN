#!/usr/bin/env python
# coding: utf-8

#%% Import TensorFlow and other libraries
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

#from sklearn.metrics import accuracy_score, precision_score, recall_score
#from sklearn.model_selection import train_test_split
from tensorflow.keras import losses
# models used
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
# layers used in models
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import Conv3DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


#%% Load the Dataset
P_data = np.load("D:\\Non Windows Stuff\\HPC Research\\Airfoil_Performance_Data.npy")

sets = 24
split = int(P_data.shape[0]/sets)
P_train = np.zeros([(sets-1)*split, P_data.shape[1], P_data.shape[2], P_data.shape[3]])
P_test = np.zeros([split, P_data.shape[1], P_data.shape[2], P_data.shape[3]])
for i in range(0,split):
    P_train[((sets-1)*i):(sets-1)*(i+1),:,:,:] = P_data[(sets*i)+1:sets*(i+1),:,:,:]
    P_test[i,:,:,:] = P_data[sets*i,:,:,:]


P_train = (P_train.astype('float32')).reshape([split*(sets-1),121,201,10,1])
P_test = (P_test.astype('float32')).reshape([split,121,201,10,1])

print (P_train.shape)
print (P_test.shape)


#%% autoencoder
latent_dim = 64 

class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    F = [16, 64, 256, 1024]
    #F = F[::-1]
    
    #================ ENCODER ================
    self.encoder = Sequential()
    
    # Input layer
    self.encoder.add(InputLayer(input_shape=(121, 201, 10, 1)))
    
    # Convolution 1 ---- (None,121,201,10,1) --> (None,121,100,8)
    self.encoder.add(Conv3D(F[0],(1,1,10),(1,1,1),padding='valid'))
    self.encoder.add(Reshape([121,201,F[0]]))
    self.encoder.add(LeakyReLU(alpha=0.2))
    self.encoder.add(BatchNormalization(axis=-1))
    
    
    # Convolution 2 ---- (None,121,201,8) --> (None,41,34,16)
    self.encoder.add(Conv2D(F[1],(5,9),(3,5),padding='same'))
    self.encoder.add(LeakyReLU(alpha=0.2))
    self.encoder.add(BatchNormalization(axis=-1))
    
    
    # Convolution 3 ---- (None,41,34,16) --> (None,21,12,32)
    self.encoder.add(Conv2D(F[2],(3,3),(2,2),padding='same'))
    self.encoder.add(LeakyReLU(alpha=0.2))
    self.encoder.add(BatchNormalization(axis=-1))
    
    
    # Convolution 4 ---- (None,21,12,32) --> (None,10,10,64)
    self.encoder.add(Conv2D(F[3],(3,3),(2,2),padding='valid'))
    self.encoder.add(LeakyReLU(alpha=0.2))
    self.encoder.add(BatchNormalization(axis=-1))
    
    
    # Prepare output
    self.encoder.add(Flatten())
    outShape = 10*10
    self.encoder.add(Dense(outShape))
    
    #================ DECODER ================
    self.decoder = Sequential()
    
    # Input layer
    self.decoder.add(InputLayer(input_shape=(100,)))
    self.decoder.add(Dense(F[3]*outShape))
    self.decoder.add(Reshape([10,10,F[3]]))
    
    # Deconvolution 2 ---- (None,10,10,64) --> (None,21,12,32)
    self.decoder.add(Conv2DTranspose(F[2],(3,3),(2,2),padding='valid'))
    self.decoder.add(LeakyReLU(alpha=0.2))
    self.decoder.add(BatchNormalization(axis=-1))
    
    
    # Deconvolution 3 ---- (None,21,12,32) --> (None,41,34,16)
    self.decoder.add(Conv2DTranspose(F[1],(3,3),(2,2),padding='same',output_padding=(0,0)))
    self.decoder.add(LeakyReLU(alpha=0.2))
    self.decoder.add(BatchNormalization(axis=-1))
    
    # Deconvolution 4 ---- (None,41,34,16) --> (None,121,201,8)
    self.decoder.add(Conv2DTranspose(F[0],(5,9),(3,5),padding='same',output_padding=(0,0)))
    self.decoder.add(Reshape([121,201,F[0]]))
    self.decoder.add(LeakyReLU(alpha=0.2))
    self.decoder.add(BatchNormalization(axis=-1))
    
    # Deconvolution 5 ---- (None,121,100,8) --> (None,121,201,10,1)
    self.decoder.add(Reshape([121,201,1,F[0]]))
    self.decoder.add(Conv3DTranspose(1,(1,1,10),(1,1,1),activation='sigmoid',padding='valid'))
    

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = Autoencoder(latent_dim)

#%% compile the model, MSE is the standard loss function for autoencoders
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

autoencoder.encoder.summary()
autoencoder.decoder.summary()


#%% Train the model with x_test as input and the target output

# input P_train should receive P_train back
autoencoder.fit(P_train, P_train,
                epochs=500,
                shuffle=True,
                validation_data=(P_test, P_test)) # validate with P_test when done


#%% Test the model by encoding and decoding the test data

encoded_imgs = autoencoder.encoder(P_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

print(decoded_imgs.shape)


#%% Plot them (not really images but they should look similar) (only used CL, CD, and CM as R, G, and B

n = 66
plt.figure(figsize=(64, 32))
r = 11
c = 12
for i in range(n):
  # display original
  ax = plt.subplot(r, c, 2*i + 1)
  plt.imshow(P_test[i,:,:,0:3,0])
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(r, c, 2 * (i + 1) )
  plt.imshow(decoded_imgs[i,:,:,0:3,0])
  plt.title("reconstructed")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()

#%% Save model 
model_Name = 'Models\\AEmodel_recent\\Model'    # Change name depending on model

#autoencoder.encoder.save_weights(model_Name)
autoencoder.save(model_Name)
      
    
#%% Load Model and predict some data
model_Name = 'Models\\AE_500E_16to1024N' #'Models\\AEmodel_recent\\Model'

NewModel = tf.keras.models.load_model(model_Name)

#%% Use Model to convert data set to input 
P_data = P_data.reshape([1584,121,201,10,1])
P_data_encoded = np.zeros([1584,100])
bn = 66
for s in range(int(P_data.shape[0]/bn)):
    Chunk = P_data[(s*bn):((s+1)*bn), :, :, :, :]
    P_data_encoded[(s*bn):((s+1)*bn),:] = autoencoder.encoder(Chunk).numpy()

np.save('Encoded_Airfoil_Performance_Data.npy',P_data_encoded)


