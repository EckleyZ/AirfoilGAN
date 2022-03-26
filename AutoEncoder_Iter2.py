#!/usr/bin/env python
# coding: utf-8

"""
Created on Tue Mar  22 3:06:00 2022

@author: Zachary Eckley

=============== = Autoencoder for GAN Iter. 5.5 = ===============
Changes:
1. Contractive autoencoder produced poor results so the contractive loss 
   function was removed.
2. Denoising autoencoder produced poor results so process of adding noise to 
   the data was removed.
3. Changed the way the data was being fed to the autoencoder.
4. Previously the entire dataset for each airfoil was being fed to the AE all 
   at once. In reality, an engineer shouldn't need an airfoil to operate at 
   specific conditions for such a large range of Re. Instead of giving the AE 
   a huge chunk to replicate the dataset will be broken into 10 ranges of Re 
   that will be used to train the AE.
5. Each chunk spans 22 Re and contains 7 different properties. The original 
   dataset included 10 properties but 3 were removed for reasons explained 
   below. The shape of the chunks are 22x202x7. An additional column was added 
   to the beginning of the dataset to include the reynolds number used to find 
   the data in the row. The Reynolds number was in tens of millions to avoid 
   problems with data magnitude.
6. 3 properties were removed from the dataset, the lift to drag ratio (LD), 
   aerodynamic center (AC) and center of pressure (CP) were removed due to 
   frequent outliers and much larger values than the other properties. The 
   lift to drag ratio can be estimated by looking at the coefficient of lift 
   and coefficient of drag, and the aerodynamic center and center of pressure 
   can be looked into further after preliminary design.
      
"""

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
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import Conv3DTranspose
# from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
#import tensorflow.keras.backend as K

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


#%% Load the Dataset
P_data = np.load("D:\\Non Windows Stuff\\HPC Research\\Airfoil_Performance_Data.npy")
#normalize LD ratio
#P_data[:,:,:,7] = P_data[:,:,:,7]/100
P_data = P_data[:,:,:,0:7]

# Reorganize data set into chunks rather than huge 10 layer sheets
rows = 22
cNum = 10
NewData = np.zeros((1584*cNum,rows,202,7))
ReNum = np.linspace(np.zeros((1,1,7)),2.1*np.ones((1,1,7)),22,axis=1).reshape((rows,1,7),order='F')
DataTypes = list(range(1,7))
for a in range(1584):
    for c in range(0,10):
        ind = c*11
        Chunk = P_data[a,ind:ind+rows,:,:]
        ReCol = ReNum*0.15+0.1+(c*1.1*0.15) #in tens of millions 
        NewData[cNum*a+c,:,:] = np.hstack((ReCol,Chunk))

NewData = NewData.reshape(NewData.shape[0],NewData.shape[1],NewData.shape[2],NewData.shape[3],1)

del P_data

IndexNums = list(range(NewData.shape[0]))
np.random.shuffle(IndexNums)
TrainData = NewData[IndexNums[0:1584*7]]
TestData = NewData[IndexNums[1584*7:1584*10]]
del NewData

#%% autoencoder
latent_dim = 64 

class Autoencoder(Model):
  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    F = [32, 64, 128, 256, 512]
    #F = F[::-1]
    
    #================ ENCODER ================
    self.encoder = Sequential()
    
    # Input layer
    self.encoder.add(InputLayer(input_shape=(22,202,7,1)))
    
    # ~~~~~ Convolution 1 ~~~~~
    # (None,110,202) --> (None,107,199)
    self.encoder.add(Conv3D(F[0],(1,4,1),(1,2,1),activation='sigmoid',padding='valid'))
    #self.encoder.add(LeakyReLU(alpha=0.2))
    self.encoder.add(BatchNormalization(axis=-1))
    
    # ~~~~~ Convolution 2 ~~~~~
    # Trim --- (None,105,99) ---> (None,103,97)
    self.encoder.add(Conv3D(F[1],(1,3,1),(1,2,1),activation='sigmoid',padding='same'))
    #self.encoder.add(LeakyReLU(alpha=0.2))
    self.encoder.add(BatchNormalization(axis=-1))
    
    # ~~~~~ Convolution 3 ~~~~~
    # Trim --- (None,51,48) ----> (None,49,47)
    self.encoder.add(Conv3D(F[2],(1,4,1),(1,2,1),activation='sigmoid',padding='valid'))
    #self.encoder.add(LeakyReLU(alpha=0.2))
    self.encoder.add(BatchNormalization(axis=-1))
    
    # ~~~~~ Convolution 4 ~~~~~
    # Trim --- (None,24,23) ----> (None,22,22)
    self.encoder.add(Conv3D(F[3],(4,6,1),(2,2,1),activation='sigmoid',padding='valid'))
    #self.encoder.add(LeakyReLU(alpha=0.2))
    self.encoder.add(BatchNormalization(axis=-1))
    
    # ~~~~~ Convolution 5 ~~~~~
    # Trim --- (None,24,23) ----> (None,22,22)
    self.encoder.add(Conv3D(F[4],(1,1,7),(1,1,1),activation='sigmoid',padding='valid'))
    #self.encoder.add(LeakyReLU(alpha=0.2))
    self.encoder.add(BatchNormalization(axis=-1))
    
    
    # Prepare output
    self.encoder.add(Flatten())
    outShape = 10*10
    self.encoder.add(Dense(outShape))
    
    #================ DECODER ================
    self.decoder = Sequential()
    
    # Input layer
    self.decoder.add(InputLayer(input_shape=(100,)))
    self.decoder.add(Dense(F[4]*outShape))
    self.decoder.add(Reshape([10,10,1,F[4]]))
    
    # ~~~~~ Deconvolution 1 ~~~~~
    # Upsample --- (None,10,10) ---> (None,22,22)
    self.decoder.add(Conv3DTranspose(F[3],(1,1,7),(1,1,1),activation='sigmoid',padding='valid'))
    #self.decoder.add(LeakyReLU(alpha=0.2))
    self.decoder.add(BatchNormalization(axis=-1))
    
    # ~~~~~ Deconvolution 2 ~~~~~
    # Upsample --- (None,24,23) ---> (None,49,47)
    self.decoder.add(Conv3DTranspose(F[2],(4,6,1),(2,2,1),activation='sigmoid',padding='valid'))
    #self.decoder.add(LeakyReLU(alpha=0.2))
    self.decoder.add(BatchNormalization(axis=-1))
    
    # ~~~~~ Deconvolution 3 ~~~~~
    # Upsample --- (None,51,48) ---> (None,103,97)
    self.decoder.add(Conv3DTranspose(F[1],(1,4,1),(1,2,1),activation='sigmoid',padding='valid'))
    #self.decoder.add(LeakyReLU(alpha=0.2))
    self.decoder.add(BatchNormalization(axis=-1))
    
    # ~~~~~ Deconvolution 4 ~~~~~
    # Upsample --- (None,105,99) ---> (None,107,199)
    self.decoder.add(Conv3DTranspose(F[0],(1,3,1),(1,2,1),activation='sigmoid',padding='same',output_padding=(0,1,0)))
    #self.decoder.add(LeakyReLU(alpha=0.2))
    self.decoder.add(BatchNormalization(axis=-1))
    
    # ~~~~~ Deconvolution 5 ~~~~~
    # Upsample --- (None,105,99) ---> (None,107,199)
    self.decoder.add(Conv3DTranspose(1,(1,4,1),(1,2,1),activation='sigmoid',padding='valid'))
    #self.decoder.add(LeakyReLU(alpha=0.2))
    self.decoder.add(BatchNormalization(axis=-1))
    

  def call(self, x):
      encoded = self.encoder(x)
      decoded = self.decoder(encoded)
      return decoded

# compile the model, MSE is the standard loss function for autoencoders
AE = Autoencoder(latent_dim)
AE.compile(optimizer='adam', loss=losses.MeanSquaredError())
AE.encoder.summary()
AE.decoder.summary()

#%% Train the model

AE.fit(TrainData, TrainData, epochs=500, shuffle=True, validation_data=(TestData, TestData))


#%% Test the model by encoding and decoding the test data

encoded_imgs = AE.encoder(TestData[0:20]).numpy()
decoded_imgs = AE.decoder(encoded_imgs).numpy()

print(decoded_imgs.shape)


#%% Plot them (not really images but they should look similar) (only used CL, CD, and CM as R, G, and B

n = 10
plt.figure(figsize=(30, 25))
r = 10
c = 2
for i in range(n):
  # display original
  ax = plt.subplot(r, c, 2*i + 1)
  plt.imshow(TestData[i,:,:,0:3].reshape(22,202,3))
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(r, c, 2 * (i + 1) )
  plt.imshow(decoded_imgs[i,:,:,0:3].reshape(22,202,3))
  plt.title("reconstructed")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()

#%% Save model 
model_Name = 'Models\\AEmodel_recent\\Model'    # Change name depending on model

#autoencoder.encoder.save_weights(model_Name)
AE.save(model_Name)
      
    
#%% Load Model and predict some data
model_Name = 'Models\\AE_5000E_16to1024N\\Model' #'Models\\AEmodel_recent\\Model'

AE = tf.keras.models.load_model(model_Name)

#%% Use Model to convert data set to input 
P_data = P_data.reshape([1584,121,201,10,1])
P_data_encoded = np.zeros([1584,100])
bn = 66
for s in range(int(P_data.shape[0]/bn)):
    Chunk = P_data[(s*bn):((s+1)*bn), :, :, :, :]
    P_data_encoded[(s*bn):((s+1)*bn),:] = autoencoder.encoder(Chunk).numpy()

np.save('Encoded_Airfoil_Performance_Data.npy',P_data_encoded)
