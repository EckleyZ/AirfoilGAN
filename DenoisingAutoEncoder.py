#!/usr/bin/env python
# coding: utf-8

"""
Created on Tue Mar  1 11:43:23 2022

@author: Zachary Eckley

=============== = Autoencoder for GAN Iter. 5 = ===============
Changes:
    - Changed activation of final layer of encoder to a sigmoid function
    - Created new loss function combining mean squared error and a contractive
      loss. Contractive loss comes from the equation below
                  lambda*(||J(x)||_F)^2
         - lambda is a scaling coefficient (usually 100)
         - J(X) is a jacobian matrix of the output after the final layer 
           of the encoder
         - ||f(X)||_F is the frobenius norm
    - The new loss function rewards the model for accurate recreation but also 
      punishes the model for weights that are unaffected by changes in the 
      inputs. This means that the model will emphasize the weights that are 
      REALLY IMPORTANT and minimize the ones that have littel impact on the 
      recreation, which is essentially the goal of the autoencoder. 

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
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Conv3D
from tensorflow.keras.layers import Conv3DTranspose
#from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
#import tensorflow.keras.backend as K

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


#%% Load the Dataset
P_data = np.load("D:\\Non Windows Stuff\\HPC Research\\Airfoil_Performance_Data.npy")

# split into training and testing sets
sets = 24
split = int(P_data.shape[0]/sets)
P_train = np.zeros([(sets-1)*split, P_data.shape[1], P_data.shape[2], P_data.shape[3]])
P_test = np.zeros([split, P_data.shape[1], P_data.shape[2], P_data.shape[3]])
for i in range(0,split):
    P_train[((sets-1)*i):(sets-1)*(i+1),:,:,:] = P_data[(sets*i)+1:sets*(i+1),:,:,:]
    P_test[i,:,:,:] = P_data[sets*i,:,:,:]


P_train = (P_train.astype('float32')).reshape([split*(sets-1),121,201,10,1])
P_test = (P_test.astype('float32')).reshape([split,121,201,10,1])

# add noise to the data
#DropPercent = 30
#P_train_noisy = P_train * (np.random.randint(1,101,P_train.shape) > DropPercent)
#P_test_noisy = P_test * (np.random.randint(1,101,P_test.shape) > DropPercent)

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
    
    # Input layer with noise
    self.encoder.add(InputLayer(input_shape=(121, 201, 10, 1)))
    
    # Convolution 1 ---- (None,121,201,10,1) --> (None,121,100,F[0])
    self.encoder.add(Conv3D(F[0],(1,1,10),(1,1,1),activation='sigmoid',padding='valid'))
    self.encoder.add(Reshape([121,201,F[0]]))
    self.encoder.add(BatchNormalization(axis=-1))
    
    
    # Convolution 2 ---- (None,121,201,F[0]) --> (None,41,34,F[1])
    self.encoder.add(Conv2D(F[1],(5,9),(3,5),activation='tanh',padding='same'))
    self.encoder.add(BatchNormalization(axis=-1))
    
    
    # Convolution 3 ---- (None,41,34,F[1]) --> (None,21,12,F[2])
    self.encoder.add(Conv2D(F[2],(3,3),(2,2),activation='tanh',padding='same'))
    self.encoder.add(BatchNormalization(axis=-1))
    
    
    # Convolution 4 ---- (None,21,12,F[2]) --> (None,10,10,F[3])
    self.encoder.add(Conv2D(F[3],(3,3),(2,2),activation='tanh',padding='valid'))
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
    self.decoder.add(Conv2DTranspose(F[2],(3,3),(2,2),activation='tanh',padding='valid'))
    self.decoder.add(BatchNormalization(axis=-1))
    
    
    # Deconvolution 3 ---- (None,21,12,32) --> (None,41,34,16)
    self.decoder.add(Conv2DTranspose(F[1],(3,3),(2,2),activation='tanh',padding='same',output_padding=(0,0)))
    self.decoder.add(BatchNormalization(axis=-1))
    
    # Deconvolution 4 ---- (None,41,34,16) --> (None,121,201,8)
    self.decoder.add(Conv2DTranspose(F[0],(5,9),(3,5),activation='tanh',padding='same',output_padding=(0,0)))
    self.decoder.add(Reshape([121,201,F[0]]))
    self.decoder.add(BatchNormalization(axis=-1))
    
    # Deconvolution 5 ---- (None,121,100,8) --> (None,121,201,10,1)
    self.decoder.add(Reshape([121,201,1,F[0]]))
    self.decoder.add(Conv3DTranspose(1,(1,1,10),(1,1,1),activation='sigmoid',padding='valid'))
    

  def call(self, x):
      # add noise to x after entering into encoder
      encoded = self.encoder(x)
      decoded = self.decoder(encoded)
      return decoded

AE = Autoencoder(latent_dim)

#%% Define gradient optimizer for model
def grad(model,inputs_noisy,inputs_normal):
    with tf.GradientTape() as tape:
        loss_fn = losses.MeanSquaredError()
        decoded = model(inputs_noisy)
        loss_value = loss_fn(decoded, tf.convert_to_tensor(inputs_normal))
    
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

#%% compile the model, MSE is the standard loss function for autoencoders
AE.compile(optimizer='adam', loss=losses.MeanSquaredError())

AE.encoder.summary()
AE.decoder.summary()


#%% Train the model with x_test as input and the target output

# input noisy data should receive normal data back
#AE.fit(P_train, P_train,epochs=500,shuffle=True,validation_data=(P_test, P_test))

optimizer = tf.optimizers.Adam(learning_rate=0.1)
global_step = tf.Variable(0)
num_epochs = 20
batch_size = 33
Best_Epoch = 10**40
Loss_All = []
for epoch in range(num_epochs):
    for x in range(0, len(P_train), batch_size):
        x_inp = P_train[x : x + batch_size]
        
        #apply noise to input data
        DropPercent = 15
        noise = np.random.randint(1,101,x_inp.shape) > DropPercent
        x_noisy = x_inp * noise
        
        #train model
        loss, grads = grad(AE, x_noisy, x_inp)
        optimizer.apply_gradients(zip(grads, AE.trainable_variables),global_step)
    
    Loss_Epoch = (tf.reduce_sum(loss))
    if Loss_Epoch<Best_Epoch:
        improvement = 'better'
        Best_Epoch = Loss_Epoch
    else:
        improvement = ''
    
    Loss_All.append(Loss_Epoch)
        
    print("Epoch: {0:4d}/{1:4d}\t \t Loss: {2:12.4f}\t {3}".format(epoch+1,num_epochs,tf.reduce_sum(loss),improvement))
    
plt.figure(figsize=(16, 9))
plt.plot(range(1,num_epochs+1),Loss_All)

#%% Test the model by encoding and decoding the test data

encoded_imgs = AE.encoder(P_test).numpy()
decoded_imgs = AE.decoder(encoded_imgs).numpy()

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


