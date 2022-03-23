#!/usr/bin/env python
# coding: utf-8

"""
Created on Tue Mar  22 3:06:00 2022

@author: Zachary Eckley

=============== = Autoencoder for GAN Iter. 5.5 = ===============
Changes:
    - Changed the way the data was being fed to the autoencoder
    - Uses combination of mean squared error and contractive loss. 
      Contractive loss comes from the equation below
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
    - Previously the entire dataset for each airfoil was being fed to the AE 
      all at once. In reality, an engineer shouldn't need an airfoil to operate
      at specific conditions for such a large range of Re. Instead of giving 
      the AE a huge chunk to replicate the dataset will be broken into 11 
      ranges of Re that will be used to train the AE.

"""


#%% Import TensorFlow and other libraries
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

#from sklearn.metrics import accuracy_score, precision_score, recall_score
#from sklearn.model_selection import train_test_split
#from tensorflow.keras import losses
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
#from tensorflow.keras.layers import Sigmoid
from tensorflow.keras.layers import BatchNormalization
#import tensorflow.keras.backend as K

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


#%% Load the Dataset
P_data = np.load("D:\\Non Windows Stuff\\HPC Research\\Airfoil_Performance_Data.npy")

sets = 24
chunkSize = 11
split = int(P_data.shape[0]/sets)
TrainingSets = (sets-1)*split*chunkSize
TestSets = split*chunkSize
P_train = np.zeros([TrainingSets, P_data.shape[1], P_data.shape[2], P_data.shape[3]])
P_test = np.zeros([TestSets, P_data.shape[1], P_data.shape[2], P_data.shape[3]])
for i in range(0,split):
    P_train[((sets-1)*i):(sets-1)*(i+1),:,:,:] = P_data[(sets*i)+1:sets*(i+1),:,:,:]
    P_test[i,:,:,:] = P_data[sets*i,:,:,:]

P_train = (P_train.astype('float32')).reshape([split*(sets-1),121,201,10,1])
P_test = (P_test.astype('float32')).reshape([split,121,201,10,1])

print (P_train.shape)
print (P_test.shape)

num = np.array(list(range(1584))).reshape(int(1584/sets),sets)
num = num[:,0:sets-1].reshape(num.size,1)


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
    
    # Convolution 1 ---- (None,121,201,10,1) --> (None,121,100,F[0])
    self.encoder.add(Conv3D(F[0],(1,1,10),(1,1,1),activation='sigmoid',padding='valid'))
    self.encoder.add(Reshape([121,201,F[0]]))
    #self.encoder.add(LeakyReLU(alpha=0.2))
    self.encoder.add(BatchNormalization(axis=-1))
    
    
    # Convolution 2 ---- (None,121,201,F[0]) --> (None,41,34,F[1])
    self.encoder.add(Conv2D(F[1],(5,9),(3,5),activation='sigmoid',padding='same'))
    #self.encoder.add(LeakyReLU(alpha=0.2))
    self.encoder.add(BatchNormalization(axis=-1))
    
    
    # Convolution 3 ---- (None,41,34,F[1]) --> (None,21,12,F[2])
    self.encoder.add(Conv2D(F[2],(3,3),(2,2),activation='sigmoid',padding='same'))
    #self.encoder.add(LeakyReLU(alpha=0.2))
    self.encoder.add(BatchNormalization(axis=-1))
    
    
    # Convolution 4 ---- (None,21,12,F[2]) --> (None,10,10,F[3])
    self.encoder.add(Conv2D(F[3],(3,3),(2,2),activation='sigmoid',padding='valid'))
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
    self.decoder.add(Conv2DTranspose(F[2],(3,3),(2,2),activation='sigmoid',padding='valid'))
    #self.decoder.add(LeakyReLU(alpha=0.2))
    self.decoder.add(BatchNormalization(axis=-1))
    
    
    # Deconvolution 3 ---- (None,21,12,32) --> (None,41,34,16)
    self.decoder.add(Conv2DTranspose(F[1],(3,3),(2,2),activation='sigmoid',padding='same',output_padding=(0,0)))
    #self.decoder.add(LeakyReLU(alpha=0.2))
    self.decoder.add(BatchNormalization(axis=-1))
    
    # Deconvolution 4 ---- (None,41,34,16) --> (None,121,201,8)
    self.decoder.add(Conv2DTranspose(F[0],(5,9),(3,5),activation='sigmoid',padding='same',output_padding=(0,0)))
    self.decoder.add(Reshape([121,201,F[0]]))
    #self.decoder.add(LeakyReLU(alpha=0.2))
    self.decoder.add(BatchNormalization(axis=-1))
    
    # Deconvolution 5 ---- (None,121,100,8) --> (None,121,201,10,1)
    self.decoder.add(Reshape([121,201,1,F[0]]))
    self.decoder.add(Conv3DTranspose(1,(1,1,10),(1,1,1),activation='sigmoid',padding='valid'))
    

  def call(self, x):
      #x_flat = (x.numpy()).flatten()
      encoded = self.encoder(x)
      decoded = self.decoder(encoded)
      return decoded, encoded

AE = Autoencoder(latent_dim)

#%% Define loss function and gradient function

def Contractive_loss(X_pred, X_true, hidden, model):
    # normal loss from MSE
    DataSize = 243210 # (121,201,10)
    MeanSquaredError = tf.reduce_mean(tf.keras.losses.mse(X_true,X_pred))
    #MeanSquaredError *= DataSize
    #print('MeanSqrErr = '+str(tf.reduce_sum(MeanSquaredError).numpy()))
    
    # Contracitve loss from frobenius norm of the jacobian
    Lambda = 10
    Weights = tf.Variable(model.encoder.layers[-1].weights[0])
    dh = hidden*(1-hidden)
    Weights = tf.transpose(Weights)
    Jacobian = tf.linalg.matmul(dh**2 ,tf.square(Weights))
    Frobenius = tf.reduce_sum(Jacobian, axis=1)
    Contractive = Lambda*Frobenius/DataSize
    #print('Contractive = '+str(tf.reduce_sum(Contractive).numpy()))
       
    return Contractive + MeanSquaredError

def grad(model,inputs):
    with tf.GradientTape() as tape:
        decoded, encoded = model(inputs)
        loss_value = Contractive_loss(decoded, inputs, encoded, model)
    
    return loss_value, tape.gradient(loss_value, model.trainable_variables), decoded


#%% compile the model, MSE is the standard loss function for autoencoders
#AE.compile(optimizer='adam', loss=losses.MeanSquaredError())
#AE.compile(optimizer='adam', loss=Contractive_loss)

#AE.encoder.summary()
#AE.decoder.summary()


#%% Train the model with x_test as input and the target output

''' 
# Reorganize data set into chunks rather than huge 10 layer sheets
NewData = np.zeros((110,202,1584*11))
ReNum = np.linspace(np.zeros((1,10)),np.ones((1,10)),11,axis=1).reshape((110,1),order='F')
DataTypes = list(range(1,10))
for a in range(1584):
    for c in range(0,11):
        ind = c*11
        Chunk = Data[a,ind:ind+11,:,:]
        Stacked = vstack((np.split(Chunk,DataTypes,axis=2))).reshape(110,201)
        ReCol = ReNum*1.5e6+1e6+(c*1.5e6)
        NewData[:,:,11*a+c] = hstack((ReCol,Stacked))
'''

optimizer = tf.optimizers.Adam(learning_rate=0.1)
global_step = tf.Variable(0)
num_epochs = 100
batch_size = 33
Best_Epoch = 10**40
Loss_All = []
for epoch in range(num_epochs):
    np.random.shuffle(num)
    for x in range(0, len(P_train)*11, batch_size):
        
        x_inp = makeInput(num[:,x])
        loss, grads, decoded = grad(AE, x_inp)
        optimizer.apply_gradients(zip(grads, AE.trainable_variables),
                              global_step)
    
    Loss_Epoch = (tf.reduce_sum(loss))/batch_size
    if Loss_Epoch>Best_Epoch:
        improvement = '        worse'
    else:
        improvement = 'better'
        Best_Epoch = Loss_Epoch
    
    Loss_All.append(Loss_Epoch)
        
    print("Epoch: {0:4d}/{1:4d}\t \t Loss: {2:12.4f}\t {3}".format(epoch,num_epochs,tf.reduce_sum(loss),improvement))
    
plt.figure(figsize=(32, 18))
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


