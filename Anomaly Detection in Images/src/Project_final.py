# -*- coding: utf-8 -*-


import glob
import numpy as np
import urllib
from urllib.request import urlopen
import tarfile
import os
from PIL import Image
from scipy import signal
from matplotlib import pyplot as plt
from keras import optimizers
from keras import losses
import tensorflow as tf
from keras import layers


tar = tarfile.open("UCSD_Anomaly_Dataset.tar.gz")
tar.extractall()
tar.close()


files = sorted(glob.glob('UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/*/*'))

a = np.zeros((len(files),160,160,1))

for idx,filename in enumerate(files):
    im = Image.open(filename)
    im = im.resize((160,160))
    a[idx,:,:,0] = np.array(im, dtype=np.float32)/255.0



#Model Design
#---------------------------------------------------------------------------------
from keras.layers import Input,Conv2DTranspose, Dense, Conv2D, MaxPooling2D, UpSampling2D,BatchNormalization,Dropout
from keras.models import Model

input_img = Input(shape=(160, 160, 1)) 


x=Conv2D(512, (15, 15),strides=4, activation='relu', padding='same')(input_img)
x=BatchNormalization()(x)
x=MaxPooling2D((2, 2),padding='same')(x)
x=BatchNormalization()(x)


x=Conv2D(256, (4, 4), activation='relu',padding='same')(x)
x=BatchNormalization()(x)
x=MaxPooling2D((2, 2),padding='same')(x)
x=BatchNormalization()(x)


x=Conv2D(128, (3, 3), activation='relu',padding='same')(x)
x=BatchNormalization()(x)




x = Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(x)
x=BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)
x=BatchNormalization()(x)



x = Conv2DTranspose(512, (4, 4), activation='relu', padding='same')(x)
x=BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)
x=BatchNormalization()(x)


decoded = Conv2DTranspose(1, (15, 15),strides=4, activation='sigmoid', padding='same')(x)


model = Model(input_img, decoded)
model.summary()

loss_function=losses.mean_squared_error
rmsprop=optimizers.RMSprop(learning_rate=0.01, rho=0.9)

model.compile(optimizer=rmsprop,loss=loss_function,metrics=['accuracy'])



history=model.fit(a,a,
                  epochs=100,
                  batch_size=20,
                  )

model.save("autoencoder_ucsd_final.h5.h5")




#loading the model
#------------------------------------------------------------------------------
from keras.models import load_model
model_l=load_model("autoencoder_ucsd_final.h5")
#------------------------------------------------------------------------------




#Function to Plot anomaly
#------------------------------------------------------------------------------
def plot(img, output, diff, H, threshold, counter):
    
    fig, (ax0, ax1, ax2,ax3) = plt.subplots(ncols=4, figsize=(10, 5))
    ax0.set_axis_off()
    ax1.set_axis_off()
    ax2.set_axis_off()
    
    ax0.set_title('input image')
    ax1.set_title('reconstructed image')
    ax2.set_title('diff ')
    ax3.set_title('anomalies')
    
    ax0.imshow(img, cmap=plt.cm.gray, interpolation='nearest') 
    ax1.imshow(output, cmap=plt.cm.gray, interpolation='nearest')   
    ax2.imshow(diff, cmap=plt.cm.viridis, vmin=0, vmax=255, interpolation='nearest')  
    ax3.imshow(img, cmap=plt.cm.gray, interpolation='nearest')
    
    x,y = np.where(H > threshold)
    ax3.scatter(y,x,color='red',s=0.1) 

    plt.axis('off')
    
    fig.savefig('images/' + str(counter) + '.png')
#------------------------------------------------------------------------------    
    





#Loading Test Data Set
#-------------------------------------------------------------------------------
files = sorted(glob.glob('UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test020/*'))

b = np.zeros((len(files),160,160,1))

for idx,filename in enumerate(files):
    im = Image.open(filename)
    im = im.resize((160,160))
    b[idx,:,:,0] = np.array(im, dtype=np.float32)/255.0
#-------------------------------------------------------------------------------



#Run this code to view original, reconstructed,diff and anomalies 
#-------------------------------------------------------------------------------
threshold = 3*255
counter = 0
try:
    os.mkdir("images")
except:
    pass

 
for image in b:        
    counter = counter + 1   
    img=np.reshape(image, (1,160,160,1))
    output = model_l.predict(img)
    output.resize(1,160,160,1)
    output*=255.0
    img*=255.0
    diff = np.abs(output-img)    
    tmp = diff[0,:,:,0]
    H = signal.convolve2d(tmp, np.ones((4,4)), mode='same') 
    plot(img[0,:,:,0], output[0,:,:,0], diff[0,:,:,0], H, threshold, counter)
#-------------------------------------------------------------------------------
