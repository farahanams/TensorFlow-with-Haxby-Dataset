###################
#### Keras with DS109 dataset
####    Deep Autoencoder
###################
#### Tensorflow backend
#### with Theano dimension ordering
###################

## Date: March 2 2017

import numpy as np
from keras import backend
backend.set_image_dim_ordering('th') # theano (channel,rows,cols)

from load_data import LoadDs109

from keras.models import Model 
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
## Maxpooling is for downsampling while upsampling is vice versa
from keras import regularizers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
plt.ion()
####
# Load data
####

data_X, data_Y = LoadDs109()
data_X = normalize(data_X)
print ("The size of Ds109 data and label data are:", data_X.shape, data_Y.shape)

####
# Preprocess data and labels
####
data_X = data_X.reshape(data_X.shape[0], 1, 72, 72).astype('float32')
X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size = 0.2, random_state = 1337)

print ("The size of train and test data are:", X_train.shape, X_test.shape)

####
# Modeling the architecture 
####
#encoding_dim = 100
input_img = Input(shape=(1, 72, 72))

x = Convolution2D(44, 3, 3, activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((2,2), border_mode = 'same')(x)
x = Convolution2D(22, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2,2), border_mode = 'same')(x)
x = Convolution2D(22, 3, 3, activation='relu', border_mode='same')(x)

encoded = MaxPooling2D((2,2), border_mode='same')(x)

x = Convolution2D(22, 3, 3, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2,2))(x)
x = Convolution2D(22, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2,2))(x)
x = Convolution2D(44, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2,2))(x)

decoded = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same')(x)

autoencoder = Model(input_img, decoded)

autoencoder.compile(loss='binary_crossentropy',
                    optimizer='adadelta') 

# Fitting model
autoencoder.fit(X_train, 
            X_train,
            batch_size=100,
            nb_epoch=20,
            shuffle=True,
            validation_data=(X_test, X_test))

# DIsplay the ouputs
#encoded_imgs = encoder.predict(X_test)
decoded_imgs = autoencoder.predict(X_test)

n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i+100].reshape(72,72))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i+100].reshape(72, 72))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('test_ouput.png')

#encoder = Model(input_img, encoded3)
#np.savetxt('decoded_train.csv', autoencoder.predict(X_train), delimiter=' ')
#np.savetxt('label_train.csv', y_train,delimiter=' ')

#np.savetxt('decoded_test.csv', autoencoder.predict(X_test), delimiter=' ')
#np.savetxt('label_test.csv', y_test, delimiter=' ')
