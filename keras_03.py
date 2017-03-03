###################
#### Keras with DS109 dataset
####    Autoencoder
###################
#### Tensorflow backend
#### with Theano dimension ordering
###################

## Date: Feb 27 2017
import numpy as np
from keras import backend
backend.set_image_dim_ordering('th') # theano (channel,rows,cols)

from load_data import LoadDs109

from keras.models import Model 
from keras.layers import Input, Dense

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
#data_X = data_X.reshape(data_X.shape[0], 1, 64, 64)
data_X = data_X.astype('float32')
X_train, X_test, y_train, y_test = train_test_split(data_X, data_Y, test_size = 0.2, random_state = 1337)

print ("The size of train and test data are:", X_train.shape, X_test.shape)

####
# Modeling the architecture 
####
encoding_dim = 100
input_img = Input(shape=(5184,))

encoded = Dense(2592, activation='relu')(input_img)
encoded1 = Dense(1296, activation='relu')(encoded)
encoded2 = Dense(648, activation='relu')(encoded1)
encoded3 = Dense(324, activation='relu')(encoded2)
encoded4 = Dense(169, activation='relu')(encoded3)
encoded5 = Dense(144, activation='relu')(encoded4)
encoded6 = Dense(encoding_dim, activation='relu')(encoded5)

decoded1 = Dense(144, activation='relu')(encoded6)
decoded2 = Dense(169, activation='relu')(decoded1)
decoded3 = Dense(324, activation='relu')(decoded2)
decoded4 = Dense(648, activation='relu')(decoded3)
decoded5 = Dense(1296, activation='relu')(decoded4)
decoded6 = Dense(2592, activation='relu')(decoded5)
decoded = Dense(5184, activation='sigmoid')(decoded6)

autoencoder = Model(input=input_img, output=decoded)

# encoder model
encoder = Model(input=input_img, output=encoded6)

# decoder model
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-7]
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

# Compiling
autoencoder.compile(loss='binary_crossentropy',
                    optimizer='adadelta') 

# Fitting model
autoencoder.fit(X_train, 
            X_train,
            batch_size=300,
            nb_epoch=2,
            shuffle=True,
            validation_data=(X_test, X_test))

# DIsplay the ouputs
encoded_imgs = encoder.predict(X_test)
decoded_imgs = autoencoder.predict(X_train)

n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X_test[i].reshape(72,72))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(72, 72))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.savefig('train_ouput.png')

#encoder = Model(input_img, encoded3)
np.savetxt('encoded_train.csv', encoder.predict(X_train), delimiter=' ')
np.savetxt('label_train.csv', y_train,delimiter=' ')

np.savetxt('encoded_test.csv', encoder.predict(X_test), delimiter=' ')
np.savetxt('label_test.csv', y_test, delimiter=' ')
