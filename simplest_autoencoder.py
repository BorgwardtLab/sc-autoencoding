# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 02:20:31 2020

@author: Mike Toreno II
"""

# %% SIMPLE autoencoder
##############################################################################

from keras.layers import Input, Dense
from keras.models import Model
global_epochs = 1

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats
## "bottleneck size"

# this is our input placeholder
input_img = Input(shape=(784,))


####### Define encoding/ decoding
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)

##### Create autoencoder
# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)




###### create encoder only (for fun)
encoder = Model(input_img, encoded)

###### Create decoder only
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

# Input() instantiates a Keras Tensor, and these allow to build a model just by 
# knowing inputs and outputs of the model.
# Note that the decoder_layer gets pulled from the complete autoencoder instance.

####### Compile
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')



# %% Preprocess minst Data
from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()


# flatten vector from 28x28 to 1x784
# also normalize values between 0 and 1. 
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)


# % Train Autoencoder 
epochs = global_epochs;

autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))



# % Evaluate by taking a few images from *test* set
# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)


# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
plt.savefig("./output/simplest_result.png")
## don't bother trying to show the encoded version, it doesn't have a 28x28 shape










# %% Sparcity constraint autoencoder
##############################################################################
''' instead of just having the bottleneck layer (which leads to sth close to PCA)
you can also add a sparcity constraint, meaning fewer units fire at any given time.
> add an activity_regularizer to the dense layer
'''

from keras import regularizers
encoding_dim = 32
input_img = Input(shape=(784,))
# add a Dense layer with a L1 activity regularizer



encoded = Dense(encoding_dim, activation='relu',
                activity_regularizer=regularizers.l1(10e-5))(input_img)
decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')




# % Train Autoencoder 
epochs = global_epochs;

autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))



# % Evaluate by taking a few images from *test* set
# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)


# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
plt.savefig("./output/constrained_result.png")














# %% Deep autoencoder
##############################################################################
epochs = global_epochs;


input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)


# compare to previous AE, where we used     
# conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
# pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
# (we come to the same result in the end.)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))


#### Visualize Deep Autoencoder
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
plt.savefig("./output/deep_result.png")
## don't bother trying to show the encoded version, it doesn't have a 28x28 shape






# %% Convolutional autoencoder
##############################################################################
'''
from the priunciple this one is identical to the one previously implemented
'''


from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model


input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')




# Prepare data (full images here, not "vectorized")
from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format



# %% Train/ follow training with TensorBoard
# '''
# to visualize results of model durin gtraining, they use the 
# "tensorflow backend" and the TensorBoard callback
# '''

# import os
# # 1. start a TensorBoard server, that will read logs stored at /tmp/autoencoder
# os.system("tensorboard --logdir=/tmp/autoencoder")

# # 2. Train the model
# # in the callbacks list we pass an instance of the tensorboard callback.
# # after every epoch, the callback will write logs to /tmp/autoencoder
# # will then be read by the tensorboard server. 
# from keras.callbacks import TensorBoard

# autoencoder.fit(x_train, x_train,
#                 epochs=epochs,
#                 batch_size=128,
#                 shuffle=True,
#                 validation_data=(x_test, x_test),
#                 callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])




epochs = global_epochs

from keras.callbacks import TensorBoard
import datetime


directory = "logZZZZ" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# Failed to create a directory: ./logs/fit/20200622-145140\train; No such file or directory [Op:CreateSummaryFileWriter] <- if you write subdirectory instead of logzzzz
# it will probably work on linux
tensorboard_callback = TensorBoard(log_dir=directory, histogram_freq = 1)


autoencoder.fit(x_train, x_train,
                epochs=epochs,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[tensorboard_callback])



'''
Try typing which tensorboard in your terminal. It should exist if you installed with pip as mentioned in the tensorboard README (although the documentation doesn't tell you that you can now launch tensorboard without doing anything else).

You need to give it a log directory. If you are in the directory where you saved your graph, you can launch it from your terminal with something like:

tensorboard --logdir .
or more generally:

tensorboard --logdir /path/to/log/directory
for any log directory.

Then open your favorite web browser and type in localhost:6006 to connect.

That should get you started. As for logging anything useful in your training process, you need to use the TensorFlow Summary API. You can also use the TensorBoard callback in Keras.
'''



# %% visualize convolutional

decoded_imgs = autoencoder.predict(x_test)


n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
plt.savefig("./output/convolutional_result.png")





# %% visualize encoded representation (8x4x4)





encoder = Model(input_img, encoded)
encoded_imgs = encoder.predict(x_test)


n = 10
plt.figure(figsize=(20, 8))
for i in range(n):
    ax = plt.subplot(1, n, i+1)
    plt.imshow(encoded_imgs[i].reshape(4, 4 * 8).T) #reshape from 8x4x4 to 4x32
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
plt.savefig("./output/convolutional_encoded.png")





##############################################################################
# %% use convolutional ae on a denoising problem. 

# Generate noisy data. 
import matplotlib.pyplot as plt
from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape) 
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape) 

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)


# inspect noisy digits
n = 10
plt.figure(figsize=(20, 2))
for i in range(n):
    ax = plt.subplot(1, n, i+ 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
plt.savefig("./output/noisy_original.png")






# %% Define Autoencoder
'''generally its the same principle, however, this model has more filters per layer'''
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from keras.callbacks import TensorBoard

input_img = Input(shape=(28, 28, 1))  # adapt this if using `channels_first` image data format


x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (7, 7, 32)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

#PREVIOUS ONE
'''
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
'''


# Train for epochs
epochs = global_epochs

autoencoder.fit(x_train_noisy, x_train,
                epochs=epochs,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test_noisy, x_test),
                callbacks=[TensorBoard(log_dir='log2_NOICE', histogram_freq=0, write_graph=False)])


# Visualize

decoded_imgs = autoencoder.predict(x_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n + 1)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
plt.savefig("./output/noisy_reconstructed.png")











# %% Sequence to Sequence / Long Short term memory LSTM
##############################################################################
'''if you want to capture temporal structure, e.g. LSTM. 
use LSTM encoder to turn inpout sequence into a single vector that contains info about the entire
sequence., then repeat vector n times with n = number of timesteps in output sequence, and
run lstm decoder to turn this constant sequence into target sequence'''
#ez pz

# Code example
# from keras.layers import Input, LSTM, RepeatVector
# from keras.models import Model
# inputs = Input(shape=(timesteps, input_dim))
# encoded = LSTM(latent_dim)(inputs)

# decoded = RepeatVector(timesteps)(encoded)
# decoded = LSTM(input_dim, return_sequences=True)(decoded)

# sequence_autoencoder = Model(inputs, decoded)
# encoder = Model(inputs, encoded)





# %% Variational autoencoder VAE
##############################################################################
'''Instead of lettng my NN learn an arbitrary function, we learn the parameters
of a probability distribution modelling our data. (the AE learns a latent variable model)

By sampling from this distribution, we can generate new input data samples. 
-> VAE is a generative model. 

input samples are turned into two parameters in the latent space: mean and sigma. 

Sample similar points from latent normal distribution via:
    z = mean + exp(sigma)*epsilon, with epsilon = random tensor. 
The decoder maps these latent space points back to the original input data. 

This AE has two loss functions. 
- Reconstruction loss, forces the decoded samples to match the initial inputs. just like before
- KL divergence: between learned latent distribution, and prior distribution (regularization term. )

'''








# Complete code

# '''Example of VAE on MNIST dataset using MLP
# The VAE has a modular design. The encoder, decoder and VAE
# are 3 models that share weights. After training the VAE model,
# the encoder can be used to generate latent vectors.
# The decoder can be used to generate MNIST digits by sampling the
# latent vector from a Gaussian distribution with mean = 0 and std = 1.
# # Reference
# [1] Kingma, Diederik P., and Max Welling.
# "Auto-Encoding Variational Bayes."
# https://arxiv.org/abs/1312.6114
# '''

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import keras
# from keras.layers import Lambda, Input, Dense
# from keras.models import Model
# from keras.datasets import mnist
# from keras.losses import mse, binary_crossentropy
# from keras.utils import plot_model
# from keras import backend as K

# import numpy as np
# import matplotlib.pyplot as plt
# import argparse
# import os


# # reparameterization trick
# # instead of sampling from Q(z|X), sample epsilon = N(0,I)
# # z = z_mean + sqrt(var) * epsilon
# def sampling(args):
#     """Reparameterization trick by sampling from an isotropic unit Gaussian.
#     # Arguments
#         args (tensor): mean and log of variance of Q(z|X)
#     # Returns
#         z (tensor): sampled latent vector
#     """

#     z_mean, z_log_var = args
#     batch = K.shape(z_mean)[0]
#     dim = K.int_shape(z_mean)[1]
#     # by default, random_normal has mean = 0 and std = 1.0
#     epsilon = K.random_normal(shape=(batch, dim))
#     return z_mean + K.exp(0.5 * z_log_var) * epsilon


# def plot_results(models,
#                   data,
#                   batch_size=128,
#                   model_name="vae_mnist"):
#     """Plots labels and MNIST digits as a function of the 2D latent vector
#     # Arguments
#         models (tuple): encoder and decoder models
#         data (tuple): test data and label
#         batch_size (int): prediction batch size
#         model_name (string): which model is using this function
#     """

#     encoder, decoder = models
#     x_test, y_test = data
#     os.makedirs(model_name, exist_ok=True)

#     filename = os.path.join(model_name, "vae_mean.png")
#     # display a 2D plot of the digit classes in the latent space
#     z_mean, _, _ = encoder.predict(x_test,
#                                     batch_size=batch_size)
#     plt.figure(figsize=(12, 10))
#     plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
#     plt.colorbar()
#     plt.xlabel("z[0]")
#     plt.ylabel("z[1]")
#     plt.savefig(filename)
#     plt.show()

#     filename = os.path.join(model_name, "digits_over_latent.png")
#     # display a 30x30 2D manifold of digits
#     n = 30
#     digit_size = 28
#     figure = np.zeros((digit_size * n, digit_size * n))
#     # linearly spaced coordinates corresponding to the 2D plot
#     # of digit classes in the latent space
#     grid_x = np.linspace(-4, 4, n)
#     grid_y = np.linspace(-4, 4, n)[::-1]

#     for i, yi in enumerate(grid_y):
#         for j, xi in enumerate(grid_x):
#             z_sample = np.array([[xi, yi]])
#             x_decoded = decoder.predict(z_sample)
#             digit = x_decoded[0].reshape(digit_size, digit_size)
#             figure[i * digit_size: (i + 1) * digit_size,
#                     j * digit_size: (j + 1) * digit_size] = digit

#     plt.figure(figsize=(10, 10))
#     start_range = digit_size // 2
#     end_range = (n - 1) * digit_size + start_range + 1
#     pixel_range = np.arange(start_range, end_range, digit_size)
#     sample_range_x = np.round(grid_x, 1)
#     sample_range_y = np.round(grid_y, 1)
#     plt.xticks(pixel_range, sample_range_x)
#     plt.yticks(pixel_range, sample_range_y)
#     plt.xlabel("z[0]")
#     plt.ylabel("z[1]")
#     plt.imshow(figure, cmap='Greys_r')
#     plt.savefig(filename)
#     plt.show()


# # MNIST dataset
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# image_size = x_train.shape[1]
# original_dim = image_size * image_size
# x_train = np.reshape(x_train, [-1, original_dim])
# x_test = np.reshape(x_test, [-1, original_dim])
# x_train = x_train.astype('float32') / 255
# x_test = x_test.astype('float32') / 255

# # network parameters
# input_shape = (original_dim, )
# intermediate_dim = 512
# batch_size = 128
# latent_dim = 2
# epochs = 50

# # VAE model = encoder + decoder
# # build encoder model
# inputs = Input(shape=input_shape, name='encoder_input')
# x = Dense(intermediate_dim, activation='relu')(inputs)
# z_mean = Dense(latent_dim, name='z_mean')(x)
# z_log_var = Dense(latent_dim, name='z_log_var')(x)

# # use reparameterization trick to push the sampling out as input
# # note that "output_shape" isn't necessary with the TensorFlow backend
# z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# # instantiate encoder model
# encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
# encoder.summary()
# plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# # build decoder model
# latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
# x = Dense(intermediate_dim, activation='relu')(latent_inputs)
# outputs = Dense(original_dim, activation='sigmoid')(x)

# # instantiate decoder model
# decoder = Model(latent_inputs, outputs, name='decoder')
# decoder.summary()
# plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# # instantiate VAE model
# outputs = decoder(encoder(inputs)[2])
# vae = Model(inputs, outputs, name='vae_mlp')

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     help_ = "Load h5 model trained weights"
#     parser.add_argument("-w", "--weights", help=help_)
#     help_ = "Use mse loss instead of binary cross entropy (default)"
#     parser.add_argument("-m",
#                         "--mse",
#                         help=help_, action='store_true')
#     args = parser.parse_args()
#     models = (encoder, decoder)
#     data = (x_test, y_test)

#     # VAE loss = mse_loss or xent_loss + kl_loss
#     if args.mse:
#         reconstruction_loss = mse(inputs, outputs)
#     else:
#         reconstruction_loss = binary_crossentropy(inputs,
#                                                   outputs)

#     reconstruction_loss *= original_dim
#     kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
#     kl_loss = K.sum(kl_loss, axis=-1)
#     kl_loss *= -0.5
#     vae_loss = K.mean(reconstruction_loss + kl_loss)
#     vae.add_loss(vae_loss)
#     vae.compile(optimizer='adam')

#     if args.weights:
#         vae.load_weights(args.weights)
#     else:
#         # train the autoencoder
#         vae.fit(x_train,
#                 epochs=epochs,
#                 batch_size=batch_size,
#                 validation_data=(x_test, None))
#         vae.save_weights('vae_mlp_mnist.h5')

#     plot_results(models,
#                   data,
#                   batch_size=batch_size,
#                   model_name="vae_mlp")









