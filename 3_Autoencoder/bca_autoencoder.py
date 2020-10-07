# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 17:54:20 2020

@author: Mike Toreno II
"""




import argparse

parser = argparse.ArgumentParser(description = "program to preprocess the raw singlecell data")  
parser.add_argument("-i","--input_dir", help="input directory", default = "../inputs/sca/sca_preprocessed_data/")
parser.add_argument("-o","--output_dir", help="output directory", default = "../inputs/sca/autoencoder_output/")
parser.add_argument("--loss", default = "mse", type = str, choices = ["poisson_loss", "poisson", "mse","mae","mape","msle","squared_hinge","hinge","binary_crossentropy","categorical_crossentropy","kld","cosine_proximity"])
args = parser.parse_args()



input_dir = args.input_dir








import numpy as np
import pandas as pd




data = np.loadtxt(open(input_dir + "matrix.tsv"), delimiter="\t")
genes = pd.read_csv(input_dir + "genes.tsv", delimiter = "\t", header = None)
barcodes = pd.read_csv(input_dir + "barcodes.tsv", delimiter = "\t", header = None)

test_index = np.loadtxt(fname = input_dir + "test_index.tsv", dtype = bool)
train_index = np.logical_not(test_index)


testdata = data[test_index]
traindata = data[train_index]






# %%

import keras
from keras import layers



numfeatures = data[0].shape


# I define the length as the number of features. 
input_img = keras.Input(shape=(numfeatures))       

# layers
encoded = layers.Dense(32, activation = "relu")(input_img)
decoded = layers.Dense(numfeatures, activation = "sigmoid")(encoded)


# models
autoencoder = keras.Model(input_img, decoded)

encoder = keras.Model(input_img, encoded)

encoded_input = keras.Input(shape = (32,))
decoder_layer = autoencoder.layers[-1]
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))


# Compile
autoencoder.compile(optimizer='adam', loss = "binary_crossentropy")


# %%






















