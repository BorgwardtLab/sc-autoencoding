# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 17:54:20 2020

@author: Mike Toreno II
"""




import argparse

parser = argparse.ArgumentParser(description = "a very simple autoencoder")  
parser.add_argument("-i","--input_dir", help="input directory", default = "../../inputs/data/preprocessed_data_autoencoder/")
parser.add_argument("-o","--output_dir", help="output directory", default = "../inputs/autoencoder_data/BCA_output/")
parser.add_argument("-p","--outputplot_dir", help="plot directory", default = "../outputs/autoencoder_data/BCA/")
parser.add_argument("--loss", default = "poisson", type = str, choices = ["poisson_loss", "poisson", "mse","mae","mape","msle","squared_hinge","hinge","binary_crossentropy","categorical_crossentropy","kld","cosine_proximity"])
parser.add_argument("--activation", default = "relu", type = str, choices = ["relu", "sigmoid", "mixed1", "mixed2"])
parser.add_argument("--optimizer", default = "Adam", type = str, choices = ["SGD", "RMSprop", "Adam", "Adadelta","Adagrad","Adamax","Nadam","Ftrl"])
parser.add_argument("--mode", default = "complete", help="chose traintest-split, nosplit or both", choices=['complete','split','nosplit'])
parser.add_argument("--verbose", type = int, default = 2, help="0: quiet, 1:progress bar, 2:1 line per epoch")
parser.add_argument("--splitnumber", type = int, help="in order to run all splits at the same time, they can be run individually. If mode == split, enter a number here to only do that split. Please ensure that the split exists. ")
args = parser.parse_args()


# if this is not imported before keras, we get weird errors >.<
from sklearn.preprocessing import StandardScaler

from datetime import datetime
import numpy as np
import pandas as pd
import keras
from keras import layers
from keras.objectives import mse, mae, mape, msle, squared_hinge, hinge, binary_crossentropy, categorical_crossentropy, sparse_categorical_crossentropy, kld, poisson#, cosine_proximity
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import os
import matplotlib.pyplot as plt
import sys


try:
    os.chdir(os.path.dirname(sys.argv[0]))
except:
    pass



import tensorflow as tf
def _nan2zero(x):
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)
def _nelem(x):
    nelem = tf.reduce_sum(tf.cast(~tf.math.is_nan(x), tf.float32))   # just summing all the elements of a tensor
    return tf.cast(tf.where(tf.equal(nelem, 0.), 1., nelem), x.dtype)
def poisson_loss(y_true, y_pred):
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    # we can use the Possion PMF from TensorFlow as well
    # dist = tf.math.contrib.distributions
    # return -tf.reduce_mean(dist.Poisson(y_pred).log_pmf(y_true))
    nelem = _nelem(y_true)
    y_true = _nan2zero(y_true)
    # last term can be avoided since it doesn't depend on y_pred
    # however keeping it gives a nice lower bound to zero
    ret = y_pred - y_true*tf.math.log(y_pred+1e-10) + tf.math.lgamma(y_true+1.0)
    print("ret = {}".format(ret))
    print("nelem = {}".format(nelem))
    result = tf.math.divide(tf.math.reduce_sum(ret), nelem)
    return result



source_input_dir = args.input_dir
source_output_dir = args.output_dir
source_outputplot_dir = args.outputplot_dir



# Define Loss for the training
if args.loss == "poisson_loss":
    loss = poisson_loss
if args.loss == "poisson":
    loss = poisson              
elif args.loss == "mse":
    loss = mse    
elif args.loss == "mae":
    loss = mae   
elif args.loss == "mape":
    loss = mape   
elif args.loss == "msle":
    loss = msle
elif args.loss == "squared_hinge":
    loss = squared_hinge
elif args.loss == "hinge":
    loss = hinge
elif args.loss == "binary_crossentropy":
    loss = binary_crossentropy
elif args.loss == "categorical_crossentropy":
    loss = categorical_crossentropy
elif args.loss == "kld":
    loss = kld
else:
    print("ERROR INVALID LOSS OR STH")



if args.activation == "relu":
    act1 = "relu"
    act2 = "relu"
elif args.activation == "sigmoid":
    act1 = "sigmoid"
    act2 = "sigmoid"
elif args.activation == "mixed1":
    act1 = "sigmoid"
    act2 = "relu"
elif args.activation == "mixed2":
    act1 = "relu"
    act2 = "sigmoid"
else:
    print("ERROR: ACTIVATION FUNCTION FAILED BECAUSE OF A REASON")
    










# args.mode == "nosplit"


if args.mode == "complete":
    nosplit = True
    split = True
elif args.mode == "split":
    nosplit = False
    split = True
elif args.mode == "nosplit":
    nosplit = True
    split = False
else:
    print("invalid mode")






    
        
# %% split = TRUE    
    
if split == True:
    print(datetime.now().strftime("%H:%M:%S>"), "Starting bca_autoencoder.py (split)")    
 
    # determine number of splits
    num_splits = 0
    cancel = False
    directory = source_input_dir + "split_" + str(num_splits + 1)
    if os.path.isdir(directory) == False:
        print("ERROR: NO SPLITS DETECTED")
        sys.exit()
    else:
        while True:
            num_splits += 1
            directory = source_input_dir + "split_" + str(num_splits + 1)
            # print(directory)
            isdirectory = os.path.isdir(directory)
            if isdirectory == False:
                print(datetime.now().strftime("%H:%M:%S>"), str(num_splits) + " splits detected\n")    
                break
             
     
    
# %% loop through splits

    splits_to_do = range(1, num_splits + 1)
    
    # check if only one split is requested
    if args.splitnumber != None and args.mode == "split":
        print("custom split number detected")
        if args.splitnumber > 0 and args.splitnumber <= num_splits:
            print("Script will only run on split", args.splitnumber)
            splits_to_do = [args.splitnumber]
        else:
            print("illegal split number entered ({:d}). Please choose a number between 1 and {:d}".format(args.splitnumber, num_splits))
            

    
    for split in splits_to_do:
        
        print(datetime.now().strftime("%H:%M:%S>"), "\nStarting split #" + str(split))      
            
        input_dir = source_input_dir + "split_" + str(split) + "/"
        output_dir = source_output_dir + "split_" + str(split) + "/"
        outputplot_dir = source_outputplot_dir + "split_" + str(split) + "/"
  

    
        # %% reading input
    
        
        print(datetime.now().strftime("%H:%M:%S>"), "loading data...")
        
        data = np.loadtxt(open(input_dir + "matrix.tsv"), delimiter="\t")
        genes = pd.read_csv(input_dir + "genes.tsv", delimiter = "\t", header = None)
        barcodes = pd.read_csv(input_dir + "barcodes.tsv", delimiter = "\t", header = None)
        
        test_index = np.loadtxt(fname = input_dir + "test_index.tsv", dtype = bool)
        train_index = np.logical_not(test_index)
        
        testdata = data[test_index]
        traindata = data[train_index]
        
        
        
        
        
        
        # %% model
        
        print(datetime.now().strftime("%H:%M:%S>"), "compile model...")
        
        
        numfeatures = data[0].shape
        print("{:d} features per sample detected".format(numfeatures[0]))
        
        
        
        # I define the length as the number of features. 
        input_data = keras.Input(shape=numfeatures)       
        
        
        
        # layers [IF YOU EXPERIMENT WITH THIS, REMEMBER THERE IS A SECOND ONE IN THIS SCRIPT]
        encoded = layers.Dense(88, activation=act1)(input_data)
        encoded = layers.Dense(51, activation=act1)(encoded)
        encoded = layers.Dense(32, activation=act1)(encoded)
        decoded = layers.Dense(45, activation=act1)(encoded)
        decoded = layers.Dense(numfeatures[0], activation=act1)(decoded)
        
        
        
        
        
        # models
        autoencoder = keras.Model(input_data, decoded)
        
        encoder = keras.Model(input_data, encoded)
        
        encoded_input = keras.Input(shape = (32,))
        decoder = keras.Model(encoded_input, autoencoder.layers[-1](autoencoder.layers[-2](encoded_input)))
        # this is complete trash, but fortunately, we don't need the decoder, so whatev.
        # I've also tried:
        # decoder = keras.Model(encoded_input, decoded)
        
        
        
        
        # Compile
        autoencoder.compile(optimizer=args.optimizer, loss = loss)
        
        
        
        
        # Callbacks
        callbacks = []
        lr_cb = ReduceLROnPlateau(monitor='val_loss', patience=15, verbose=1)      #patientce = how many must plateau
        callbacks.append(lr_cb)
        
        es_cb = EarlyStopping(monitor='val_loss', patience=35, verbose=1)
        callbacks.append(es_cb)
                


        
        
        # %%
        print(datetime.now().strftime("%H:%M:%S>"), "Train model...")
        
        # normalize and scale the data. 
        
        scaler = StandardScaler()
        scaler.fit(traindata)
        traindata = scaler.transform(traindata)
        testdata = scaler.transform(testdata)
        
        
        autoencoder.fit(traindata, traindata,
                        epochs=500,
                        batch_size=256,
                        shuffle=True,
                        callbacks = callbacks,
                        #validation_data=(testdata, testdata))
                        validation_split = 0.1,
                        verbose=args.verbose) # remember to change the other one too if you change this one
        
        history = autoencoder.history
        
        
        
        
        
        # %% Get Result:
                
        print(datetime.now().strftime("%H:%M:%S>"), "create output data & plots...")
            
        latent_testdata = encoder.predict(testdata)
        denoised_testdata = autoencoder.predict(testdata)
            
        latent_traindata = encoder.predict(traindata)
        denoised_traindata = autoencoder.predict(traindata)
            
        
        
        # %%
        
        
        ploss = history.history["loss"]
        plr = history.history["lr"]
        pval_loss = history.history["val_loss"]
        
        num_epochs = len(ploss)
        epochs = history.epoch
        
        
        fig, (ax1, ax2) = plt.subplots(2, 1)
        fig.suptitle("History")
        
        ax1.plot(range(num_epochs), plr, 'b')
        ax1.set_ylabel("learning rate")
        #ax1.set_xticks([])
        ax1.tick_params(axis = 'x', which='both', bottom = True, top = False, labelbottom = False)
        
        
        ax2.plot(range(num_epochs), ploss, 'r')
        ax2.plot(range(num_epochs), pval_loss, 'g')
        ax2.legend(['loss', 'validation loss'])
        ax2.set_xlabel("epoch number")
        ax2.set_ylabel("losses")
        
        
        fig.show()
        
        os.makedirs(outputplot_dir, exist_ok=True)
        plt.savefig(outputplot_dir + "training_history.png")
        
        
        
        
        
        
        
        # %% Write output
        print(datetime.now().strftime("%H:%M:%S>"), "write output...")
        
        
        os.makedirs(output_dir, exist_ok=True)
        
        
        
        denoised_outdata = np.zeros(shape = (data.shape))
        denoised_outdata[train_index] = denoised_traindata
        denoised_outdata[test_index] = denoised_testdata
        
        
        latent_outdata = np.zeros(shape = (data.shape[0], latent_testdata.shape[1]))
        latent_outdata[train_index] = latent_traindata
        latent_outdata[test_index] = latent_testdata
        
        
        
        # pd.DataFrame(denoised_outdata).to_csv(output_dir + "denoised_matrix.tsv",
        #                                                       sep='\t',
        #                                                       index=None,
        #                                                       header=None,
        #                                                       float_format='%.6f')
        
        pd.DataFrame(latent_outdata).to_csv(output_dir + "latent_layer.tsv",
                                                              sep='\t',
                                                              index=None,
                                                              header=None,
                                                              float_format='%.6f')
        
        pd.DataFrame(latent_outdata).to_csv(output_dir + "matrix.tsv",
                                                              sep='\t',
                                                              index=None,
                                                              header=None,
                                                              float_format='%.6f')
        
        
        barcodes.to_csv(output_dir + "barcodes.tsv", sep = "\t", index = False, header = False)
        
        genes.to_csv(output_dir + "genes.tsv", sep = "\t", index = False, header = False)
        
        np.savetxt(output_dir + "test_index.tsv", test_index, fmt = "%d")
        
        
    
    
    







#########################################################################################################################
# %% 


        
       
# %% NO SPLIT
if nosplit == True:
    
    print(datetime.now().strftime("%H:%M:%S>"), "Starting bca_autoencoder.py (nosplit)")    
 
    input_dir = source_input_dir + "no_split/"
    output_dir = source_output_dir + "no_split/"
    outputplot_dir = source_outputplot_dir + "no_split/"
    
    
    
    
    
    
      # %%  
    
    print(datetime.now().strftime("%H:%M:%S>"), "loading data...")
        
    data = np.loadtxt(open(input_dir + "matrix.tsv"), delimiter="\t")
    genes = pd.read_csv(input_dir + "genes.tsv", delimiter = "\t", header = None)
    barcodes = pd.read_csv(input_dir + "barcodes.tsv", delimiter = "\t", header = None)
    
    
    
    
    # %%
    
    print(datetime.now().strftime("%H:%M:%S>"), "compile model...")
    
    
    numfeatures = data[0].shape
    print("{:d} features per sample detected".format(numfeatures[0]))
    
    
    
    # I define the length as the number of features. 
    input_data = keras.Input(shape=numfeatures)       
    
    
    # layers [IF YOU EXPERIMENT WITH THIS, REMEMBER THERE IS A SECOND ONE IN THIS SCRIPT]
    encoded = layers.Dense(88, activation=act1)(input_data)
    encoded = layers.Dense(51, activation=act1)(encoded)
    encoded = layers.Dense(32, activation=act1)(encoded)
    decoded = layers.Dense(45, activation=act1)(encoded)
    decoded = layers.Dense(numfeatures[0], activation=act1)(decoded)
    
    
    
    # models
    autoencoder = keras.Model(input_data, decoded)
    
    encoder = keras.Model(input_data, encoded)
    
    encoded_input = keras.Input(shape = (32,))
    decoder = keras.Model(encoded_input, autoencoder.layers[-1](autoencoder.layers[-2](encoded_input)))
    # this is complete trash, but fortunately, we don't need the decoder, so whatev.
    # I've also tried:
    # decoder = keras.Model(encoded_input, decoded)
    
    
    # Compile
    autoencoder.compile(optimizer=args.optimizer, loss = loss)
    
    
    
    
    
    
    # Callbacks
    callbacks = []
    lr_cb = ReduceLROnPlateau(monitor='val_loss', patience=15, verbose=1)      #patientce = how many must plateau
    callbacks.append(lr_cb)
    
    es_cb = EarlyStopping(monitor='val_loss', patience=35, verbose=1)
    callbacks.append(es_cb)
    
    
    
    
    
    
    
    
    # %%
    print(datetime.now().strftime("%H:%M:%S>"), "Train model...")
    
    # normalize and scale the data. 
    '''so I don't know if this makes it any better, cuz then we could have not 
    gone the extra mile to avoid doing this in the real preprocessing, but idunno.
    
    also they use another scaling, one that scales between 0 and 1 (manually, just divide by largest)
    idk what makes sense here lmao lmao
    '''
    
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    
    
    
    autoencoder.fit(data, data,
                    epochs=500,
                    batch_size=256,
                    shuffle=True,
                    callbacks = callbacks,
                    #validation_data=(testdata, testdata))
                    validation_split = 0.1,
                    verbose=args.verbose)     # remember to change the other one too if you change this one
    
    history = autoencoder.history
    
    
    
    
    
    # %% Get Result:
            
    print(datetime.now().strftime("%H:%M:%S>"), "create output data & plots...")
        
    latent = encoder.predict(data)
    denoised = autoencoder.predict(data)
        


    # %% Plot
    
    ploss = history.history["loss"]
    plr = history.history["lr"]
    pval_loss = history.history["val_loss"]
    
    num_epochs = len(ploss)
    epochs = history.epoch
    
    
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle("History")
    
    ax1.plot(range(num_epochs), plr, 'b')
    ax1.set_ylabel("learning rate")
    #ax1.set_xticks([])
    ax1.tick_params(axis = 'x', which='both', bottom = True, top = False, labelbottom = False)
    
    
    ax2.plot(range(num_epochs), ploss, 'r')
    ax2.plot(range(num_epochs), pval_loss, 'g')
    ax2.legend(['loss', 'validation loss'])
    ax2.set_xlabel("epoch number")
    ax2.set_ylabel("losses")
    
    
    fig.show()
    os.makedirs(outputplot_dir, exist_ok=True)
    plt.savefig(outputplot_dir + "training_history.png")
    
    
    
    
    # %% Write output
    print(datetime.now().strftime("%H:%M:%S>"), "write output...")
    
    
    os.makedirs(output_dir, exist_ok=True)
    
    # pd.DataFrame(denoised).to_csv(output_dir + "denoised_matrix.tsv",
    #                                                       sep='\t',
    #                                                       index=None,
    #                                                       header=None,
    #                                                       float_format='%.6f')
    
    pd.DataFrame(latent).to_csv(output_dir + "latent_layer.tsv",
                                                          sep='\t',
                                                          index=None,
                                                          header=None,
                                                          float_format='%.6f')
    
    pd.DataFrame(latent).to_csv(output_dir + "matrix.tsv",
                                                          sep='\t',
                                                          index=None,
                                                          header=None,
                                                          float_format='%.6f')
    
    barcodes.to_csv(output_dir + "barcodes.tsv", sep = "\t", index = False, header = False)
    
    genes.to_csv(output_dir + "genes.tsv", sep = "\t", index = False, header = False)
    
    np.savetxt(output_dir + "test_index.tsv", test_index, fmt = "%d")
    
    
    
    

