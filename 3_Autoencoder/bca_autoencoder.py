# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 17:54:20 2020

@author: Mike Toreno II
"""




import argparse

parser = argparse.ArgumentParser(description = "a very simple autoencoder")  
parser.add_argument("-i","--input_dir", help="input directory", default = "../inputs/sca/sca_preprocessed_data/")
parser.add_argument("-o","--output_dir", help="output directory", default = "../inputs/sca/BCA_output/")
parser.add_argument("--loss", default = "poisson", type = str, choices = ["poisson_loss", "poisson", "mse","mae","mape","msle","squared_hinge","hinge","binary_crossentropy","categorical_crossentropy","kld","cosine_proximity"])
parser.add_argument("--activation", default = "relu", type = str, choices = ["relu", "sigmoid", "mixed1", "mixed2"])
parser.add_argument("--optimizer", default = "Adam", type = str, choices = ["SGD", "RMSprop", "Adam", "Adadelta","Adagrad","Adamax","Nadam","Ftrl"])

args = parser.parse_args()


input_dir = args.input_dir
output_dir = args.output_dir



# if this is not imported before keras, we get weird errors >.<
from sklearn.preprocessing import StandardScaler




from datetime import datetime
print(datetime.now().strftime("\n\n%d. %b %Y, %H:%M:%S>"), "Starting BCA.py")





from keras.objectives import mse, mae, mape, msle, squared_hinge, hinge, binary_crossentropy, categorical_crossentropy, sparse_categorical_crossentropy, kld, poisson#, cosine_proximity
# Define Loss for the training
# if args.loss == "poisson_loss":
#     loss = poisson_loss
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
    




print(datetime.now().strftime("%H:%M:%S>"), "loading data...")

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

print(datetime.now().strftime("%H:%M:%S>"), "compile model...")

import keras
from keras import layers



numfeatures = data[0].shape


# I define the length as the number of features. 
input_data = keras.Input(shape=numfeatures)       

# layers

encoded = layers.Dense(64, activation=act1)(input_data)
encoded = layers.Dense(32, activation=act2)(encoded)
decoded = layers.Dense(64, activation=act2)(encoded)
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
autoencoder.compile(optimizer=args.optimizer, loss = "poisson")




# %%
print(datetime.now().strftime("%H:%M:%S>"), "Train model...")

# normalize and scale the data. 
'''so I don't know if this makes it any better, cuz then we could have not 
gone the extra mile to avoid doing this in the real preprocessing, but idunno.

also they use another scaling, one that scales between 0 and 1 (manually, just divide by largest)
idk what makes sense here lmao lmao
'''

scaler = StandardScaler()
scaler.fit(traindata)
traindata = scaler.transform(traindata)
testdata = scaler.transform(testdata)





from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Callbacks
callbacks = []
lr_cb = ReduceLROnPlateau(monitor='val_loss', patience=15, verbose=1)      #patientce = how many must plateau
callbacks.append(lr_cb)

es_cb = EarlyStopping(monitor='val_loss', patience=35, verbose=1)
callbacks.append(es_cb)







autoencoder.fit(traindata, traindata,
                epochs=1000,
                batch_size=256,
                shuffle=True,
                callbacks = callbacks,
                validation_data=(testdata, testdata))


history = autoencoder.history





# %% Get Result:
        
print(datetime.now().strftime("%H:%M:%S>"), "create output data & plots...")
    
latent_testdata = encoder.predict(testdata)
denoised_testdata = autoencoder.predict(testdata)
    
latent_traindata = encoder.predict(traindata)
denoised_traindata = autoencoder.predict(traindata)
    


# %%
import matplotlib.pyplot as plt

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

plt.savefig(output_dir + "training_history.png")












# %% Write output
print(datetime.now().strftime("%H:%M:%S>"), "write output...")

import os
os.makedirs(output_dir, exist_ok=True)





denoised_outdata = np.zeros(shape = (data.shape))
denoised_outdata[train_index] = denoised_traindata
denoised_outdata[test_index] = denoised_testdata


latent_outdata = np.zeros(shape = (data.shape[0], latent_testdata.shape[1]))
latent_outdata[train_index] = latent_traindata
latent_outdata[test_index] = latent_testdata



pd.DataFrame(denoised_outdata).to_csv(output_dir + "denoised_matrix.tsv",
                                                      sep='\t',
                                                      index=None,
                                                      header=None,
                                                      float_format='%.6f')

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


















