# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 17:22:21 2020

@author: Mike Toreno II
"""

import argparse

parser = argparse.ArgumentParser(description = "program to preprocess the raw singlecell data")  
parser.add_argument("-i","--input_dir", help="input directory", default = "../inputs/sca/sca_preprocessed_data/")
parser.add_argument("-o","--output_dir", help="output directory", default = "../inputs/sca/autoencoder_output/")
parser.add_argument("--loss", default = "mse", type = str, choices = ["poisson_loss", "poisson", "mse","mae","mape","msle","squared_hinge","hinge","binary_crossentropy","categorical_crossentropy","kld","cosine_proximity"])
args = parser.parse_args()


import os
import pickle
from datetime import datetime

from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization, Lambda
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from keras.objectives import mse, mae, mape, msle, squared_hinge, hinge, binary_crossentropy, categorical_crossentropy, sparse_categorical_crossentropy, kld, poisson#, cosine_proximity
# from keras.objectives import cosine_proximity

import numpy as np
import anndata

import tensorflow as tf

import keras.optimizers as opt




from keras import backend as K
MeanAct = lambda x: tf.clip_by_value(K.exp(x), 1e-5, 1e6)
DispAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)

# In the implementations, I try to keep the function signature
# similar to those of Keras objective functions so that
# later on we can use them in Keras smoothly:
# https://github.com/fchollet/keras/blob/master/keras/objectives.py#L7

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


def _nan2zero(x):
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)


def _nelem(x):
    nelem = tf.reduce_sum(tf.cast(~tf.math.is_nan(x), tf.float32))   # just summing all the elements of a tensor
    return tf.cast(tf.where(tf.equal(nelem, 0.), 1., nelem), x.dtype)













# %%

class Autoencoder():
    def __init__(self,
                 input_size,
                 output_size = None,
                 hidden_size=(64, 32, 64),
                 hidden_dropout= 0,
                 #hidden_dropout= (0, 0, 0),
                 input_dropout = 0,
                 batchnorm = True,
                 initializer = 'glorot_uniform',
                 regularizer = None,
                 activation = "relu"):
        
        self.input_size = input_size 
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.hidden_dropout= hidden_dropout
        self.input_dropout= input_dropout
        self.batchnorm = batchnorm
        
        self.initializer = initializer
        self.regularizer = regularizer
        self.activation = activation
## end of inputparsing

        self.sf_layer = None        # created by build
        self.input_layer = None     # created by build
        self.decoder_output = None  # created by build

        self.extra_models = {}      # created by build_output
        self.model = None           # created by build_output

        self.encoder = None         # Model(input layer, center layer), gets cretaed by get_encoder (subpart from build)



        if self.output_size is None:
            self.output_size = input_size

        if isinstance(self.hidden_dropout, list):
            assert len(self.hidden_dropout) == len(self.hidden_size)
        else:
            self.hidden_dropout = [self.hidden_dropout]*len(self.hidden_size)



    def build(self):
        ''' so this is setting up the autoencoder and all its layers, however, 
        they do not get saved individually, instead we only have the last hidden?
        '''
        
        # the names for these are non-negotiable. For the model.fit, a dictionary with exactly these keys is given as input.    
        self.input_layer = Input(shape=(self.input_size,), name='count')
        last_hidden = self.input_layer
        
        if self.input_dropout > 0.0:
            last_hidden = Dropout(self.input_dropout, name='input_dropout')(last_hidden)
                
        self.sf_layer = Input(shape=(1,), name='size_factors')
        


# loop through all layers
        for i, (hid_size, hid_drop) in enumerate(zip(self.hidden_size, self.hidden_dropout)):   
            
# name the layers
            center_idx = int(np.floor(len(self.hidden_size) / 2.0))        
            if i == center_idx:
                layer_name = 'center'
                stage = 'center'  # let downstream know where we are
            elif i < center_idx:
                layer_name = 'enc%s' % (i)
                stage = 'encoder'
            else:
                layer_name = 'dec%s' % (i-center_idx-1)
                stage = 'decoder'

# Create Layers
            last_hidden = Dense(units = hid_size,
                        activation = None,
                        kernel_initializer = self.initializer,
                        kernel_regularizer = self.regularizer,
                        name = layer_name)(last_hidden)
            #dense implements the operation output = activation(dot(input, kernel) + bias)
            # the activation is applied in a separate step though
            
            if self.batchnorm:
                last_hidden = BatchNormalization(center=True, scale=False)(last_hidden)
            
        
# Activation: 
            # applies the activation function to the layer. 
            last_hidden = Activation(self.activation, name='%s_activation'%layer_name)(last_hidden)
            #for advanced activations "PReLU" or "LeakyReLU": check their code again
  
    
# Dropout:
            # randomly sets input units to 0 with a frequency of rate, to prevent overfitting
            if hid_drop > 0.0:
                last_hidden = Dropout(rate = hid_drop, name='%s_drop'%layer_name)(last_hidden)
      
        self.decoder_output = last_hidden
        self.build_output()
        

  
    def build_output(self):
        print(datetime.now().strftime("%H:%M:%S>"), "Building output with loss function: " + self.loss_name)
                
        
        
        
# Define Loss for the training
        if self.loss_name == "poisson_loss":
            self.loss = poisson_loss
            
        elif self.loss_name == "poisson":
            self.loss = poisson            
            
        elif self.loss_name == "mse":
            self.loss = mse
            
        elif self.loss_name == "mae":
            self.loss = mae
            
        elif self.loss_name == "mape":
            self.loss = mape
            
        elif self.loss_name == "msle":
            self.loss = msle

        elif self.loss_name == "squared_hinge":
            self.loss = squared_hinge

        elif self.loss_name == "hinge":
            self.loss = hinge

        elif self.loss_name == "binary_crossentropy":
            self.loss = binary_crossentropy

        elif self.loss_name == "categorical_crossentropy":
            self.loss = categorical_crossentropy

        elif self.loss_name == "kld":
            self.loss = kld


        
        # elif self.loss_name == "cosine_proximity":                # cosine proximity cannot be imported on the server. idk why
        #     self.loss = cosine_proximity

        # elif self.loss_name == "sparse_categorical_crossentropy":
        #     self.loss = sparse_categorical_crossentropy 
        #     print("WARNING/ERROR the sparse_categorial_crossentropy is not suited, it is the same as categorial but for sparse labels.")
            
        else: 
            print("couldn't assign loss correctly :/")
            assert self.loss is not None

        # working       poisson_loss, poisson, mean_squared_error, mse, mae, mape, msle, squared_hinge, hinge, binary_crossentropy, categorical_crossentropy, kld, cosine_proximity
        # not working   sparse_categorical_crossentropy,




# Create the output layer (mean), as well as size/factors lambda       
        mean = Dense(self.output_size, activation = MeanAct,
                     kernel_initializer=self.initializer,
                     kernel_regularizer=self.regularizer,
                     name='mean')(self.decoder_output)
        
        ### ColwiseMultLayer        
        lamfun = lambda l: l[0]*tf.reshape(l[1], (-1,1))
        # lambda function "lamfun" receives an object l. l[0] is multiplied with l[1], thats reshaped (probably to fit)
        ColwiseMultLayer = Lambda(lamfun)
        # Lambda function wrapps the expression as a "layer" object
        output = ColwiseMultLayer([mean, self.sf_layer])
        # it takes the additional mean layer we created above, and multiplies it with the size factor level. (= None)


# Create the model
        self.model = Model(inputs=[self.input_layer, self.sf_layer], outputs=output)

        self.encoder = self.get_encoder()       
        # keep unscaled output as an extra model
        # self.extra_models['mean_norm'] = Model(inputs=self.input_layer, outputs=mean)
        # self.extra_models['decoded'] = Model(inputs=self.input_layer, outputs=self.decoder_output)



    def save_autoencoder_pickle(self, save_dir):
        if not os.path.exists(save_dir):
            print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Directory...")
            os.makedirs(save_dir)
        
        with open(os.path.join(save_dir, 'model.pickle'), 'wb') as f:
                pickle.dump(self, f)        
  
    
    def save_model(self, save_dir):
        if not os.path.exists(save_dir):
            print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Directory...")
            os.makedirs(save_dir)
            
        print(datetime.now().strftime("%H:%M:%S>"), "saving model as hdf5")
        self.model.save(save_dir + "model.hdf5")     
    
    

    def get_decoder(self):
        print("GET DECODER FUNCTION WAS CALLED")
        i = 0
        for l in self.model.layers:
            if l.name == 'center_drop':
                break
            i += 1
        return Model(inputs=self.model.get_layer(index=i+1).input,
                     outputs=self.model.output)
        
          

    def get_encoder(self, manual_activation=True):      
        if manual_activation:
            ret = Model(inputs=self.model.input,
                        outputs=self.model.get_layer('center_activation').output)
        else:
            ret = Model(inputs=self.model.input,
                        outputs=self.model.get_layer('center').output)
        return ret
        




    def predict(self, adata, mode='denoise'):
       
        assert mode in ('denoise', 'latent', 'full'), 'Unknown mode'
        adata = adata.copy()

        if mode in ('denoise', 'full'):
            print(datetime.now().strftime("%H:%M:%S>"), 'Calculating reconstructions...')

            adata.X = self.model.predict(x = {'count': adata.X,
                                          'size_factors': adata.obs.size_factors})

            adata.uns['dca_loss'] = self.model.test_on_batch(x = {'count': adata.X,
                                                              'size_factors': adata.obs.size_factors},
                                                              y = adata.raw.X) 
            
        if mode in ('latent', 'full'):
            print(datetime.now().strftime("%H:%M:%S>"), 'Calculating low dimensional representations...')
            adata.obsm['latent'] = self.encoder.predict({'count': adata.X,
                                                        'size_factors': adata.obs.size_factors})
            
        # if mode == 'latent':
        #     adata.X = adata.raw.X.copy() #recover normalized expression values
        return adata
        



    def write_output(self, adata, file_path, mode='full', colnames=None):
        
        colnames = adata.var_names.values if colnames is None else colnames
        rownames = adata.obs_names.values
        
        

        print(datetime.now().strftime("%H:%M:%S>"), 'Saving output(s)...')
        os.makedirs(file_path, exist_ok=True)
        
        if mode in ('denoise', 'full'):
            print(datetime.now().strftime("%H:%M:%S>"), 'Saving denoised expression...')
            pd.DataFrame(adata.X, index=rownames, columns=colnames).to_csv(file_path + "denoised_matrix.tsv",
                                                                  sep='\t',
                                                                  index=(rownames is not None),
                                                                  header=(colnames is not None),
                                                                  float_format='%.6f')


        if mode in ('latent', 'full'):
            print(datetime.now().strftime("%H:%M:%S>"), 'Saving latent representations...')
            pd.DataFrame(adata.obsm['latent'], index=rownames).to_csv(file_path + "latent_layer.tsv",
                                                                  sep='\t',
                                                                  index=(rownames is not None),
                                                                  header=(colnames is not None),
                                                                  float_format='%.6f')
            
            pd.DataFrame(adata.obsm['latent'], index=rownames).to_csv(file_path + "matrix.tsv",
                                                                  sep='\t',
                                                                  index=False,
                                                                  header=False,
                                                                  float_format='%.6f')

# %%


#------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------



def train(adata, network, 
          optimizer='RMSprop', 
          learning_rate=None,
          epochs=300,
          reduce_lr=10, 
          use_raw_as_output=False, ###i still don't understand output. I think it is what we compare the end result to, so you can have a slightly fake autoencoder, in which you compare input = X, out'put = raw.X, that is e.g. not normalized or whatever.
          early_stop=35,
          batch_size=32, 
          clip_grad=5.,                 # todo find out effect on optimizer
          validation_split=0.1,     # the fraction of data, that is validation data (on which the loss and model metrics are calculated)
          verbose=False, 
          ):
    
    
    model = network.model
    loss = network.loss


### chose optimizer    
    # this is an important decision. I decided to keep following them for now, but here we could change a lot. 

    if learning_rate is None:
        optimizer = opt.__dict__[optimizer](clipvalue=clip_grad)
    else:
        optimizer = opt.__dict__[optimizer](lr=learning_rate, clipvalue=clip_grad)
    # there is nothing actually happening here.
    # opt.__dict__[optimizer] corresponds to opt.RMSprop here (based on whatever optimizer is passed)
    # if we have a learning rate it gets passed, otherwise the model gets compiled without. (no clue why we can even do that)    
    
    ''' the clipvalue thresholds the gradient, as to avoid exploding (and vanishin?) gradient problem.
    
    remember: W += lr * gradient. 

    learning rate can also be scheduled/ optimized here a lot
    '''
       
    # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate=1e-2,
    #     decay_steps=10000,
    #     decay_rate=0.9)
    # opt = keras.optimizers.SGD(learning_rate=lr_schedule)    
    
    # opt = SGD(lr=1e-2, momentum=0.9, decay=1e-2/epochs)


### compiling        
    model.compile(loss = loss, optimizer = optimizer)

### some extra things
  
    # Callbacks
    callbacks = []

    if reduce_lr:
        lr_cb = ReduceLROnPlateau(monitor='val_loss', patience=reduce_lr, verbose=verbose)      #patientce = how many must plateau
        callbacks.append(lr_cb)
    if early_stop:
        es_cb = EarlyStopping(monitor='val_loss', patience=early_stop, verbose=verbose)
        callbacks.append(es_cb)


    if verbose:
        model.summary()


    # todo
    inputs = {'count': adata.X, 'size_factors': adata.obs.size_factors}      



# size factors = Normalize means by library size

    if use_raw_as_output:
        output = adata.raw.X
    else:
        output = adata.X    # can maybe be omitted. 
        
    loss = model.fit(x = inputs, 
                      y = output,
                      epochs=epochs,
                      batch_size=batch_size,
                      shuffle=True,      # shuffle training data before each epoch
                      callbacks=callbacks, # for early stopping / reduce lr
                      validation_split=validation_split, # the fraction of data, that is validation data (on which the loss and model metrics are calculated)
                      verbose=verbose,
                      #**kwds
                      )

    return loss        
        



def plot_history(adata, output_dir = "./"):
    
    assert isinstance(adata, anndata.AnnData), 'adata must be an AnnData instance'
    
    
    import matplotlib.pyplot as plt
    
    
    ploss = adata.uns['train_history']["loss"]
    plr = adata.uns['train_history']["lr"]
    pval_loss = adata.uns['train_history']["val_loss"]
    
    num_epochs = len(ploss)



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





# %% Main


def sca(adata_train, 
        adata_test,
        loss_name,
        
        mode = "full",
        ae_type = "normal",
        
        # training args
        epochs = 300,
        reduce_lr = 10,
        early_stop = 35,
        batch_size = 32,
        optimizer = "RMSprop",
        batchnorm = True,
        learning_rate = None,
        random_state = 0,
        verbose = True,
        threads = None,
        
        output_dir = ("./sca_output/")
        ):
    
    

    ## input checker    
    assert isinstance(adata_train, anndata.AnnData), 'adata must be an AnnData instance'
    assert mode in ('denoise', 'latent', 'full'), '%s is not a valid mode.' % mode
    


## do stuff
    input_size = adata_train.n_vars
    ae = Autoencoder(input_size = input_size, 
                     output_size = input_size, 
                     hidden_size = (64, 32, 64),
                     hidden_dropout = 0.00001, 
                     input_dropout = 0.0001, 
                     batchnorm = batchnorm,
                     initializer = 'glorot_uniform',
                     regularizer = None,
                     activation = "relu")

    # ae.save("./saved_aes")
    
    ae.loss_name = loss_name

    ae.build()





    hist = train(adata_train[adata_train.obs.dca_split == 'train'], 
                 network = ae, 
                  epochs = epochs, 
                  reduce_lr = reduce_lr, 
                  early_stop = early_stop, 
                  batch_size = batch_size, 
                  optimizer = optimizer, 
                  verbose = verbose, 
                  learning_rate = learning_rate)

    denoised_train = ae.predict(adata = adata_train, mode = mode)
    denoised_test = ae.predict(adata = adata_test, mode = mode)
    #def predict(self, adata, mode='denoise', return_info=False, copy=False):
    
    '''denoise now contains:
        n_obs:              (input), cells x dca_split x size_factors
        n_obsm['latent']:   cells x latent layer: this is the latent representation
        raw.X:              the original count matrix representation
        X:                  the reconstructed "count" matrix
        var = raw.var:      the original vars, =  #genes x 0 dataframe
        uns["dca_loss"]     the single (final?) loss value. Actually, this is the result (=loss) of a test of the model on a single batch
        uns.overloaded["neighbors"]     This was done by scanpy. I can call sc.pp.neighbors(adata), and I think this would add distances and connectivities...?
        uns["train_history"]    contains the evolution of lr, loss and val_loss
    '''
    
    adata_train = denoised_train
    adata_test = denoised_test
    
    # add loss history
    adata_train.uns['train_history'] = hist.history
    
    os.makedirs(output_dir, exist_ok=True)
    plot_history(adata_train, output_dir)
    
    
    ae.write_output(adata = adata_train, file_path = output_dir + "train_data/", mode = "full")
    ae.write_output(adata = adata_test, file_path = output_dir + "test_data/", mode = "full")
    
    return (adata_train, adata_test, ae)






def sca_preprocess(adata, test_split = False, filter_ = True, size_factors = True, logtrans = True, normalize = True):
    
    if test_split:
        train_idx, test_idx = train_test_split(np.arange(adata.n_obs), test_size=0.1, random_state=42)
        spl = pd.Series(['train'] * adata.n_obs)        # first make all train, then overwrite the tests
        spl.iloc[test_idx] = 'test'
        adata.obs['dca_split'] = spl.values
    else:
        adata.obs['dca_split'] = 'train'

    if filter_:
        # filter min coutns
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)
    
    adata.raw = adata.copy()

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0
 
    if logtrans:
        sc.pp.log1p(adata)
            
    if normalize:
        sc.pp.scale(adata)

    return adata
    




def read_input(input_dir, output_dir):
    
    data = np.loadtxt(open(input_dir + "matrix.tsv"), delimiter="\t")
    genes = pd.read_csv(input_dir + "genes.tsv", delimiter = "\t", header = None)
    barcodes = pd.read_csv(input_dir + "barcodes.tsv", delimiter = "\t", header = None)
    
    test_index = np.loadtxt(fname = input_dir + "test_index.tsv", dtype = bool)
    train_index = np.logical_not(test_index)
    
    
    testdata = data[test_index]
    traindata = data[train_index]
    
    adata_train = sc.AnnData(traindata)
    adata_train.obs_names = barcodes.iloc[train_index,0] + "_" + barcodes.iloc[train_index,1]
    adata_train.var_names = genes.iloc[:,0]

    adata_test = sc.AnnData(testdata)
    adata_test.obs_names = barcodes.iloc[test_index,0] + "_" + barcodes.iloc[test_index,1]
    adata_test.var_names = genes.iloc[:,0]
    
    
    
    nonzero_genes, _ = sc.pp.filter_genes(adata_test.X, min_counts=1)
    assert nonzero_genes.all(), 'Please remove all-zero genes before using DCA. Test'
    nonzero_genes, _ = sc.pp.filter_genes(adata_train.X, min_counts=1)
    assert nonzero_genes.all(), 'Please remove all-zero genes before using DCA. Train'

    return adata_train, adata_test, genes, barcodes
    






def sca_main(input_dir, output_dir, loss_name):
    
    
    # generate AnnData
    adata_train, adata_test, genes, barcodes = read_input(input_dir = input_dir, output_dir = output_dir)    


    # check if observations are unnormalized using first 10
    # X_subset = adata.X[:10]
    # norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    # if sp.sparse.issparse(X_subset):
    #     assert (X_subset.astype(int) != X_subset).nnz == 0, norm_error          # i'm not entirely sure what exactly this proves.
    # else:
    #     assert np.all(X_subset.astype(int) == X_subset), norm_error


    adata_train = sca_preprocess(adata_train, 
                   test_split = False, 
                   filter_ = True,
                   size_factors = True,
                   logtrans = True,
                   normalize = True
                   )
    adata_test = sca_preprocess(adata_test, 
                   test_split = False, 
                   filter_ = True,
                   size_factors = True,
                   logtrans = True,
                   normalize = True
                   )
    

    
    adata_train, adata_test, net = sca(adata_train = adata_train,
                        adata_test = adata_test,
                        loss_name = loss_name,
                        
                        mode = "full",
                        ae_type = "normal",
                        
                        # training args
                        epochs = 300,
                        reduce_lr = 10,
                        early_stop = 15,
                        batch_size = 32,
                        batchnorm = True,
                        optimizer = "RMSprop",
                        learning_rate = None,
                        random_state = 0,
                        verbose = True,
                        threads = None,
                        
                        output_dir = (output_dir)
                        )
    
    net.save_model(output_dir)


    # transfer genes and barcodes (only to test data, but its identical for train. Just no point putting it in there as well. )
    genes.to_csv(output_dir + "test_data/genes.tsv", sep = "\t", index = False, header = False)
    barcodes.to_csv(output_dir + "test_data/barcodes.tsv", sep = "\t", index = False, header = False)






# %%
#------------------------------------------------------------------------------
# Call program
#------------------------------------------------------------------------------

import pandas as pd
import scanpy as sc # import AnnData
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    
    sca_main(input_dir = args.input_dir,  
             output_dir = args.output_dir,
             loss_name = args.loss)
 