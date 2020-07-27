# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 17:22:21 2020

@author: Mike Toreno II
"""








import os
import pickle
from datetime import datetime



import keras
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization, Lambda
from keras.models import Model
from keras.regularizers import l1_l2
from keras.objectives import mean_squared_error
from keras.initializers import Constant
from keras import backend as K
from keras.callbacks import EarlyStopping, ReduceLROnPlateau



import numpy as np
import anndata

from keras.layers import Lambda
import tensorflow as tf




# %%
import keras.optimizers as opt



globi = None





# %%

class Autoencoder():
    def __init__(self,
                 input_size,
                 output_size = None,
                 hidden_size=(64, 32, 64),
                 hidden_dropout= 0,
                 #hidden_dropout= (0, 0, 0),
                 input_dropout = 0,
                 initializer = 'glorot_uniform',
                 regularizer = None,
                 activation = "relu"):
                  
        
        self.input_size = input_size 
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.hidden_dropout= hidden_dropout
        self.input_dropout= input_dropout
        
        self.initializer = initializer
        self.regularizer = regularizer
        self.activation = activation
## end of inputparsing

    

        self.sf_layer = None        # I HAVE NO CLUE WHAT THIS IS. sIZE FACTOR LAYER, HAS SHAPE (1,)
        self.input_layer = None 

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
        
        
        self.input_layer = Input(shape=(self.input_size,), name='input_layer-s_myname')
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
        print(datetime.now().strftime("%H:%M:%S>"), "Building output with loss function: mean_squared_error...")
        
        self.loss = mean_squared_error



        # we apply again a Dense over our last hidden layer        
        mean = Dense(self.output_size, 
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


        self.model = Model(inputs=[self.input_layer, self.sf_layer], outputs=output)

        self.encoder = self.get_encoder()
        
        # keep unscaled output as an extra model
        # self.extra_models['mean_norm'] = Model(inputs=self.input_layer, outputs=mean)
        # self.extra_models['decoded'] = Model(inputs=self.input_layer, outputs=self.decoder_output)
        



    def save(self, save_dir):
        if not os.path.exists(save_dir):
            print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Directory...")
            os.makedirs(save_dir)
        
        with open(os.path.join(save_dir, 'model.pickle'), 'wb') as f:
                pickle.dump(self, f)        
  
  
    
  
    def load_weights(self, filename):
        print("asdf") 
        
    def get_decoder(self):
        print("asdf")
        

    def get_encoder(self, activation=False):
        if activation:
            ret = Model(inputs=self.model.input,
                        outputs=self.model.get_layer('center_act').output)
        else:
            ret = Model(inputs=self.model.input,
                        outputs=self.model.get_layer('center').output)
        return ret
        





    def predict(self, adata, mode='denoise', return_info=False, copy=False):
       
        assert mode in ('denoise', 'latent', 'full'), 'Unknown mode'

        adata = adata.copy() if copy else adata

        if mode in ('denoise', 'full'):
            print('dca: Calculating reconstructions...')

            adata.X = self.model.predict({'count': adata.X,
                                          'size_factors': adata.obs.size_factors})

            adata.uns['dca_loss'] = self.model.test_on_batch({'count': adata.X,
                                                              'size_factors': adata.obs.size_factors},
                                                             adata.raw.X)
        if mode in ('latent', 'full'):
            print('dca: Calculating low dimensional representations...')

            adata.obsm['X_dca'] = self.encoder.predict({'count': adata.X,
                                                        'size_factors': adata.obs.size_factors})
        if mode == 'latent':
            adata.X = adata.raw.X.copy() #recover normalized expression values

        return adata if copy else None




    def write(self, adata, file_path, mode='denoise', colnames=None):
        print("asdf")
        

# %%

#------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------



def train(adata, network, 
          output_dir="./_aa_output_dir_deleteme", 
          optimizer='rmsprop', 
          learning_rate=None,
          epochs=300,
          reduce_lr=10, 
          use_raw_as_output=False, ###i still don't understand output
          early_stop=15,
          batch_size=32, 
          clip_grad=5.,                 # todo find out effect on optimizer
          validation_split=0.1,     # the fraction of data, that is validation data (on which the loss and model metrics are calculated)
          verbose=False, 
          ):
    
    model = network.model
    loss = network.loss
    
    os.makedirs(output_dir, exist_ok=True)
    

### chose optimizer    
    # this is an important decision. I decided to keep following them for now, but here we could change a lot. 

    if learning_rate is None:
        optimizer = opt.__dict__[optimizer](clipvalue=clip_grad)
    else:
        optimizer = opt.__dict__[optimizer](lr=learning_rate, clipvalue=clip_grad)
    # there is nothing actually happening here.
    # opt.__dict__[optimizer] corresponds to opt.RMSprop here (based on whatever optimizer is passed)
    # if we have a learning rate it gets passed, otherwise the model gets compiled without. (no clue why we can even do that)    
        
    ''' remember: W += lr * gradient. 
    we can decrease it over time though. 
    
    e.g. standard decay: every batch (num training samples / batch size) = steps per epoch
    
    however, they don't specify their learning rate yet here, so I am gonna follow them for now. 
    
    But here is an interesting section, that we could change in many ways. '
    some examples, for SDG
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
        lr_cb = ReduceLROnPlateau(monitor='val_loss', patience=reduce_lr, verbose=verbose)
        callbacks.append(lr_cb)
    if early_stop:
        es_cb = EarlyStopping(monitor='val_loss', patience=early_stop, verbose=verbose)
        callbacks.append(es_cb)


    if verbose:
        model.summary()

    # todo
    # inputs = {'count': adata.X, 'size_factors': adata.obs.size_factors}      ### it's part of the adata read in pipeline, seems to be something like normalized per cell data       
    inputs = {'count': adata.X}      ### it's part of the adata read in pipeline, seems to be something like normalized per cell data       

# size factors = Normalize means by library size



    print("inputs is of type:")
    print(type(inputs["count"]))    

    
    if use_raw_as_output:
        output = adata.raw.X
    else:
        output = adata.X    # can maybe be omitted. 
        
    print("output is of type:")
    print(type(output))    
        
    

    global globi
    globi = inputs   
    
    
    
    
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
        





# %% Main



def sca(adata, 
        mode = "denoise",
        ae_type = "normal",
        
        
        # training args
        epochs = 300,
        reduce_lr = 10,
        early_stop = 15,
        batch_size = 32,
        optimizer = "rmsprop",
        learning_rate = None,
        random_state = 0,
        verbose = True,
        threads = None
        ):
    
    
## input checker    
    
    assert isinstance(adata, anndata.AnnData), 'adata must be an AnnData instance'
    assert mode in ('denoise', 'latent'), '%s is not a valid mode.' % mode
    
    
    nonzero_genes, _ = sc.pp.filter_genes(adata.X, min_counts=1)
    assert nonzero_genes.all(), 'Please remove all-zero genes before using DCA.'
    
    
## do stuff
    ae = Autoencoder(input_size = 73926, output_size = 73926)
    
    # ae.save("./saved_aes")
    
    ae.build()
    


    hist = train(adata[adata.obs.dca_split == 'train'], ae, 
                  epochs = epochs, 
                  reduce_lr = reduce_lr, 
                  early_stop = early_stop, 
                  batch_size = batch_size, 
                  optimizer = optimizer, 
                  verbose = verbose, 
                  learning_rate = learning_rate)
    
    
    return hist
    
    
    # to implement test split at some point
    #     train_idx, test_idx = train_test_split(np.arange(adata.n_obs), test_size=0.1, random_state=42)
    #     spl = pd.Series(['train'] * adata.n_obs)
    #     spl.iloc[test_idx] = 'test'
    #     adata.obs['dca_split'] = spl.values
    




# %%
#------------------------------------------------------------------------------
# Call program
#------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import scanpy as sc # import AnnData
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    
    # generate AnnData
    input_dir = "../inputs/dca/toydata/"
    
    data = np.loadtxt(open(input_dir + "countmatrix.tsv"), delimiter="\t")
    genes = pd.read_csv(input_dir + "genes.tsv", delimiter = "\t", header = None)
    barcodes = pd.read_csv(input_dir + "barcodes.tsv", delimiter = "\t", header = None)
    
    
    adata = sc.AnnData(data)
    
    adata.obs_names = barcodes.iloc[:,0]
    adata.var_names = genes.iloc[:,0]
    
    
    
    # I don't know if the train-split is even necessary
    train_idx, test_idx = train_test_split(np.arange(adata.n_obs), test_size=0.1, random_state=42)
    spl = pd.Series(['train'] * adata.n_obs)    # initialize with all train  
    spl.iloc[test_idx] = 'test'             
    adata.obs['dca_split'] = spl.values    
    
    
    
    import pickle
    file = open("D:/Dropbox/Internship/gitrepo/inputs/dca/toydata/adata_ae.obj", "rb")
    obi = pickle.load(file)
    adata = obi
    
    exploreobject = sca(adata = adata)






# %%





























