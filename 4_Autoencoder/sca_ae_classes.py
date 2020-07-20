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



import numpy as np


from keras.layers import Lambda
import tensorflow as tf




# %%




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
        
        
        self.input_layer = Input(shape=(self.input_size,), name='input_layer-s')
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




        # keep unscaled output as an extra model
        self.extra_models['mean_norm'] = Model(inputs=self.input_layer, outputs=mean)
        self.extra_models['decoded'] = Model(inputs=self.input_layer, outputs=self.decoder_output)
        self.model = Model(inputs=[self.input_layer, self.sf_layer], outputs=output)



        self.encoder = self.get_encoder()









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
        print("asdf")




    def write(self, adata, file_path, mode='denoise', colnames=None):
        print("asdf")
        

##



#------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------






def sca(adata, 
        MODE = "denoise",
        AE_TYPE = "normal",
        
        
        
        # training args
        epochs = 300,
        reduce_lr = 10,
        early_stop = 15,
        batch_size = 32,
        optimizer = "rmsprop",
        learning_rate = None,
        random_state = 0,
        verbose = False,
        threads = None
        ):

    ae = Autoencoder(input_size = 2000, output_size = 2000)
    
    ae.save("./saved_aes")
    
    ae.build()


    # hist = train(adata[adata.obs.dca_split == 'train'], ae, 
    #              epochs = epochs, 
    #              reduce_lr = reduce_lr, 
    #              early_stop = early_stop, 
    #              batch_size = batch_size, 
    #              optimizer = optimizer, 
    #              verbose = verbose, 
    #              threads = threads, 
    #              learning_rate = learning_rate)






#------------------------------------------------------------------------------
# Main program
#------------------------------------------------------------------------------



if __name__ == "__main__":
    sca




