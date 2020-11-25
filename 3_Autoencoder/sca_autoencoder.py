# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 17:22:21 2020

@author: Mike Toreno II
"""


    # AE_types = {'': Autoencoder, '': PoissonAutoencoder,
    #         '': NBConstantDispAutoencoder, 'nb-conddisp': NBAutoencoder,
    #         'nb-shared': NBSharedAutoencoder, 'nb-fork': NBForkAutoencoder,
    #         'zinb': ZINBConstantDispAutoencoder, 'zinb-conddisp': ZINBAutoencoder,
    #         'zinb-shared': ZINBSharedAutoencoder, 'zinb-fork': ZINBForkAutoencoder,
    #         'zinb-elempi': ZINBAutoencoderElemPi}    




import argparse

parser = argparse.ArgumentParser(description = "program to preprocess the raw singlecell data")  
parser.add_argument("-i","--input_dir", help="input directory", default = "../inputs/data/preprocessed_data_autoencoder/")
parser.add_argument("-o","--output_dir", help="output directory", default = "../inputs/autoencoder_data/SCA_output/")
parser.add_argument("-p","--outputplot_dir", help="plot directory", default = "../outputs/autoencoder_data/SCA/")
parser.add_argument("--loss", default = "mse", type = str, choices = ["poisson_loss", "poisson", "mse","mae","mape","msle","squared_hinge","hinge","binary_crossentropy","categorical_crossentropy","kld","cosine_proximity"])
parser.add_argument("--mode", default = "complete", help="chose k-split, unsplit or both", choices=['complete','split','nosplit'])
parser.add_argument("--splitnumber", type = int, help="in order to run all splits at the same time, they can be run individually. If mode == split, enter a number here to only do that split. Please ensure that the split exists. ")
parser.add_argument("--AEtype", default = "nb-conddisp", choices=['normal','poisson','nb','nb-shared','nb-conddisp','nb-fork','zinb','zinb-shared','zinb-elempi','zinb-conddisp','zinb-fork'])
#parser.add_argument("--verbose", type = int, default = 2, help="0: quiet, 1:progress bar, 2:1 line per epoch") 
# whatever, verbosity is always 2 now. (I do this, because of the other "verbose" variable floating around - let's keep it simple. )

args = parser.parse_args()



# args.input_dir = "M:/Projects/simon_streib_internship/sc-autoencoding/inputs/data/preprocessed_data_autoencoder/"




import pandas as pd
import scanpy as sc # import AnnData
#from sklearn.model_selection import train_test_split


# THE EXTRA IMPORTS
############################################################################################################################################################
import keras
from keras.regularizers import l1_l2
  
# THE EXTRA IMPORTS
############################################################################################################################################################



import os
import sys
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
        
    
    
    
    
    
# EXPERIMENTAL SOLUTION TO THE PROBLEM THAT THE CODE DOESN'T WORK ON SERVER, BUT WORKS ON LOCAL MACHINE
#tf.config.experimental_run_functions_eagerly(True)
tf.compat.v1.disable_eager_execution()
# THIS ONE DOES IT. SO FUCK IT. 








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
        
        self.init = initializer
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
            
            
            
            
##########################################################################################################################################################
# some exra stuff for compatibility
        self.l2_coef = 0
        self.l1_coef = 0
        self.l2_enc_coef = 0
        self.l1_enc_coef = 0
        self.ridge = 0
        self.loss = None
        self.file_path = None
        self.decoder = None
        self.debug = None
        
            

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
                        kernel_initializer = self.init,
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
                     kernel_initializer=self.init,
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
  
    
    def save_model(self, save_dir, *args, **kwargs):
        # if not os.path.exists(save_dir):
        #     print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Directory...")
        #     os.makedirs(save_dir)
            
        # print(datetime.now().strftime("%H:%M:%S>"), "saving model as hdf5")
        # self.model.save(save_dir + "model.hdf5")     
        print("model was not saved, because I don't want to.")
    
    

    def get_decoder(self):
        print("GET DECODER FUNCTION WAS CALLED")
        i = 0
        for l in self.model.layers:
            if l.name == 'center_drop':
                break
            i += 1
        return Model(inputs=self.model.get_layer(index=i+1).input,
                     outputs=self.model.output)
        
          

    def get_encoder(self, activation=False):
        if activation:
            ret = Model(inputs=self.model.input,
                        outputs=self.model.get_layer('center_act').output)
        else:
            ret = Model(inputs=self.model.input,
                        outputs=self.model.get_layer('center').output)
        return ret




    def predict(self, adata, mode='denoise', return_info = False, copy = False, colnames = None):
       
        assert mode in ('denoise', 'latent', 'full'), 'Unknown mode'
        # adata = adata.copy()

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
        



    def write_output_deprecated(self, adata, file_path, mode='full', colnames=None):
        print("PLEASE DON'T USE THIS FUNCTION")
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




    def write_output(self, test_adata, train_adata, test_index, file_path, mode='full', colnames=None):
        
        if test_adata is not None:
            train_index = np.logical_not(test_index)
        else:
            train_index = None
        
        # to give the columns and rows names. I did it before, but why bother?
        # colnames = test_adata.var_names.values if colnames is None else colnames
        # rownames_test = test_adata.obs_names.values
        # rownames_train = train_adata.obs_names.values


        print(datetime.now().strftime("%H:%M:%S>"), 'Saving output(s)...')
        os.makedirs(file_path, exist_ok=True)
        
        global globi
        globi = train_adata
        

        
        if mode in ('denoise', 'full'):
            print(datetime.now().strftime("%H:%M:%S>"), 'Saving denoised expression...')
            
            
            if test_adata is not None:
                denoised_traindata = train_adata.X
                denoised_testdata = test_adata.X
                
                denoised_outdata = np.zeros(shape = (denoised_testdata.shape[0] + denoised_traindata.shape[0], denoised_testdata.shape[1]))
                denoised_outdata[train_index] = denoised_traindata
                denoised_outdata[test_index] = denoised_testdata       
                
                pd.DataFrame(denoised_outdata).to_csv(file_path + "denoised_matrix.tsv",
                                                                    sep='\t',
                                                                    index=None,
                                                                    header=None,
                                                                    float_format='%.6f')
            else:
                denoised_traindata = train_adata.X
                
                pd.DataFrame(denoised_traindata).to_csv(file_path + "denoised_matrix.tsv",
                                                    sep='\t',
                                                    index=None,
                                                    header=None,
                                                    float_format='%.6f')


        if mode in ('latent', 'full'):
            print(datetime.now().strftime("%H:%M:%S>"), 'Saving latent representations...')
            
            if test_adata is not None:
                latent_testdata = test_adata.obsm['latent']
                latent_traindata = train_adata.obsm['latent']
        
                latent_outdata = np.zeros(shape = (latent_testdata.shape[0] + latent_traindata.shape[0], latent_traindata.shape[1]))
                latent_outdata[train_index] = latent_traindata
                latent_outdata[test_index] = latent_testdata                
                
                pd.DataFrame(latent_outdata).to_csv(file_path + "latent_layer.tsv",
                                                                    sep='\t',
                                                                    index=None,
                                                                    header=None,
                                                                    float_format='%.6f')
                
                pd.DataFrame(latent_outdata).to_csv(file_path + "matrix.tsv",
                                                                    sep='\t',
                                                                    index=None,
                                                                    header=None,
                                                                    float_format='%.6f')
            else:
                latent_data = train_adata.obsm['latent']
                
                pd.DataFrame(latent_data).to_csv(file_path + "latent_layer.tsv",
                                                    sep='\t',
                                                    index=None,
                                                    header=None,
                                                    float_format='%.6f')
                
                pd.DataFrame(latent_data).to_csv(file_path + "matrix.tsv",
                                                    sep='\t',
                                                    index=None,
                                                    header=None,
                                                    float_format='%.6f')
        print(self.loss)    
            
            
            
            
            
            
            
######################################################################################################################################################################################################################################################################
# end of superclass AE


# MIND THERE ARE A LOT OF IMPORTS FOR THIS


class PoissonAutoencoder(Autoencoder):

    def build_output(self):
        mean = Dense(self.output_size, activation=MeanAct, kernel_initializer=self.init,
                     kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                     name='mean')(self.decoder_output)
        output = ColwiseMultLayer([mean, self.sf_layer])
        self.loss = poisson_loss

        self.extra_models['mean_norm'] = Model(inputs=self.input_layer, outputs=mean)
        self.extra_models['decoded'] = Model(inputs=self.input_layer, outputs=self.decoder_output)
        
        self.model = Model(inputs=[self.input_layer, self.sf_layer], outputs=output)

        self.encoder = self.get_encoder()


class NBConstantDispAutoencoder(Autoencoder):

    def build_output(self):
        mean = Dense(self.output_size, activation=MeanAct, kernel_initializer=self.init,
                     kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                     name='mean')(self.decoder_output)

        # Plug in dispersion parameters via fake dispersion layer
        disp = ConstantDispersionLayer(name='dispersion')
        mean = disp(mean)

        output = ColwiseMultLayer([mean, self.sf_layer])

        nb = NB(disp.theta_exp)
        self.loss = nb.loss
        self.extra_models['dispersion'] = lambda :K.function([], [nb.theta])([])[0].squeeze()
        self.extra_models['mean_norm'] = Model(inputs=self.input_layer, outputs=mean)
        self.extra_models['decoded'] = Model(inputs=self.input_layer, outputs=self.decoder_output)
        self.model = Model(inputs=[self.input_layer, self.sf_layer], outputs=output)

        self.encoder = self.get_encoder()

    def predict(self, adata, mode='denoise', return_info=False, copy=False):
        colnames = adata.var_names.values
        rownames = adata.obs_names.values
        res = super().predict(adata, mode, return_info, copy)
        adata = res

        if return_info:
            adata.var['X_dca_dispersion'] = self.extra_models['dispersion']()

        return adata

    def write(self, adata, file_path, mode='denoise', colnames=None):
        colnames = adata.var_names.values if colnames is None else colnames
        rownames = adata.obs_names.values

        super().write(adata, file_path, mode, colnames=colnames)
        if 'X_dca_dispersion' in adata.var_keys():
            write_text_matrix(adata.var['X_dca_dispersion'].reshape(1, -1),
                              os.path.join(file_path, 'dispersion.tsv'),
                              colnames=colnames, transpose=True)


class NBAutoencoder(Autoencoder):

    def build_output(self):
        disp = Dense(self.output_size, activation=DispAct,
                           kernel_initializer=self.init,
                           kernel_regularizer=l1_l2(self.l1_coef,
                               self.l2_coef),
                           name='dispersion')(self.decoder_output)

        mean = Dense(self.output_size, activation=MeanAct, kernel_initializer=self.init,
                       kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                       name='mean')(self.decoder_output)
        output = ColwiseMultLayer([mean, self.sf_layer])
        output = SliceLayer(0, name='slice')([output, disp])

        nb = NB(theta=disp, debug=self.debug)
        self.loss = nb.loss
        self.extra_models['dispersion'] = Model(inputs=self.input_layer, outputs=disp)
        self.extra_models['mean_norm'] = Model(inputs=self.input_layer, outputs=mean)
        self.extra_models['decoded'] = Model(inputs=self.input_layer, outputs=self.decoder_output)

        self.model = Model(inputs=[self.input_layer, self.sf_layer], outputs=output)

        self.encoder = self.get_encoder()

    def predict(self, adata, mode='denoise', return_info=False, copy=False):
        colnames = adata.var_names.values
        rownames = adata.obs_names.values

        res = super().predict(adata, mode, return_info, copy)
        adata = res

        if return_info:
            adata.obsm['X_dca_dispersion'] = self.extra_models['dispersion'].predict(adata.X)

        return adata

    def write(self, adata, file_path, mode='denoise', colnames=None):
        colnames = adata.var_names.values if colnames is None else colnames
        rownames = adata.obs_names.values

        super().write(adata, file_path, mode, colnames=colnames)

        if 'X_dca_dispersion' in adata.obsm_keys():
            write_text_matrix(adata.obsm['X_dca_dispersion'],
                              os.path.join(file_path, 'dispersion.tsv'),
                              colnames=colnames, transpose=True)

class NBSharedAutoencoder(NBAutoencoder):

    def build_output(self):
        disp = Dense(1, activation=DispAct,
                     kernel_initializer=self.init,
                     kernel_regularizer=l1_l2(self.l1_coef,
                                              self.l2_coef),
                     name='dispersion')(self.decoder_output)

        mean = Dense(self.output_size, activation=MeanAct, kernel_initializer=self.init,
                       kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                       name='mean')(self.decoder_output)
        output = ColwiseMultLayer([mean, self.sf_layer])
        output = SliceLayer(0, name='slice')([output, disp])

        nb = NB(theta=disp, debug=self.debug)
        self.loss = nb.loss
        self.extra_models['dispersion'] = Model(inputs=self.input_layer, outputs=disp)
        self.extra_models['mean_norm'] = Model(inputs=self.input_layer, outputs=mean)
        self.extra_models['decoded'] = Model(inputs=self.input_layer, outputs=self.decoder_output)

        self.model = Model(inputs=[self.input_layer, self.sf_layer], outputs=output)
        self.encoder = self.get_encoder()


class ZINBAutoencoder(Autoencoder):

    def build_output(self):
        pi = Dense(self.output_size, activation='sigmoid', kernel_initializer=self.init,
                       kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                       name='pi')(self.decoder_output)

        disp = Dense(self.output_size, activation=DispAct,
                           kernel_initializer=self.init,
                           kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                           name='dispersion')(self.decoder_output)

        mean = Dense(self.output_size, activation=MeanAct, kernel_initializer=self.init,
                       kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                       name='mean')(self.decoder_output)
        output = ColwiseMultLayer([mean, self.sf_layer])
        output = SliceLayer(0, name='slice')([output, disp, pi])

        zinb = ZINB(pi, theta=disp, ridge_lambda=self.ridge, debug=self.debug)
        self.loss = zinb.loss
        self.extra_models['pi'] = Model(inputs=self.input_layer, outputs=pi)
        self.extra_models['dispersion'] = Model(inputs=self.input_layer, outputs=disp)
        self.extra_models['mean_norm'] = Model(inputs=self.input_layer, outputs=mean)
        self.extra_models['decoded'] = Model(inputs=self.input_layer, outputs=self.decoder_output)

        self.model = Model(inputs=[self.input_layer, self.sf_layer], outputs=output)

        self.encoder = self.get_encoder()

    def predict(self, adata, mode='denoise', return_info=False, copy=False, colnames=None):

        if return_info:
            adata.obsm['X_dca_dispersion'] = self.extra_models['dispersion'].predict(adata.X)
            adata.obsm['X_dca_dropout']    = self.extra_models['pi'].predict(adata.X)

        # warning! this may overwrite adata.X
        super().predict(adata, mode, return_info, copy=False)
        return adata

    def write(self, adata, file_path, mode='denoise', colnames=None):
        colnames = adata.var_names.values if colnames is None else colnames
        rownames = adata.obs_names.values

        super().write(adata, file_path, mode, colnames=colnames)

        if 'X_dca_dispersion' in adata.obsm_keys():
            write_text_matrix(adata.obsm['X_dca_dispersion'],
                              os.path.join(file_path, 'dispersion.tsv'),
                              colnames=colnames, transpose=True)

        if 'X_dca_dropout' in adata.obsm_keys():
            write_text_matrix(adata.obsm['X_dca_dropout'],
                              os.path.join(file_path, 'dropout.tsv'),
                              colnames=colnames, transpose=True)


class ZINBAutoencoderElemPi(ZINBAutoencoder):
    def __init__(self, sharedpi=False, **kwds):
        super().__init__(**kwds)
        self.sharedpi = sharedpi

    def build_output(self):
        disp = Dense(self.output_size, activation=DispAct,
                           kernel_initializer=self.init,
                           kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                           name='dispersion')(self.decoder_output)

        mean_no_act = Dense(self.output_size, activation=None, kernel_initializer=self.init,
                       kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                       name='mean_no_act')(self.decoder_output)

        minus = Lambda(lambda x: -x)
        mean_no_act = minus(mean_no_act)
        pidim = self.output_size if not self.sharedpi else 1

        pi = ElementwiseDense(pidim, activation='sigmoid', kernel_initializer=self.init,
                       kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                       name='pi')(mean_no_act)

        mean = Activation(MeanAct, name='mean')(mean_no_act)

        output = ColwiseMultLayer([mean, self.sf_layer])
        output = SliceLayer(0, name='slice')([output, disp, pi])

        zinb = ZINB(pi, theta=disp, ridge_lambda=self.ridge, debug=self.debug)
        self.loss = zinb.loss
        self.extra_models['pi'] = Model(inputs=self.input_layer, outputs=pi)
        self.extra_models['dispersion'] = Model(inputs=self.input_layer, outputs=disp)
        self.extra_models['mean_norm'] = Model(inputs=self.input_layer, outputs=mean)
        self.extra_models['decoded'] = Model(inputs=self.input_layer, outputs=self.decoder_output)

        self.model = Model(inputs=[self.input_layer, self.sf_layer], outputs=output)

        self.encoder = self.get_encoder()



class ZINBSharedAutoencoder(ZINBAutoencoder):

    def build_output(self):
        pi = Dense(1, activation='sigmoid', kernel_initializer=self.init,
                   kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                   name='pi')(self.decoder_output)

        disp = Dense(1, activation=DispAct,
                     kernel_initializer=self.init,
                     kernel_regularizer=l1_l2(self.l1_coef,
                                              self.l2_coef),
                     name='dispersion')(self.decoder_output)

        mean = Dense(self.output_size, activation=MeanAct, kernel_initializer=self.init,
                       kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                       name='mean')(self.decoder_output)
        output = ColwiseMultLayer([mean, self.sf_layer])
        output = SliceLayer(0, name='slice')([output, disp, pi])

        zinb = ZINB(pi, theta=disp, ridge_lambda=self.ridge, debug=self.debug)
        self.loss = zinb.loss
        self.extra_models['pi'] = Model(inputs=self.input_layer, outputs=pi)
        self.extra_models['dispersion'] = Model(inputs=self.input_layer, outputs=disp)
        self.extra_models['mean_norm'] = Model(inputs=self.input_layer, outputs=mean)
        self.extra_models['decoded'] = Model(inputs=self.input_layer, outputs=self.decoder_output)

        self.model = Model(inputs=[self.input_layer, self.sf_layer], outputs=output)

        self.encoder = self.get_encoder()


class ZINBConstantDispAutoencoder(Autoencoder):

    def build_output(self):
        pi = Dense(self.output_size, activation='sigmoid', kernel_initializer=self.init,
                   kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                   name='pi')(self.decoder_output)

        mean = Dense(self.output_size, activation=MeanAct, kernel_initializer=self.init,
                     kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                     name='mean')(self.decoder_output)

        # NB dispersion layer
        disp = ConstantDispersionLayer(name='dispersion')
        mean = disp(mean)

        output = ColwiseMultLayer([mean, self.sf_layer])

        zinb = ZINB(pi, theta=disp.theta_exp, ridge_lambda=self.ridge, debug=self.debug)
        self.loss = zinb.loss
        self.extra_models['pi'] = Model(inputs=self.input_layer, outputs=pi)
        self.extra_models['dispersion'] = lambda :K.function([], [zinb.theta])([])[0].squeeze()
        self.extra_models['mean_norm'] = Model(inputs=self.input_layer, outputs=mean)
        self.extra_models['decoded'] = Model(inputs=self.input_layer, outputs=self.decoder_output)

        self.model = Model(inputs=[self.input_layer, self.sf_layer], outputs=output)

        self.encoder = self.get_encoder()

    def predict(self, adata, mode='denoise', return_info=False, copy=False):
        colnames = adata.var_names.values
        rownames = adata.obs_names.values

        if return_info:
            adata.var['X_dca_dispersion'] = self.extra_models['dispersion']()
            adata.obsm['X_dca_dropout']    = self.extra_models['pi'].predict(adata.X)

        super().predict(adata, mode, return_info, copy=False)
        return adata

    def write(self, adata, file_path, mode='denoise', colnames=None):
        colnames = adata.var_names.values if colnames is None else colnames
        rownames = adata.obs_names.values

        super().write(adata, file_path, mode)

        if 'X_dca_dispersion' in adata.var_keys():
            write_text_matrix(adata.var['X_dca_dispersion'].reshape(1, -1),
                              os.path.join(file_path, 'dispersion.tsv'),
                              colnames=colnames, transpose=True)

        if 'X_dca_dropout' in adata.obsm_keys():
            write_text_matrix(adata.obsm['X_dca_dropout'],
                              os.path.join(file_path, 'dropout.tsv'),
                              colnames=colnames, transpose=True)


class ZINBForkAutoencoder(ZINBAutoencoder):

    def build(self):

        self.input_layer = Input(shape=(self.input_size,), name='count')
        self.sf_layer = Input(shape=(1,), name='size_factors')
        last_hidden = self.input_layer

        if self.input_dropout > 0.0:
            last_hidden = Dropout(self.input_dropout, name='input_dropout')(last_hidden)

        for i, (hid_size, hid_drop) in enumerate(zip(self.hidden_size, self.hidden_dropout)):
            center_idx = int(np.floor(len(self.hidden_size) / 2.0))
            if i == center_idx:
                layer_name = 'center'
                stage = 'center'  # let downstream know where we are
            elif i < center_idx:
                layer_name = 'enc%s' % i
                stage = 'encoder'
            else:
                layer_name = 'dec%s' % (i-center_idx)
                stage = 'decoder'

            # use encoder-specific l1/l2 reg coefs if given
            if self.l1_enc_coef != 0. and stage in ('center', 'encoder'):
                l1 = self.l1_enc_coef
            else:
                l1 = self.l1_coef

            if self.l2_enc_coef != 0. and stage in ('center', 'encoder'):
                l2 = self.l2_enc_coef
            else:
                l2 = self.l2_coef

            if i > center_idx:
                self.last_hidden_mean = Dense(hid_size, activation=None, kernel_initializer=self.init,
                                    kernel_regularizer=l1_l2(l1, l2),
                                    name='%s_last_mean'%layer_name)(last_hidden)
                self.last_hidden_disp = Dense(hid_size, activation=None, kernel_initializer=self.init,
                                    kernel_regularizer=l1_l2(l1, l2),
                                    name='%s_last_disp'%layer_name)(last_hidden)
                self.last_hidden_pi = Dense(hid_size, activation=None, kernel_initializer=self.init,
                                    kernel_regularizer=l1_l2(l1, l2),
                                    name='%s_last_pi'%layer_name)(last_hidden)

                if self.batchnorm:
                    self.last_hidden_mean = BatchNormalization(center=True, scale=False)(self.last_hidden_mean)
                    self.last_hidden_disp = BatchNormalization(center=True, scale=False)(self.last_hidden_disp)
                    self.last_hidden_pi = BatchNormalization(center=True, scale=False)(self.last_hidden_pi)

                # Use separate act. layers to give user the option to get pre-activations
                # of layers when requested
                self.last_hidden_mean = Activation(self.activation, name='%s_mean_act'%layer_name)(self.last_hidden_mean)
                self.last_hidden_disp = Activation(self.activation, name='%s_disp_act'%layer_name)(self.last_hidden_disp)
                self.last_hidden_pi = Activation(self.activation, name='%s_pi_act'%layer_name)(self.last_hidden_pi)

                if hid_drop > 0.0:
                    self.last_hidden_mean = Dropout(hid_drop, name='%s_mean_drop'%layer_name)(self.last_hidden_mean)
                    self.last_hidden_disp = Dropout(hid_drop, name='%s_disp_drop'%layer_name)(self.last_hidden_disp)
                    self.last_hidden_pi = Dropout(hid_drop, name='%s_pi_drop'%layer_name)(self.last_hidden_pi)

            else:
                last_hidden = Dense(hid_size, activation=None, kernel_initializer=self.init,
                                    kernel_regularizer=l1_l2(l1, l2),
                                    name=layer_name)(last_hidden)

                if self.batchnorm:
                    last_hidden = BatchNormalization(center=True, scale=False)(last_hidden)

                # Use separate act. layers to give user the option to get pre-activations
                # of layers when requested
                if self.activation in advanced_activations:
                    last_hidden = keras.layers.__dict__[self.activation](name='%s_act'%layer_name)(last_hidden)
                else:
                    last_hidden = Activation(self.activation, name='%s_act'%layer_name)(last_hidden)

                if hid_drop > 0.0:
                    last_hidden = Dropout(hid_drop, name='%s_drop'%layer_name)(last_hidden)

        self.build_output()


    def build_output(self):
        pi = Dense(self.output_size, activation='sigmoid', kernel_initializer=self.init,
                       kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                       name='pi')(self.last_hidden_pi)

        disp = Dense(self.output_size, activation=DispAct,
                           kernel_initializer=self.init,
                           kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                           name='dispersion')(self.last_hidden_disp)

        mean = Dense(self.output_size, activation=MeanAct, kernel_initializer=self.init,
                       kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                       name='mean')(self.last_hidden_mean)

        output = ColwiseMultLayer([mean, self.sf_layer])
        output = SliceLayer(0, name='slice')([output, disp, pi])

        zinb = ZINB(pi, theta=disp, ridge_lambda=self.ridge, debug=self.debug)
        self.loss = zinb.loss
        self.extra_models['pi'] = Model(inputs=self.input_layer, outputs=pi)
        self.extra_models['dispersion'] = Model(inputs=self.input_layer, outputs=disp)
        self.extra_models['mean_norm'] = Model(inputs=self.input_layer, outputs=mean)

        self.model = Model(inputs=[self.input_layer, self.sf_layer], outputs=output)

        self.encoder = self.get_encoder()


class NBForkAutoencoder(NBAutoencoder):

    def build(self):

        self.input_layer = Input(shape=(self.input_size,), name='count')
        self.sf_layer = Input(shape=(1,), name='size_factors')
        last_hidden = self.input_layer

        if self.input_dropout > 0.0:
            last_hidden = Dropout(self.input_dropout, name='input_dropout')(last_hidden)

        for i, (hid_size, hid_drop) in enumerate(zip(self.hidden_size, self.hidden_dropout)):
            center_idx = int(np.floor(len(self.hidden_size) / 2.0))
            if i == center_idx:
                layer_name = 'center'
                stage = 'center'  # let downstream know where we are
            elif i < center_idx:
                layer_name = 'enc%s' % i
                stage = 'encoder'
            else:
                layer_name = 'dec%s' % (i-center_idx)
                stage = 'decoder'

            # use encoder-specific l1/l2 reg coefs if given
            if self.l1_enc_coef != 0. and stage in ('center', 'encoder'):
                l1 = self.l1_enc_coef
            else:
                l1 = self.l1_coef

            if self.l2_enc_coef != 0. and stage in ('center', 'encoder'):
                l2 = self.l2_enc_coef
            else:
                l2 = self.l2_coef

            if i > center_idx:
                self.last_hidden_mean = Dense(hid_size, activation=None, kernel_initializer=self.init,
                                    kernel_regularizer=l1_l2(l1, l2),
                                    name='%s_last_mean'%layer_name)(last_hidden)
                self.last_hidden_disp = Dense(hid_size, activation=None, kernel_initializer=self.init,
                                    kernel_regularizer=l1_l2(l1, l2),
                                    name='%s_last_disp'%layer_name)(last_hidden)

                if self.batchnorm:
                    self.last_hidden_mean = BatchNormalization(center=True, scale=False)(self.last_hidden_mean)
                    self.last_hidden_disp = BatchNormalization(center=True, scale=False)(self.last_hidden_disp)

                # Use separate act. layers to give user the option to get pre-activations
                # of layers when requested
                self.last_hidden_mean = Activation(self.activation, name='%s_mean_act'%layer_name)(self.last_hidden_mean)
                self.last_hidden_disp = Activation(self.activation, name='%s_disp_act'%layer_name)(self.last_hidden_disp)

                if hid_drop > 0.0:
                    self.last_hidden_mean = Dropout(hid_drop, name='%s_mean_drop'%layer_name)(self.last_hidden_mean)
                    self.last_hidden_disp = Dropout(hid_drop, name='%s_disp_drop'%layer_name)(self.last_hidden_disp)

            else:
                last_hidden = Dense(hid_size, activation=None, kernel_initializer=self.init,
                                    kernel_regularizer=l1_l2(l1, l2),
                                    name=layer_name)(last_hidden)

                if self.batchnorm:
                    last_hidden = BatchNormalization(center=True, scale=False)(last_hidden)

                # Use separate act. layers to give user the option to get pre-activations
                # of layers when requested
                if self.activation in advanced_activations:
                    last_hidden = keras.layers.__dict__[self.activation](name='%s_act'%layer_name)(last_hidden)
                else:
                    last_hidden = Activation(self.activation, name='%s_act'%layer_name)(last_hidden)

                if hid_drop > 0.0:
                    last_hidden = Dropout(hid_drop, name='%s_drop'%layer_name)(last_hidden)

        self.build_output()


    def build_output(self):

        disp = Dense(self.output_size, activation=DispAct,
                           kernel_initializer=self.init,
                           kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                           name='dispersion')(self.last_hidden_disp)

        mean = Dense(self.output_size, activation=MeanAct, kernel_initializer=self.init,
                       kernel_regularizer=l1_l2(self.l1_coef, self.l2_coef),
                       name='mean')(self.last_hidden_mean)

        output = ColwiseMultLayer([mean, self.sf_layer])
        output = SliceLayer(0, name='slice')([output, disp])

        nb = NB(theta=disp, debug=self.debug)
        self.loss = nb.loss
        self.extra_models['dispersion'] = Model(inputs=self.input_layer, outputs=disp)
        self.extra_models['mean_norm'] = Model(inputs=self.input_layer, outputs=mean)

        self.model = Model(inputs=[self.input_layer, self.sf_layer], outputs=output)

        self.encoder = self.get_encoder()

######################################################################################################################################################################################################################################################################     

def _nan2zero(x):
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x), x)

def _nan2inf(x):
    return tf.where(tf.math.is_nan(x), tf.zeros_like(x)+np.inf, x)

def _nelem(x):
    nelem = tf.reduce_sum(tf.cast(~tf.math.is_nan(x), tf.float32))
    return tf.cast(tf.where(tf.equal(nelem, 0.), 1., nelem), x.dtype)


def _reduce_mean(x):
    nelem = _nelem(x)
    x = _nan2zero(x)
    return tf.divide(tf.reduce_sum(x), nelem)


def mse_loss(y_true, y_pred):
    ret = tf.square(y_pred - y_true)

    return _reduce_mean(ret)


class NB(object):
    def __init__(self, theta=None, masking=False, scope='nbinom_loss/',
                 scale_factor=1.0, debug=False):

        # for numerical stability
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.debug = debug
        self.scope = scope
        self.masking = masking
        self.theta = theta

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps

        with tf.name_scope(self.scope):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32) * scale_factor

            if self.masking:
                nelem = _nelem(y_true)
                y_true = _nan2zero(y_true)

            # Clip theta
            theta = tf.minimum(self.theta, 1e6)

            t1 = tf.math.lgamma(theta+eps) + tf.math.lgamma(y_true+1.0) - tf.math.lgamma(y_true+theta+eps)
            t2 = (theta+y_true) * tf.math.log(1.0 + (y_pred/(theta+eps))) + (y_true * (tf.math.log(theta+eps) - tf.math.log(y_pred+eps)))

            if self.debug:
                assert_ops = [
                        tf.verify_tensor_all_finite(y_pred, 'y_pred has inf/nans'),
                        tf.verify_tensor_all_finite(t1, 't1 has inf/nans'),
                        tf.verify_tensor_all_finite(t2, 't2 has inf/nans')]

                tf.summary.histogram('t1', t1)
                tf.summary.histogram('t2', t2)

                with tf.control_dependencies(assert_ops):
                    final = t1 + t2

            else:
                final = t1 + t2

            final = _nan2inf(final)

            if mean:
                if self.masking:
                    final = tf.divide(tf.reduce_sum(final), nelem)
                else:
                    final = tf.reduce_mean(final)


        return final

class ZINB(NB):
    def __init__(self, pi, ridge_lambda=0.0, scope='zinb_loss/', **kwargs):
        super().__init__(scope=scope, **kwargs)
        self.pi = pi
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps

        with tf.name_scope(self.scope):
            # reuse existing NB neg.log.lik.
            # mean is always False here, because everything is calculated
            # element-wise. we take the mean only in the end
            nb_case = super().loss(y_true, y_pred, mean=False) - tf.math.log(1.0-self.pi+eps)

            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32) * scale_factor
            theta = tf.minimum(self.theta, 1e6)

            zero_nb = tf.pow(theta/(theta+y_pred+eps), theta)
            zero_case = -tf.math.log(self.pi + ((1.0-self.pi)*zero_nb)+eps)
            result = tf.where(tf.less(y_true, 1e-8), zero_case, nb_case)
            
            ridge = self.ridge_lambda*tf.square(self.pi)
            
            result += ridge

            if mean:
                if self.masking:
                    result = _reduce_mean(result)
                else:
                    result = tf.reduce_mean(result)

            result = _nan2inf(result)

            if self.debug:
                tf.summary.histogram('nb_case', nb_case)
                tf.summary.histogram('zero_nb', zero_nb)
                tf.summary.histogram('zero_case', zero_case)
                tf.summary.histogram('ridge', ridge)

        return result
    
    
    
    
from keras.layers import Lambda, Dense
from keras.engine.topology import Layer
from keras.engine.base_layer import InputSpec

class ConstantDispersionLayer(Layer):
    '''
        An identity layer which allows us to inject extra parameters
        such as dispersion to Keras models
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.theta = self.add_weight(shape=(1, input_shape[1]),
                                     initializer='zeros',
                                     trainable=True,
                                     name='theta')
        self.theta_exp = tf.clip_by_value(K.exp(self.theta), 1e-3, 1e4)
        super().build(input_shape)

    def call(self, x):
        return tf.identity(x)

    def compute_output_shape(self, input_shape):
        return input_shape


class SliceLayer(Layer):
    def __init__(self, index, **kwargs):
        self.index = index
        super().__init__(**kwargs)

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError('Input should be a list')

        super().build(input_shape)

    def call(self, x):
        assert isinstance(x, list), 'SliceLayer input is not a list'
        return x[self.index]

    def compute_output_shape(self, input_shape):
        return input_shape[self.index]


class ElementwiseDense(Dense):
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        assert (input_dim == self.units) or (self.units == 1), \
               "Input and output dims are not compatible"

        # shape=(input_units, ) makes this elementwise bcs of broadcasting
        self.kernel = self.add_weight(shape=(self.units,),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        # use * instead of tf.matmul, we need broadcasting here
        output = inputs * self.kernel
        if self.use_bias:
            output = output + self.bias
        if self.activation is not None:
            output = self.activation(output)
        return output


nan2zeroLayer = Lambda(lambda x: tf.where(tf.math.is_nan(x), tf.zeros_like(x), x))
ColwiseMultLayer = Lambda(lambda l: l[0]*tf.reshape(l[1], (-1,1)))



def write_text_matrix(matrix, filename, rownames=None, colnames=None, transpose=False):
    if transpose:
        matrix = matrix.T
        rownames, colnames = colnames, rownames

    pd.DataFrame(matrix, index=rownames, columns=colnames).to_csv(filename,
                                                                  sep='\t',
                                                                  index=(rownames is not None),
                                                                  header=(colnames is not None),
                                                                  float_format='%.6f')
######################################################################################################################################################################################################################################################################     
























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
          verbose=True,
          verbosity = 2,    # added by me later, to hide progres bar
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
        lr_cb = ReduceLROnPlateau(monitor='val_loss', patience=reduce_lr, verbose=1)      #patientce = how many must plateau
        callbacks.append(lr_cb)
    if early_stop:
        es_cb = EarlyStopping(monitor='val_loss', patience=early_stop, verbose=1)
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
                      verbose=verbosity,
                      #**kwds
                      )

    return loss        
        



def plot_history(adata, outputplot_dir = "./"):
    
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
    
    os.makedirs(outputplot_dir, exist_ok=True)
    plt.savefig(outputplot_dir + "training_history.png")





# %% Main


def sca(adata_train, 
        adata_test,
        loss_name,
        test_index,
        
        output_dir,
        outputplot_dir,
        
        mode = "full",
        ae_type = "normal",
        
        # training args
        epochs = 300,
        reduce_lr = 15,
        early_stop = 35,
        batch_size = 32,
        optimizer = "RMSprop",
        batchnorm = True,
        learning_rate = None,
        random_state = 0,
        verbose = True,
        threads = None
        ):
    
    

    ## input checker    
    assert isinstance(adata_train, anndata.AnnData), 'adata must be an AnnData instance'
    assert mode in ('denoise', 'latent', 'full'), '%s is not a valid mode.' % mode
    


## do stuff
    input_size = adata_train.n_vars
    
    
    AE_types = {'normal': Autoencoder, 'poisson': PoissonAutoencoder,
            'nb': NBConstantDispAutoencoder, 'nb-conddisp': NBAutoencoder,
            'nb-shared': NBSharedAutoencoder, 'nb-fork': NBForkAutoencoder,
            'zinb': ZINBConstantDispAutoencoder, 'zinb-conddisp': ZINBAutoencoder,
            'zinb-shared': ZINBSharedAutoencoder, 'zinb-fork': ZINBForkAutoencoder,
            'zinb-elempi': ZINBAutoencoderElemPi}    
    
    
    
    ae = AE_types[args.AEtype](input_size = input_size, 
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
    
    if adata_test is not None:
        denoised_test = ae.predict(adata = adata_test, mode = mode)
    else:
        denoised_test = None

    
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
    plot_history(adata = adata_train, outputplot_dir = outputplot_dir)
    
    
    
    # old way
    # ae.write_output_deprecated(adata = adata_train, file_path = output_dir + "test_dir/", mode = "full")
    # ae.write_output_deprecated(adata = adata_test, file_path = output_dir + "train_dir/", mode = "full")
    
    ae.write_output(test_adata = adata_test, train_adata = adata_train, test_index = test_index, file_path = output_dir, mode = mode)


    return (adata_train, adata_test, ae)






def sca_preprocess(adata, test_split = False, filter_ = True, size_factors = True, logtrans = True, normalize = True):
    
    # if test_split:
    #     train_idx, test_idx = train_test_split(np.arange(adata.n_obs), test_size=0.1, random_state=42)
    #     spl = pd.Series(['train'] * adata.n_obs)        # first make all train, then overwrite the tests
    #     spl.iloc[test_idx] = 'test'
    #     adata.obs['dca_split'] = spl.values
    # else:
    #     adata.obs['dca_split'] = 'train'
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
    




def read_input(input_dir, output_dir, split):
    
    data = np.loadtxt(open(input_dir + "matrix.tsv"), delimiter="\t")
    genes = pd.read_csv(input_dir + "genes.tsv", delimiter = "\t", header = None)
    barcodes = pd.read_csv(input_dir + "barcodes.tsv", delimiter = "\t", header = None)
    
    

    if split:
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
    
        return adata_train, adata_test, genes, barcodes, test_index
                
    
    else:
        test_index = None
        train_index = None
    
        adata_train = sc.AnnData(data)
        adata_train.obs_names = barcodes.iloc[:,0] + "_" + barcodes.iloc[:,1]
        adata_train.var_names = genes.iloc[:,0]

        adata_test = None
        
        
        nonzero_genes, _ = sc.pp.filter_genes(adata_train.X, min_counts=1)
        assert nonzero_genes.all(), 'Please remove all-zero genes before using DCA. Train'
    
        return adata_train, adata_test, genes, barcodes, test_index        



def sca_main(input_dir, output_dir, outputplot_dir, loss_name, split):
    
    # generate AnnData
    adata_train, adata_test, genes, barcodes, test_index = read_input(input_dir = input_dir, output_dir = output_dir, split = split)    


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
    
    
    if split:
        
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
                        test_index = test_index,
                        
                        mode = "latent",
                        ae_type = "normal",
                        
                        # training args
                        epochs = 300,
                        reduce_lr = 15,
                        early_stop = 35,
                        batch_size = 32,
                        batchnorm = True,
                        optimizer = "RMSprop",
                        learning_rate = None,
                        random_state = 0,
                        verbose = True,
                        threads = None,
                        
                        output_dir = output_dir,
                        outputplot_dir = outputplot_dir
                        )
    
    
    net.save_model(output_dir)

    # transfer genes and barcodes
    # I trust this more, if I do it here, insted of reading the genes from the adata again in the write_output function.

    genes.to_csv(output_dir + "genes.tsv", sep = "\t", index = False, header = False)
    barcodes.to_csv(output_dir + "barcodes.tsv", sep = "\t", index = False, header = False)

    if split:
        np.savetxt(output_dir + "test_index.tsv", test_index, fmt = "%d")





# %%
#------------------------------------------------------------------------------
# Call program
#------------------------------------------------------------------------------

if __name__ == "__main__":
    

        

    
    MeanAct = lambda x: tf.clip_by_value(K.exp(x), 1e-5, 1e6)
    DispAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)
    advanced_activations = ('PReLU', 'LeakyReLU')    
        
    

    source_input_dir = args.input_dir
    source_output_dir = args.output_dir
    source_outputplot_dir = args.outputplot_dir
        
    
    
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
        print(datetime.now().strftime("%H:%M:%S>"), "Starting sca_autoencoder.py")    
     
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
        
        
        if args.splitnumber != None: # and args.mode == "split":
            
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
    
            sca_main(input_dir = input_dir,  
                      output_dir = output_dir,
                      outputplot_dir = outputplot_dir,
                      loss_name = args.loss,
                      split = True)
                
            
            
            
            
            
            
            
    # %% NO SPLIT
    if nosplit == True:
        
        print(datetime.now().strftime("%H:%M:%S>"), "Starting sca_autoencoder.py (nosplit)")    
     
        input_dir = source_input_dir + "no_split/"
        output_dir = source_output_dir + "no_split/"
        outputplot_dir = source_outputplot_dir + "no_split/"
        
        
        sca_main(input_dir = input_dir,  
                  output_dir = output_dir,
                  outputplot_dir = outputplot_dir,
                  loss_name = args.loss,
                  split = False)
        
        



            
            
            
            