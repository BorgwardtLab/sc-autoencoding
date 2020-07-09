# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 21:11:37 2020

@author: Mike Toreno II
"""


# %% Load Data
import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm # colourpalette
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


print(datetime.now().strftime("%H:%M:%S>"), "Starting sca_classifcation.py")


try:
    os.chdir(os.path.dirname(sys.argv[0]))
except:
    pass
         


parser = argparse.ArgumentParser(description = "calculate PCAs")  #required
parser.add_argument("-k","--kfold", help="the number of k-folds to test the classifier on", type = int, default = 5)
parser.add_argument("-i","--input_dir", help="input directory", default = "../inputs/baseline_data/scaPCA_output/")
parser.add_argument("-p","--outputplot_dir", help="plot directory", default = "../outputs/classification/")
parser.add_argument("-c","--classifier", help="helptext", default = "logreg", choices = ["logreg","b","c"])
args = parser.parse_args() #required



input_path = args.input_dir
outputplot_dir = args.outputplot_dir
kfold = args.kfold
classifier = args.classifier




# %% functions


def choose_classifier(traindata, trainlabels, testdata, testlabels, classifier):
    
    if classifier == "logreg":
        metric = logistic_regression(traindata, trainlabels, testdata, testlabels)
    else:
        print("illegal method lol, this wasn't supposed to be possible")
    
    return metric



def logistic_regression(traindata, trainlabels, testdata, testlabels):
    
    ### scaler = StandardScaler().fit(unscaled_traindata);
    ### traindata = scaler.transform(unscaled_traindata);    
    
    log_regressor = LogisticRegression(penalty = "l2", 
                                       C = 1.0, # regularizer, = none
                                       solver = "liblinear") # with other solvers, apparently multiclass is possible
    ### testdata = scaler.transform(unscaled_testdata)

    log_regressor.fit(X = traindata, y = trainlabels.iloc[:,1])
    
    prediction = log_regressor.predict(testdata)

    metric = compute_metrics(testlabels.iloc[:,1], prediction)
    
    return metric






def compute_metrics(y_true, y_pred):
    '''
    Computes several quality metrics of the predicted labels and prints
    them to `stdout`.

    :param y_true: true class labels
    :param y_pred: predicted class labels
    '''

    #tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    gullli = confusion_matrix(y_true, y_pred)
    
    acc = accuracy_score(y_true, y_pred)


    # print('TP: {0:d}'.format(tp))
    # print('FP: {0:d}'.format(fp))
    # print('TN: {0:d}'.format(tn))
    # print('FN: {0:d}'.format(fn))
    # print('Accuracy: {0:.3f}'.format(acc))
    
    results = [tp, fp, fn, tp, acc]
    return results










# %% Read Input data

print(datetime.now().strftime("%H:%M:%S>"), "reading input data...")

data = np.loadtxt(open(input_path + "coordinates.tsv"), delimiter="\t")
genes = pd.read_csv(input_path + "genes.tsv", delimiter = "\t", header = None)
barcodes = pd.read_csv(input_path + "barcodes.tsv", delimiter = "\t", header = None)
labels = barcodes.iloc[:,1]


# %%




kfolder = StratifiedKFold(n_splits=kfold, shuffle=True)

for trainindices, testindices in kfolder.split(data, labels):
    traindata = data[trainindices,:]
    trainlabels = barcodes.iloc[trainindices,:]
    testdata = data[testindices,:]
    testlabels = barcodes.iloc[testindices,:]
    
    
    #result = choose_classifier(traindata, trainlabels, testdata, testlabels, classifier)
    
    


# %% this is the tryout section here




### testdata = scaler.transform(unscaled_testdata)


labels = set(trainlabels.iloc[:,1])





for label in labels:
    binary_trainlabels = (np.array(trainlabels.iloc[:,1]) == label)
    binary_testlabels = (np.array(testlabels.iloc[:,1]) == label)
      
        
        
        
        
        
log_regressor = LogisticRegression(penalty = "l2", 
                               C = 1.0, # regularizer, = none I think
                               solver = "liblinear") # with other solvers, apparently multiclass is possible
log_regressor.fit(X = traindata, y = binary_trainlabels)

prediction = log_regressor.predict(testdata)

metric = confusion_matrix(binary_testlabels, prediction)


#tn, fp, fn, tp = confusion_matrix(current_testlabels, prediction).ravel()






# %%

labels_copy = np.array(trainlabels.iloc[:,1])



binary_labels = (labels_copy == "frozen_bmmc_healthy_donor2")

















































