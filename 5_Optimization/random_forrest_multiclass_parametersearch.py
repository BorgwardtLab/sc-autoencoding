# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 02:00:12 2020

@author: Emilia Radikov





TODO
- instead of randomly picking a label, loop through all.

"""


# %% Load Data
import sys
import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt



try:
    os.chdir(os.path.dirname(sys.argv[0]))
except:
    pass



parser = argparse.ArgumentParser(description = "calculate PCAs")  #required
parser.add_argument("-i","--input_dir", help="input directory", default = "../inputs/baselines/baseline_data/scaPCA_output/")
#parser.add_argument("-p","--output_dir", help="out directory", default = "../outputs/baselines/random_forrest/")
parser.add_argument("-p","--outputplot_dir", help="out directory", default = "../outputs/optimization/random_forest/multiclass/")

parser.add_argument('--n_trees', nargs='+', type = int, default = [1, 2, 6, 12, 32, 50, 64, 86, 100, 120, 150, 250, 500], help="default [1, 2, 4, 8, 16, 32, 64, 100, 200]")
parser.add_argument('--max_max_depth', type = int, default = 100, help = "it will try out the values from np.linspace(1, max_max_depth, 20)")
parser.add_argument('--max_min_samples_leaf', type = float, default = 0.5)
parser.add_argument('--max_min_samples_split', type = float, default = 0.5)


args = parser.parse_args() #required



input_dir = args.input_dir
outputplot_dir = args.outputplot_dir






# %% Read Input data

print(datetime.now().strftime("%H:%M:%S>"), "reading input data...")
print(input_dir)

data = np.loadtxt(open(input_dir + "matrix.tsv"), delimiter="\t")
genes = pd.read_csv(input_dir + "genes.tsv", delimiter = "\t", header = None)
barcodes = pd.read_csv(input_dir + "barcodes.tsv", delimiter = "\t", header = None)


labels = barcodes.iloc[:,1]




# %% Individual Parameteresearch

os.makedirs(outputplot_dir, exist_ok=True)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.25)

# %%


n_estimators = args.n_trees
max_depths = np.linspace(1, args.max_max_depth, 20, dtype = int)
min_samples_leafs = np.linspace(0.0001, args.max_min_samples_leaf, 12, dtype = float)
min_samples_splits = np.linspace(0.0001, args.max_min_samples_split, 12, dtype = float)
max_features = np.linspace(1, data.shape[1], 30, dtype = int)

print(datetime.now().strftime("%H:%M:%S>"), "Starting sca_randforrest.py")
print(n_estimators)
print(max_depths)
print(min_samples_leafs)
print(min_samples_splits)
print(max_features)


# %%



print("n_estimators")
train_results = []
test_results = []

for estimator in n_estimators:
    rf = RandomForestClassifier(n_estimators=estimator, n_jobs=-1)
    rf.fit(x_train, y_train)
    
    train_pred = rf.predict(x_train)
    accuracy = sum(train_pred == y_train)/len(y_train)
    train_results.append(accuracy)   
    
    test_pred = rf.predict(x_test)
    accuracy = sum(test_pred == y_test)/len(y_test)
    test_results.append(accuracy)       
    

   
from matplotlib.legend_handler import HandlerLine2D
plt.figure()
line1, = plt.plot(n_estimators, train_results, "b", label="Train accuracy")
line2, = plt.plot(n_estimators, test_results, "r", label= "Test accuracy")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel("accuracy")
plt.xlabel("n_estimators")
plt.title(input_dir)
plt.show()
plt.savefig(outputplot_dir + "auc_n_estimators.png")






### max depth
print("max depth")
train_results = []
test_results = []
for max_depth in max_depths:
    rf = RandomForestClassifier(max_depth=max_depth, n_jobs=-1)
    rf.fit(x_train, y_train)
    
    train_pred = rf.predict(x_train)
    accuracy = sum(train_pred == y_train)/len(y_train)
    train_results.append(accuracy)   
    
    test_pred = rf.predict(x_test)
    accuracy = sum(test_pred == y_test)/len(y_test)
    test_results.append(accuracy)       
    

from matplotlib.legend_handler import HandlerLine2D
plt.figure()
line1, = plt.plot(max_depths, train_results, "b", label="Train accuracy")
line2, = plt.plot(max_depths, test_results, "r", label= "Test accuracy")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel("accuracy")
plt.xlabel("max depth")
plt.show()
plt.savefig(outputplot_dir + "auc_max_depth.png")







    
### min samples split
# the higher, the more constrained does each tree get, as it has to consider more samples  
print("min sample split")  
train_results = []
test_results = []
for min_samples_split in min_samples_splits:
    rf = RandomForestClassifier(min_samples_split=min_samples_split)
    rf.fit(x_train, y_train)
    
    train_pred = rf.predict(x_train)
    accuracy = sum(train_pred == y_train)/len(y_train)
    train_results.append(accuracy)   
    
    test_pred = rf.predict(x_test)
    accuracy = sum(test_pred == y_test)/len(y_test)
    test_results.append(accuracy)       
    

from matplotlib.legend_handler import HandlerLine2D
plt.figure()
line1, = plt.plot(min_samples_splits, train_results, "b", label="Train accuracy")
line2, = plt.plot(min_samples_splits, test_results, "r", label= "Test accuracy")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel("accuracy")
plt.xlabel("min sample split")
plt.show() 
plt.savefig(outputplot_dir + "auc_min_samples_split.png")







### min sample leaf 
# -> how many samples a leaf must have at least
print("min sample leaf")
train_results = []
test_results = []
for min_samples_leaf in min_samples_leafs:
    rf = RandomForestClassifier(min_samples_leaf=min_samples_leaf)
    rf.fit(x_train, y_train)

    train_pred = rf.predict(x_train)
    accuracy = sum(train_pred == y_train)/len(y_train)
    train_results.append(accuracy)   
    
    test_pred = rf.predict(x_test)
    accuracy = sum(test_pred == y_test)/len(y_test)
    test_results.append(accuracy)       
    
from matplotlib.legend_handler import HandlerLine2D
plt.figure()
line1, = plt.plot(min_samples_leafs, train_results, "b", label="Train accuracy")
line2, = plt.plot(min_samples_leafs, test_results, "r", label= "Test accuracy")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel("accuracy")
plt.xlabel("min sample leaf")
plt.show()
plt.savefig(outputplot_dir + "auc_min_samples_leafs.png")




### max features
print("max features")
train_results = []
test_results = []
for max_feature in max_features:
    rf = RandomForestClassifier(max_features=max_feature)
    rf.fit(x_train, y_train)
    
    train_pred = rf.predict(x_train)
    accuracy = sum(train_pred == y_train)/len(y_train)
    train_results.append(accuracy)   
    
    test_pred = rf.predict(x_test)
    accuracy = sum(test_pred == y_test)/len(y_test)
    test_results.append(accuracy)       
    
from matplotlib.legend_handler import HandlerLine2D
plt.figure()
line1, = plt.plot(max_features, train_results, "b", label="Train accuracy")
line2, = plt.plot(max_features, test_results, "r", label= "Test accuracy")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel("accuracy")
plt.xlabel("max features")
plt.show() 
plt.savefig(outputplot_dir + "auc_max_features.png")











