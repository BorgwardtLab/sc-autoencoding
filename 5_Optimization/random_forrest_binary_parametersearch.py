# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 02:00:12 2020

@author: Emilia Radikov





TODO
- instead of randomly picking a label, loop through all.
- add the parameterranges as arguments, so they can be controlled externally


"""


# %% Load Data
import sys
import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

import matplotlib.pyplot as plt




import random




try:
    os.chdir(os.path.dirname(sys.argv[0]))
except:
    pass



parser = argparse.ArgumentParser(description = "calculate PCAs")  #required
parser.add_argument("-i","--input_dir", help="input directory", default = "../inputs/baselines/baseline_data/scaPCA_output/")
#parser.add_argument("-p","--output_dir", help="out directory", default = "../outputs/baselines/random_forrest/")
parser.add_argument("-p","--outputplot_dir", help="out directory", default = "../outputs/optimization/random_forrest/binary/")



args = parser.parse_args() #required



input_path = args.input_dir
outputplot_dir = args.outputplot_dir


print(datetime.now().strftime("%H:%M:%S>"), "Starting sca_randforrest.py")
print(input_path)










# %% Read Input data

print(datetime.now().strftime("%H:%M:%S>"), "reading input data...")
print(input_path)

data = np.loadtxt(open(input_path + "matrix.tsv"), delimiter="\t")
genes = pd.read_csv(input_path + "genes.tsv", delimiter = "\t", header = None)
barcodes = pd.read_csv(input_path + "barcodes.tsv", delimiter = "\t", header = None)


labels = barcodes.iloc[:,1]
# labelset = list(set(labels))






# %% Individual Parameteresearch

os.makedirs(outputplot_dir, exist_ok=True)


# chose a random label
my_label = random.choice(labels)
print(my_label)

true_binlabel = labels == my_label

x_train, x_test, y_train, y_test = train_test_split(data, true_binlabel, test_size = 0.25)






print("n_estimators")
### n_estimators
n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]
train_results = []
test_results = []

for estimator in n_estimators:
    rf = RandomForestClassifier(n_estimators=estimator, n_jobs=-1)
    rf.fit(x_train, y_train)
    train_pred = rf.predict(x_train)
   
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
   
    train_results.append(roc_auc)
    y_pred = rf.predict(x_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)
   
   
from matplotlib.legend_handler import HandlerLine2D
plt.figure()
line1, = plt.plot(n_estimators, train_results, "b", label="Train AUC")
line2, = plt.plot(n_estimators, test_results, "r", label= "Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel("AUC score")
plt.xlabel("n_estimators")
plt.title("mylabel = " + my_label)
plt.show()
plt.savefig(outputplot_dir + "auc_n_estimators.png")







### max depth
print("max depth")
max_depths = np.linspace(1, 32, 32, endpoint=True)
train_results = []
test_results = []
for max_depth in max_depths:
    rf = RandomForestClassifier(max_depth=max_depth, n_jobs=-1)
    rf.fit(x_train, y_train)
    train_pred = rf.predict(x_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = rf.predict(x_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
plt.figure()
line1, = plt.plot(max_depths, train_results, "b", label="Train AUC")
line2, = plt.plot(max_depths, test_results, "r", label= "Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel("AUC score")
plt.xlabel("max depth")
plt.show()
plt.savefig(outputplot_dir + "auc_max_depth.png")

    
### min samples split
# the higher, the more constrained does each tree get, as it has to consider more samples  
print("min sample split")  
min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
train_results = []
test_results = []
for min_samples_split in min_samples_splits:
    rf = RandomForestClassifier(min_samples_split=min_samples_split)
    rf.fit(x_train, y_train)
    train_pred = rf.predict(x_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = rf.predict(x_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
plt.figure()
line1, = plt.plot(min_samples_splits, train_results, "b", label="Train AUC")
line2, = plt.plot(min_samples_splits, test_results, "r", label= "Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel("AUC score")
plt.xlabel("min sample split")
plt.show() 
plt.savefig(outputplot_dir + "auc_min_samples_split.png")


### min sample leaf 
# -> how many samples a leaf must have at least
print("min sample leaf")
min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
train_results = []
test_results = []
for min_samples_leaf in min_samples_leafs:
    rf = RandomForestClassifier(min_samples_leaf=min_samples_leaf)
    rf.fit(x_train, y_train)
    train_pred = rf.predict(x_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = rf.predict(x_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
plt.figure()
line1, = plt.plot(min_samples_leafs, train_results, "b", label="Train AUC")
line2, = plt.plot(min_samples_leafs, test_results, "r", label= "Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel("AUC score")
plt.xlabel("min sample leaf")
plt.show()
plt.savefig(outputplot_dir + "auc_min_samples_leafs.png")



### max features
print("max features")
max_features = list(range(1,data.shape[1]))
train_results = []
test_results = []
for max_feature in max_features:
    rf = RandomForestClassifier(max_features=max_feature)
    rf.fit(x_train, y_train)
    train_pred = rf.predict(x_train)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = rf.predict(x_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)
from matplotlib.legend_handler import HandlerLine2D
plt.figure()
line1, = plt.plot(max_features, train_results, "b", label="Train AUC")
line2, = plt.plot(max_features, test_results, "r", label= "Test AUC")
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel("AUC score")
plt.xlabel("max features")
plt.show() 
plt.savefig(outputplot_dir + "auc_max_features.png")











