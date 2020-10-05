# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 01:18:36 2020

@author: Mike Toreno II
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









try:
    os.chdir(os.path.dirname(sys.argv[0]))
except:
    pass



parser = argparse.ArgumentParser(description = "calculate PCAs")  #required
parser.add_argument("-i","--input_dir", help="input directory", default = "../inputs/baselines/baseline_data/scaPCA_output/")
parser.add_argument("-p","--output_dir", help="out directory", default = "../outputs/baselines/random_forrest/")
#parser.add_argument("-t","--title", help="title that will be written into the output file", default = "title placeholder")
#parser.add_argument("-r", "--reset", help="if this is called, the previous results file will be overwritten, otherwise results are appended - call for the first run of the classifier", action="store_true")
# parser.add_argument("--ova", help="to use one-versus all classification instead of multiclass", action="store_true")
parser.add_argument("--plotonly", help="to only show the plots in order to find sweet parameters for the forest", action="store_true")

args = parser.parse_args() #required



input_path = args.input_dir
output_dir = args.output_dir


print(datetime.now().strftime("%H:%M:%S>"), "Starting sca_randforrest.py")
print(input_path)




# %%

n_trees = 100


















# %% functions





def compute_metrics(y_true, y_pred):
    '''
    Computes several quality metrics of the predicted labels and prints
    them to `stdout`.

    :param y_true: true class labels
    :param y_pred: predicted class labels
    '''

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # print('TP: {0:d}'.format(tp))
    # print('FP: {0:d}'.format(fp))
    # print('TN: {0:d}'.format(tn))
    # print('FN: {0:d}'.format(fn))
    # print('Accuracy: {0:.3f}'.format(acc))
    
    accuracy = (tp+tn)/(tp+fp+fn+tn)
    precision = tp/(tp+fp)
    print("the line is {0:d}/({0:d}+{1:d})".format(tp, fp))
    
    recall = tp/(tp+fn)
    f1score = 2*recall*precision/(recall + precision)
    
    
    results = [accuracy, precision, recall, f1score, tn, fp, fn, tp]
    return results







# %% Read Input data

print(datetime.now().strftime("%H:%M:%S>"), "reading input data...")
print(input_path)

data = np.loadtxt(open(input_path + "matrix.tsv"), delimiter="\t")
genes = pd.read_csv(input_path + "genes.tsv", delimiter = "\t", header = None)
barcodes = pd.read_csv(input_path + "barcodes.tsv", delimiter = "\t", header = None)


labels = barcodes.iloc[:,1]
# labelset = list(set(labels))


# %%

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size = 0.25)







# %%


if args.ova:

    print("OVA NOT IMPLEMENTED YET?")    

    kfold = 4
    labels = 3
    labelset = 2
    result = 1
    

    
    print(datetime.now().strftime("%H:%M:%S>"), "starting classification...")
    
    
    kfolder = StratifiedKFold(n_splits=kfold, shuffle=True)
    
    pandas = pd.DataFrame(index = ["Accuracy ", "Precision", "Recall   ", "F1 Score ", "TN", "FP", "FN", "TP"])
    foldnumber = 0
    
    for trainindices, testindices in kfolder.split(data, labels):
        traindata = data[trainindices,:]
        trainlabels = barcodes.iloc[trainindices,:]
        testdata = data[testindices,:]
        testlabels = barcodes.iloc[testindices,:]
        
        foldnumber = foldnumber + 1
        
        for label in labelset:
            binary_trainlabels = (np.array(trainlabels.iloc[:,1]) == label)
            binary_testlabels = (np.array(testlabels.iloc[:,1]) == label)
            
            
            
            classifier = RandomForestClassifier(n_estimators = n_trees, criterion = "gini")
            classifier.fit(x_train, y_train)
            
            y_pred = classifier.predict(x_test)  
        

            
            result = compute_metrics()
            
            
            
            
            
            resultname = "Fold " + str(foldnumber) + ": " + label
            pandas[resultname] = result
            
        


 
    
    
    
    


# %% Calculate celltype averages



# celltype_averages = pd.DataFrame(index = ["Accuracy ", "Precision", "Recall   ", "F1 Score ", "TN", "FP", "FN", "TP"])
# columns = pandas.columns

# for cellidx in range(len(labelset)):
    
#     tempframe = pd.DataFrame(index = ["Accuracy ", "Precision", "Recall   ", "F1 Score ", "TN", "FP", "FN", "TP"])    
    
#     for foldidx in range(kfold):
#         idx = cellidx + foldidx * len(labelset)       
#         tempframe[columns[idx]] = pandas.iloc[:,idx]
        
#     colname = labelset[cellidx]
#     celltype_averages[colname] = tempframe.mean(axis = 1)










# # %% gENERATE Output


# if not os.path.exists(output_dir):
#     print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Directory...")
#     os.makedirs(output_dir)


# print(datetime.now().strftime("%H:%M:%S>"), "writing data to output file...")


# if args.reset:
#     file = open(output_dir + "one_versus_all_classification.txt", "w")
# else:
#     file = open(output_dir + "one_versus_all_classification.txt", "a")
#     file.write("\n")
#     file.write("\n")
#     file.write("\n")
#     file.write("\n")
#     file.write("\n")    

# file.write("######" + args.title + "######\n")
# file.write("input_data from " + input_path + ", Classifier " + classifier + "\n")


# averages = pandas.mean(axis = 1)

# file.write("\nAverage Accuracy: \t" + '{:.4f}'.format(averages.iloc[0]))
# file.write("\nAverage Precision:\t" + '{:.4f}'.format(averages.iloc[1]))
# file.write("\nAverage Recall:   \t" + '{:.4f}'.format(averages.iloc[2]))
# file.write("\nAverage F1 Score: \t" + '{:.4f}'.format(averages.iloc[3]))

# file.write("\n\nCelltype Averages:\n")
# file.close()
# celltype_averages.to_csv(output_dir + "one_versus_all_classification.txt", mode = "a", sep = "\t")


# file = open(output_dir + "one_versus_all_classification.txt", "a")
# file.write("\nComplete Dataframe:\n")
# file.close()
# pandas.to_csv(output_dir + "one_versus_all_classification.txt", mode = "a", sep = "\t")


# # %% 
# print(datetime.now().strftime("%H:%M:%S>"), "sca_classification.py terminated successfully\n")
























