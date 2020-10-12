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

from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier



import matplotlib.pyplot as plt




try:
    os.chdir(os.path.dirname(sys.argv[0]))
except:
    pass



parser = argparse.ArgumentParser(description = "calculate PCAs")  #required
parser.add_argument("-i","--input_dir", help="input directory", default = "../inputs/baselines/baseline_data/scaPCA_output/")
parser.add_argument("-p","--output_dir", help="out directory", default = "../outputs/baselines/random_forrest/")
parser.add_argument("-o","--outputplot_dir", help="out directory", default = "../outputs/baselines/random_forrest/")
parser.add_argument("-t","--title", help="title that will be written into the output file", default = "title placeholder")
parser.add_argument("-r", "--reset", help="if this is called, the previous results file will be overwritten, otherwise results are appended - call for the first run of the classifier", action="store_true")

parser.add_argument("--n_trees", type = int, default = 100)
parser.add_argument("--max_depth", default = None)
parser.add_argument("--min_samples_split", default = 2)
parser.add_argument("--min_samples_leaf", default = 1)
parser.add_argument("--max_features", default = "auto")

args = parser.parse_args() 





input_dir = args.input_dir
output_dir = args.output_dir
outputplot_dir = args.outputplot_dir
n_trees = args.n_trees
max_depth = args.max_depth
min_samples_split = args.min_samples_split
min_samples_leaf = args.min_samples_leaf
max_features = args.max_features




print(datetime.now().strftime("%H:%M:%S>"), "\n\nStarting sca_randforrest.py")
print(input_dir)





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
print(input_dir)

data = np.loadtxt(open(input_dir + "matrix.tsv"), delimiter="\t")
genes = pd.read_csv(input_dir + "genes.tsv", delimiter = "\t", header = None)
barcodes = pd.read_csv(input_dir + "barcodes.tsv", delimiter = "\t", header = None)

test_index = np.loadtxt(fname = input_dir + "test_index.tsv", dtype = bool)
train_index = np.logical_not(test_index)



# %% Handle Train Test Split

complete_data = data
test_data = data[test_index]
train_data = data[train_index]    

labels = barcodes.iloc[:,1]
test_labels = labels[test_index]
train_labels = labels[train_index]  



# %%
print(datetime.now().strftime("%H:%M:%S>"), "starting classification...")



forest = RandomForestClassifier(n_estimators = n_trees,
                                criterion = "gini",
                                max_depth = max_depth,
                                min_samples_split = min_samples_split,
                                min_samples_leaf = min_samples_leaf,
                                max_features = max_features
                                )

forest.fit(train_data, train_labels)

prediction = forest.predict(test_data)

    
# %%



num_correct = sum(test_labels == prediction)
accuracy = num_correct/len(prediction)





# %% NOTE: THIS WAY OF PLOTTING IS very SLOW. AVOID IT IN THE FUTURE

if not os.path.exists(outputplot_dir):
    print(datetime.now().strftime("%H:%M:%S>"), "Creating Outputplot Directory...")
    os.makedirs(outputplot_dir)




truth = np.array(test_labels == prediction)


plt.figure()
for i in range(len(prediction)):
    if(truth[i]):
        plt.scatter(test_data[i,0], test_data[i,1], c = "k", s = 20, marker = ".", alpha = 0.5, label = "tru")
    else:
        plt.scatter(test_data[i,0], test_data[i,1], c = "r", s = 40, marker = "x", label = "fa")
        
#plt.legend(["correct","incorrect"])
#plt.legend(labels = ["tru", "fa"])
plt.show()
plt.savefig(outputplot_dir + "correct_assignments.png")


# %% Output


if not os.path.exists(output_dir):
    print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Directory...")
    os.makedirs(output_dir)


print(datetime.now().strftime("%H:%M:%S>"), "writing data to output file...")


if args.reset:
    file = open(output_dir + "random_forest_mult.txt", "w")
else:
    file = open(output_dir + "random_forest_mult.txt", "a")
    file.write("\n")
    file.write("\n")
    file.write("\n")
    file.write("\n")
    file.write("\n")    

file.write("######" + args.title + "######\n")
file.write("input_data from " + input_dir + "\n")
file.write("Accuracy = " + str(accuracy) + "\t(" + str(num_correct) + "/" + str(len(prediction)) + ")\n")
file.close()



# %% 
print(datetime.now().strftime("%H:%M:%S>"), "sca_randforest.py terminated successfully\n")



