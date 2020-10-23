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
from datetime import datetime


from sklearn.metrics import confusion_matrix

from collections import Counter







try:
    os.chdir(os.path.dirname(sys.argv[0]))
except:
    pass



parser = argparse.ArgumentParser(description = "calculate PCAs")  #required
parser.add_argument("-k","--kfold", help="the number of k-folds to test the classifier on", type = int, default = 5)
parser.add_argument("-i","--input_dir", help="input directory", default = "../inputs/baselines/baseline_data/scaPCA_output/")
parser.add_argument("-p","--output_dir", help="plot directory", default = "../outputs/baselines/ova_classification/")
parser.add_argument("-c","--classifier", help="helptext", default = "forest", choices = ["logreg","lda","forest"])
parser.add_argument("-t","--title", help="title that will be written into the output file", default = "title placeholder")
parser.add_argument("-r", "--reset", help="if this is called, the previous results file will be overwritten, otherwise results are appended - call for the first run of the classifier", action="store_true")
args = parser.parse_args() #required



input_dir = args.input_dir
output_dir = args.output_dir
kfold = args.kfold
classifier = args.classifier





print(datetime.now().strftime("\n%d. %b %Y, %H:%M:%S>"), "Starting sca_classifcation.py with " + classifier)






# %% functions


def classify(traindata, trainlabels, testdata, testlabels, classifier):
    if classifier == "logreg":
        metric = logistic_regression(traindata, trainlabels, testdata, testlabels)
    elif classifier == "lda":
        metric = linear_discriminant_analysis(traindata, trainlabels, testdata, testlabels)  
    elif classifier == "forest":
        metric = random_forest(traindata, trainlabels, testdata, testlabels)
    else:
        print("illegal method lol, this wasn't supposed to be possible")
    return metric




def random_forest(traindata, trainlabels, testdata, testlabels):
    from sklearn.ensemble import RandomForestClassifier
    
    max_depth = 20
    max_features = min(30, len(data[0]))
    min_samples_leaf = 1; # keep in mind, as integer it works differently than as float
    min_samples_split = 2; #same
    n_trees = 150;
    
    classifier = RandomForestClassifier(n_estimators = n_trees, criterion = "gini",
                                        max_depth = max_depth, max_features = max_features,
                                        min_samples_leaf = min_samples_leaf, min_samples_split = min_samples_split)
    classifier.fit(traindata, trainlabels)
    y_pred = classifier.predict(testdata) 
    
    metric = compute_metrics(testlabels, y_pred)
    return metric




def linear_discriminant_analysis(traindata, trainlabels, testdata, testlabels):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis()
    lda.fit(traindata, trainlabels)
    
    prediction = lda.predict(testdata)
    
    metric = compute_metrics(testlabels, prediction)
    return metric




def logistic_regression(traindata, trainlabels, testdata, testlabels):
    from sklearn.linear_model import LogisticRegression
    
    ### scaler = StandardScaler().fit(unscaled_traindata);
    ### traindata = scaler.transform(unscaled_traindata);    
    
    log_regressor = LogisticRegression(penalty = "l2", 
                                       C = 1.0, # regularizer, = none
                                       solver = "liblinear") # with other solvers, apparently multiclass is possible
    ### testdata = scaler.transform(unscaled_testdata)
    log_regressor.fit(X = traindata, y = trainlabels)
    prediction = log_regressor.predict(testdata)

    metric = compute_metrics(testlabels, prediction)
    return metric






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
    #print("the line is {0:d}/({0:d}+{1:d})".format(tp, fp))
    #print(["tp, fp, fn, tn"])
    #print([tp, fp, fn, tn])
    #print(Counter(y_true)) # remove me
    
    
    if tp + fp == 0:
        print("WARNING: this classifier predicts only negative class")
    
    
    
    recall = tp/(tp+fn)
    f1score = 2*recall*precision/(recall + precision)
    
    size = sum(y_true)
    
    results = [accuracy, precision, recall, f1score, tn, fp, fn, tp, size]
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


labelset = list(set(labels))


# %%
print(datetime.now().strftime("%H:%M:%S>"), "starting classification...")

pandas = pd.DataFrame(index = ["Accuracy ", "Precision", "Recall   ", "F1 Score ", "TN", "FP", "FN", "TP", "size"])



# %%

'''old way, using stratifiedkfold to generate test and train indices is appended at the very end.'''

'''no longer doing kfolds, instead just once with the train and test labels from file'''


print("overview of all labels:")
print(Counter(labels))
print(Counter(train_labels))
print(Counter(test_labels))



for label in labelset:
    # print("\ncurrent label is: " + str(label))     # remove me
    binary_trainlabels = (np.array(train_labels == label))
    binary_testlabels = (np.array(test_labels == label))
    
    result = classify(train_data, binary_trainlabels, test_data, binary_testlabels, classifier)
    
    pandas[label] = result
    






# %% gENERATE Output


if not os.path.exists(output_dir):
    print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Directory...")
    os.makedirs(output_dir)


print(datetime.now().strftime("%H:%M:%S>"), "writing data to output file...")


if args.reset:
    file = open(output_dir + "one_versus_all_classification_" + classifier + ".txt", "w")
else:
    file = open(output_dir + "one_versus_all_classification_" + classifier + ".txt", "a")
    file.write("\n")
    file.write("\n")
    file.write("\n")
    file.write("\n")
    file.write("\n")    

file.write("######" + args.title + "######\n")
file.write("Classifier " + classifier + ", input_data from " + input_dir + "\n")


averages = pandas.mean(axis = 1)

file.write("\nAverage Accuracy: \t" + '{:.4f}'.format(averages.iloc[0]))
file.write("\nAverage Precision:\t" + '{:.4f}'.format(averages.iloc[1]))
file.write("\nAverage Recall:   \t" + '{:.4f}'.format(averages.iloc[2]))
file.write("\nAverage F1 Score: \t" + '{:.4f}'.format(averages.iloc[3]))


file = open(output_dir + "one_versus_all_classification_" + classifier + ".txt", "a")
file.write("\n\nComplete Dataframe:\n")
file.close()
pandas.to_csv(output_dir + "one_versus_all_classification_" + classifier + ".txt", mode = "a", sep = "\t")


# %% 
print(datetime.now().strftime("%H:%M:%S>"), "sca_classification.py terminated successfully\n")





























# %% appendix



# kfolder = StratifiedKFold(n_splits=kfold, shuffle=True)

# for trainindices, testindices in kfolder.split(train_data, train_labels):
#     traindata = data[trainindices,:]
#     trainlabels = barcodes.iloc[trainindices,:]
#     testdata = data[testindices,:]
#     testlabels = barcodes.iloc[testindices,:]
    
#     foldnumber = foldnumber + 1
    
#     for label in list(set(train_labels)):
#         binary_trainlabels = (np.array(trainlabels.iloc[:,1]) == label)
#         binary_testlabels = (np.array(testlabels.iloc[:,1]) == label)
        
#         result = classify(traindata, binary_trainlabels, testdata, binary_testlabels, classifier)
        
#         resultname = "Fold " + str(foldnumber) + ": " + label
#         pandas[resultname] = result
        

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
# file.write("input_data from " + input_dir + ", Classifier " + classifier + "\n")


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





