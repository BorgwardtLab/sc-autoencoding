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
from sklearn import svm



import matplotlib.pyplot as plt




try:
    os.chdir(os.path.dirname(sys.argv[0]))
except:
    pass



parser = argparse.ArgumentParser(description = "Support Vector Machine")  #required
parser.add_argument("-t","--title", help="title that will be written into the output file", default = "placeholder")

parser.add_argument("--limit_dims", default = 0, help="number of input dimensions to consider", type = int)

parser.add_argument("-i","--input_dir", help="input directory", default = "../inputs/baseline_data/scaPCA_output/")
parser.add_argument("-p","--output_dir", help="out directory", default = "../outputs/results/svm/")
#parser.add_argument("-r", "--reset", help="if this is called, the previous results file will be overwritten, otherwise results are appended - call for the first run of the classifier", action="store_true")
args = parser.parse_args() 



source_input_dir = args.input_dir
source_output_dir = args.output_dir
source_outputplot_dir = args.output_dir + "plots/"

firstrun = True   # always on, now that multiple splits get written in one file anyway. Deaktivates itself in the first run.



# these only matter if the legacy mode is run
input_dir = source_input_dir
output_dir = source_output_dir
outputplot_dir = source_outputplot_dir





# %% handle splits

print(datetime.now().strftime("%H:%M:%S>"), "Starting sca_svm.py with n_Trees = {numcom:d}".format(numcom = -99))    

# determine number of splits
num_splits = 0
cancel = False


directory = source_input_dir + "split_" + str(num_splits + 1)
if os.path.isdir(directory) == False:
    print("ERROR: NO SPLITS DETECTED")
    print(directory)        
    sys.exit()
    
    
else:
    while True:
        num_splits += 1
        directory = source_input_dir + "split_" + str(num_splits + 1)
        
        isdirectory = os.path.isdir(directory)
        
        if isdirectory == False:
            print(datetime.now().strftime("%H:%M:%S>"), str(num_splits) + " splits detected\n")    
            break





# %% Start the loop

#panda = pd.DataFrame(index = ["Accuracy"])
pandas = pd.DataFrame(columns=["Accuracy"])


for split in range(1, num_splits + 1):
    
    print(datetime.now().strftime("%H:%M:%S>"), "Starting split #" + str(split))       
    
 
    input_dir = source_input_dir + "split_" + str(split) + "/"
    output_dir = source_output_dir
    outputplot_dir = source_outputplot_dir
    data_dir = output_dir + "dataframes/"


    
    # %% Read Input data
    print(datetime.now().strftime("%H:%M:%S>"), "reading input data...")
    print(input_dir)
    
    data = np.loadtxt(open(input_dir + "matrix.tsv"), delimiter="\t")
    genes = pd.read_csv(input_dir + "genes.tsv", delimiter = "\t", header = None)
    barcodes = pd.read_csv(input_dir + "barcodes.tsv", delimiter = "\t", header = None)
    
    test_index = np.loadtxt(fname = input_dir + "test_index.tsv", dtype = bool)
    train_index = np.logical_not(test_index)
    

    if args.limit_dims > 0:
        if args.limit_dims <= data.shape[1]:
            data = data[:,0:args.limit_dims]
            print("restricting input dimensions")
        else:
            print("cannot restrict dims. Limit dims = {:d}, input dimension = {:d}".format(args.limit_dims, data.shape[1]))
            
    

    
    # %% Handle Train Test Split
    complete_data = data.copy()
    test_data = data[test_index]
    train_data = data[train_index]    
    
    labels = barcodes.iloc[:,1]
    test_labels = labels[test_index]
    train_labels = labels[train_index]  

    
    # %%
    print(datetime.now().strftime("%H:%M:%S>"), "starting classification with svm...")

    clf = svm.SVC()
    clf.fit(train_data, train_labels)
    
    prediction = clf.predict(test_data)
    


 
    # %%

    num_correct = sum(test_labels == prediction)
    accuracy = num_correct/len(prediction)
    
    
    
    #panda["Split_" + str(split)] = accuracy
    newrow = pd.Series({"Accuracy": accuracy}).rename("Split_"+str(split))
    pandas = pandas.append(newrow)
    
    
    
    
    
    
    # %%        
    # evaluate per celltype
    test_labels = np.array(test_labels)
    
    for celltype in np.unique(test_labels):
        
        current_set_indexes = np.where(test_labels == celltype)
        current_labels = test_labels[current_set_indexes]
        current_prediction = prediction[current_set_indexes]
    
        current_accuracy = sum(current_labels == current_prediction)/len(current_labels)
    
    
        # now this is the tricky part lmao
        pandas.loc["Split_"+str(split),celltype] = current_accuracy
        
    
    
    
    

    # %% NOTE: THIS WAY OF PLOTTING IS very SLOW. AVOID IT IN THE FUTURE
    figurename_appendix = "_{:s}_split{:d}".format(args.title, split)
    
    
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
    
    plt.title("{:s}_s{:d}: accuracy = {:f}".format(args.title, split, accuracy))                
    #plt.legend(["correct","incorrect"])
    #plt.legend(labels = ["tru", "fa"])
    plt.show()
    plt.savefig(outputplot_dir + "correct_assignments" + figurename_appendix)




    # %% Human Output

    if not os.path.exists(output_dir):
        print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Directory...")
        os.makedirs(output_dir)
    
    
    print(datetime.now().strftime("%H:%M:%S>"), "writing data to output file...")
    
    
    if firstrun:
        filename = output_dir + "svm_" + args.title + ".txt"
        
        if os.path.exists(filename):
            separator = "\n\n\n\n"
        else:
            separator = ""
        
        file = open(filename, "a")

        file.write(separator)
        file.write("\n##############################################################\n")
        file.write(datetime.now().strftime("%d. %b %Y, %H:%M:%S \t - " + args.title) + "\n\n")

        firstrun = False 
    else:
        file = open(output_dir + "svm_" + args.title + ".txt", "a")
        file.write("\n")
        file.write("\n")
        
    file.write("###### " + "split_" + str(split) + " ######\n")
    file.write("input_data from " + input_dir + "\n")
    file.write("Accuracy = " + str(accuracy) + "\t(" + str(num_correct) + "/" + str(len(prediction)) + ")\n")
    file.close()

    print("")


# Machine Output
os.makedirs(data_dir, exist_ok=True)
pandas.to_csv(data_dir + "svm_" + args.title + ".tsv", sep = "\t", index = True, header = True)

print(datetime.now().strftime("%H:%M:%S>"), "sca_svm.py terminated successfully")
    

# second plot: accuracy per celltype
# I wanted to make a fancy barplot here, but I decided that I don't care for now. 
plt.figure()
pandas.iloc[:,1:].plot()
plt.title(args.title)
plt.ylabel("Accuracies")
plt.savefig(outputplot_dir + "celltype_accuracies_" + args.title)







