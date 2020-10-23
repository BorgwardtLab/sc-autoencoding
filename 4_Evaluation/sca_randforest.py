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


from sklearn.ensemble import RandomForestClassifier



import matplotlib.pyplot as plt




try:
    os.chdir(os.path.dirname(sys.argv[0]))
except:
    pass



parser = argparse.ArgumentParser(description = "calculate PCAs")  #required
parser.add_argument("-i","--input_dir", help="input directory", default = "../inputs/baseline_data/scaPCA_output/")
parser.add_argument("-p","--output_dir", help="out directory", default = "../outputs/random_forrest/")
#parser.add_argument("-o","--outputplot_dir", help="out directory", default = "../outputs/random_forrest/PCA/")
parser.add_argument("-t","--title", help="title that will be written into the output file", default = "placeholder")
#parser.add_argument("-r", "--reset", help="if this is called, the previous results file will be overwritten, otherwise results are appended - call for the first run of the classifier", action="store_true")
parser.add_argument("--legacy", help="can be called to let the script run only on one fold.", action="store_true")


parser.add_argument("--n_trees", type = int, default = 100)
parser.add_argument("--max_depth", default = None)
parser.add_argument("--min_samples_split", default = 2)
parser.add_argument("--min_samples_leaf", default = 1)
parser.add_argument("--max_features", default = "auto")

args = parser.parse_args() 


n_trees = args.n_trees
max_depth = args.max_depth
min_samples_split = args.min_samples_split
min_samples_leaf = args.min_samples_leaf
max_features = args.max_features






source_input_dir = args.input_dir
source_output_dir = args.output_dir
source_outputplot_dir = args.output_dir + "plots/"

firstrun = True   # always on, now that multiple splits get written in one file anyway. Deaktivates itself in the first run.





# these only matter if the legacy mode is run
input_dir = source_input_dir
output_dir = source_output_dir
outputplot_dir = source_outputplot_dir





# %% handle splits


if args.legacy == False:    # "normal mode"
  
    print(datetime.now().strftime("%H:%M:%S>"), "Starting sca_randforrest.py with n_Trees = {numcom:d}".format(numcom = n_trees))    

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
        output_dir = source_output_dir #+ "split_" + str(split) + "/"
        #outputplot_dir = source_outputplot_dir + "split_" + str(split) + "/"
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
    
        
        # %% Handle Train Test Split
        complete_data = data.copy()
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
        
        
        
        #panda["Split_" + str(split)] = accuracy
        newrow = pd.Series({"Accuracy": accuracy}).rename("Split_"+str(split))
        pandas = pandas.append(newrow)
        
        
        
        
    
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
            
            filename = output_dir + "randomforest_" + args.title + ".txt"
            
            
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
            file = open(output_dir + "randomforest_" + args.title + ".txt", "a")
            file.write("\n")
            file.write("\n")
            
        
        file.write("###### " + "split_" + str(split) + " ######\n")
        file.write("input_data from " + input_dir + "\n")
        file.write("Accuracy = " + str(accuracy) + "\t(" + str(num_correct) + "/" + str(len(prediction)) + ")\n")
        file.close()
    
        print("")
    
    
    # Machine Output
    os.makedirs(data_dir, exist_ok=True)
    #panda.to_csv(data_dir + "randomforest_" + args.title + ".tsv", sep = "\t", index = True, header = True)
    pandas.to_csv(data_dir + "randomforest_" + args.title + ".tsv", sep = "\t", index = True, header = True)

    print(datetime.now().strftime("%H:%M:%S>"), "sca_randforest.py terminated successfully")
        



















# %% legacy function
if args.legacy == True:
    
    input_dir = source_input_dir
    output_dir = source_output_dir
    outputplot_dir = source_outputplot_dir
    

    
    print(datetime.now().strftime("\n\n%d. %b %Y, %H:%M:%S>"), "Starting sca_randforrest.py")
    print(input_dir)
    
    
    
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
    
    if firstrun:
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
    print(datetime.now().strftime("%H:%M:%S>"), "sca_randforest.py terminated successfully")
    
    

