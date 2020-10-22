# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 00:49:30 2020

@author: Mike Toreno II
"""



import argparse

parser = argparse.ArgumentParser(description = "calculate ICAs")  #required
parser.add_argument("-n","--num_components", help="the number of ICAs to calculate", type = int, default = 100)
parser.add_argument("-i","--input_dir", help="input directory", default = "../inputs/baseline_data/scaPCA_output/")
parser.add_argument("-o","--output_dir", help="output directory", default = "../inputs/baseline_data/scaICA_output/")
parser.add_argument("-p","--outputplot_dir", help="plot directory", default = "../outputs/baseline_data/scaICA_output/")
parser.add_argument("--mode", help="chose k-split, unsplit or both", choices=['complete','split','nosplit'], default = "complete")
args = parser.parse_args() #required



import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import matplotlib.cm as cm # colourpalette

import sys



try:
    os.chdir(os.path.dirname(sys.argv[0]))
except:
    pass



source_input_dir = args.input_dir
source_output_dir = args.output_dir
source_outputplot_dir = args.outputplot_dir

component_name = "IC"
num_components = args.num_components


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




    
    
# %% KFOLD == TRUE    

if split == True:
    print(datetime.now().strftime("%H:%M:%S>"), "Starting sca_PCA.py (split) with num_components = {numcom:d}".format(numcom = num_components))    
 

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
            print(directory)
            
            isdirectory = os.path.isdir(directory)
            
            if isdirectory == False:
                print(datetime.now().strftime("%H:%M:%S>"), str(num_splits) + " splits detected\n")    
                break
     
    
# %% loop through splits
 
    for split in range(1, num_splits + 1):
        
        print(datetime.now().strftime("%H:%M:%S>"), "Starting split #" + str(split))       
        
     
        input_dir = source_input_dir + "split_" + str(split) + "/"
        output_dir = source_output_dir + "split_" + str(split) + "/"
        outputplot_dir = source_outputplot_dir + "split_" + str(split) + "/"
        
        
        
        
        # %% Read Input data
        print(datetime.now().strftime("%H:%M:%S>"), "reading input data...")
        
        
        matrix_file = input_dir + "matrix.tsv"
        data = np.loadtxt(open(matrix_file), delimiter="\t")
        
        
        file = open(input_dir + "genes.tsv", "r")
        genes = file.read().split("\n")
        file.close()
        genes.remove("") 
        
        
        barcodes = pd.read_csv(input_dir + "barcodes.tsv", delimiter = "\t", header = None)
        labels = barcodes.iloc[:,1]
        
        
        test_index = np.loadtxt(fname = input_dir + "test_index.tsv", dtype = bool)
        train_index = np.logical_not(test_index)
        
        
        
        

        original_data = data.copy()
        
        testdata = data[test_index]
        traindata = data[train_index]
        
        

        # %% do ICA
        
        
        print(datetime.now().strftime("%H:%M:%S>"), "scaling data...")
        myscaler = StandardScaler()
        traindata =  myscaler.fit_transform(traindata)
        testdata = myscaler.transform(testdata)
        
        
        print(datetime.now().strftime("%H:%M:%S>"), "calculating independant components...")
        ica = FastICA(n_components=num_components)
        train_ICs = ica.fit_transform(traindata)
        test_ICs = ica.transform(testdata)
        
        
        
        
        
        #%% Plots
        
        if not os.path.exists(outputplot_dir):
            print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Plot Directory...")
            os.makedirs(outputplot_dir)
            
        
        
        ### Create Plot
        print(datetime.now().strftime("%H:%M:%S>"), "drawing plots...")
        targets = set(labels) # what it will draw in plot, previously it was targets = ['b_cells' ... 'cytotoxic_t'], now its dynamic :*
        
        # construct dataframe for 2d plot TRAINDATA
        df_train = pd.DataFrame(data = train_ICs[:,[0,1]], columns = [component_name + '_1', component_name + '_2'])
        df_train['celltype'] = np.array(labels[train_index]) # conversion necessary to drop the "index" form labels, otherwise it gets wrongly attached
        
        
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(1,1,1) 
        ax.set_xlabel(component_name + "_1", fontsize = 15)
        ax.set_ylabel(component_name + "_2", fontsize = 15)
        ax.set_title('Most Powerful ICAs', fontsize = 20)
        colors = cm.rainbow(np.linspace(0, 1, len(targets)))
        for target, color in zip(targets,colors):
            indicesToKeep = df_train['celltype'] == target
            ax.scatter(df_train.loc[indicesToKeep, component_name + '_1']
                       , df_train.loc[indicesToKeep, component_name + '_2']
                       , c = color.reshape(1,-1)
                       , s = 5)
        ax.legend(targets)
        ax.grid()
        plt.savefig(outputplot_dir + "ICA_plot_train.png")
        
        
        
        
        # construct dataframe for 2d plot TESTDATA
        df_test = pd.DataFrame(data = test_ICs[:,[0,1]], columns = [component_name + '_1', component_name + '_2'])
        df_test['celltype'] = np.array(labels[test_index]) # conversion necessary to drop the "index" form labels, otherwise it gets wrongly attached
        
        
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(1,1,1) 
        ax.set_xlabel(component_name + "_1", fontsize = 15)
        ax.set_ylabel(component_name + "_2", fontsize = 15)
        ax.set_title('Most Powerful ICAs', fontsize = 20)
        colors = cm.rainbow(np.linspace(0, 1, len(targets)))
        for target, color in zip(targets,colors):
            indicesToKeep = df_test['celltype'] == target
            ax.scatter(df_test.loc[indicesToKeep, component_name + '_1']
                       , df_test.loc[indicesToKeep, component_name + '_2']
                       , c = color.reshape(1,-1)
                       , s = 5)
        ax.legend(targets)
        ax.grid()
        plt.savefig(outputplot_dir + "ICA_plot_test.png")
        
        
        
        
        
        
        # Loading scores for PC1
        
        how_many = 10
        
        #generate index
        index = []
        for i in range(len(ica.components_[0,:])):
            index.append("PC " + str(i))
            
        loading_scores = pd.Series(ica.components_[0,:], index = index)
        sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
        top_genes = sorted_loading_scores[0:how_many].index.values
            
        
        file = open(outputplot_dir + 'most_important_components_train.log', 'w')
        for i in range(how_many):
            text = (str(top_genes[i]) + "\t" + str(sorted_loading_scores[i]) + "\n")
            file.write(text)
        file.close()
        
        
        
        # %% Recombine Data
        outdata = np.zeros(shape = (original_data.shape[0], num_components))
        
        outdata[train_index] = train_ICs
        outdata[test_index] = test_ICs
        
        
        
        
        
        # %% Saving the data

        new_genes =  []
        for i in range(num_components):
            new_genes.append("IC_" + str(i + 1))
        genes = pd.DataFrame(new_genes)
    


        if not os.path.exists(output_dir):
            print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Directory...")
            os.makedirs(output_dir)    
    
        print(datetime.now().strftime("%H:%M:%S>"), "Saving output...")
        
        np.savetxt(output_dir + "matrix.tsv", outdata, delimiter = "\t")
        genes.to_csv(output_dir + "genes.tsv", sep = "\t", index = False, header = False)
        barcodes.to_csv(output_dir + "barcodes.tsv", sep = "\t", index = False, header = False)
        np.savetxt(output_dir + "test_index.tsv", test_index, fmt = "%d")
        
        print(datetime.now().strftime("%H:%M:%S>"), "sca_ICA.py (split " + str(split) + ") terminated successfully\n")
                








# %% NO SPLIT
if nosplit == True:
    print(datetime.now().strftime("%H:%M:%S>"), "Starting sca_ICA.py (nosplit) with num_components = {numcom:d}".format(numcom = num_components))    
 
    input_dir = source_input_dir + "no_split/"
    output_dir = source_output_dir + "no_split/"
    outputplot_dir = source_outputplot_dir + "no_split/"
        
    
    
    # %% Read Input data
    print(datetime.now().strftime("%H:%M:%S>"), "reading input data...")

    
    assert os.path.isfile(input_dir + "test_index.tsv") == False


    matrix_file = input_dir + "matrix.tsv"
    data = np.loadtxt(open(matrix_file), delimiter="\t")
    
    # load genes (for last task, finding most important genes)
    file = open(input_dir + "genes.tsv", "r")
    genes = file.read().split("\n")
    file.close()
    genes.remove("") 
    
    
    barcodes = pd.read_csv(input_dir + "barcodes.tsv", delimiter = "\t", header = None)
    labels = barcodes.iloc[:,1]
    

    
    
    # %% do ICA
    
    
    print(datetime.now().strftime("%H:%M:%S>"), "scaling data...")
    myscaler = StandardScaler()
    data =  myscaler.fit_transform(data)

    
    print(datetime.now().strftime("%H:%M:%S>"), "calculating independant components...")
    ica = FastICA(n_components=num_components)
    ICs = ica.fit_transform(data)
 
    
    
    
    #%% Outputs
    
    
    
    if not os.path.exists(outputplot_dir):
        print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Directory...")
        os.makedirs(outputplot_dir)
        
    
    
    ### Create Plot
    print(datetime.now().strftime("%H:%M:%S>"), "drawing plots...")
    targets = set(labels) # what it will draw in plot, previously it was targets = ['b_cells' ... 'cytotoxic_t'], now its dynamic :*
    
    # construct dataframe for 2d plot
    df = pd.DataFrame(data = ICs[:,[0,1]], columns = [component_name + '_1', component_name + '_2'])
    df['celltype'] = labels
    
    

    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel(component_name + "_1", fontsize = 15)
    ax.set_ylabel(component_name + "_2", fontsize = 15)
    ax.set_title('Most Powerful ICAs', fontsize = 20)
    colors = cm.rainbow(np.linspace(0, 1, len(targets)))
    for target, color in zip(targets,colors):
        indicesToKeep = df['celltype'] == target
        ax.scatter(df.loc[indicesToKeep, component_name + '_1']
                   , df.loc[indicesToKeep, component_name + '_2']
                   , c = color.reshape(1,-1)
                   , s = 5)
    ax.legend(targets)
    ax.grid()
    plt.savefig(outputplot_dir + "ICA_plot.png")
    
    
    # Loading scores for PC1
    
    how_many = 10
    
    #generate index
    index = []
    for i in range(len(ica.components_[0,:])):
        index.append("PC " + str(i))
        
    loading_scores = pd.Series(ica.components_[0,:], index = index)
    sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
    top_genes = sorted_loading_scores[0:how_many].index.values
        
    
    file = open(outputplot_dir + 'most_important_components.log', 'w')
    for i in range(how_many):
        text = (str(top_genes[i]) + "\t" + str(sorted_loading_scores[i]) + "\n")
        file.write(text)
    file.close()
    
    
    

    
    # %% Saving the data

    new_genes =  []
    for i in range(num_components):
        new_genes.append("IC_" + str(i + 1))
    genes = pd.DataFrame(new_genes)


    if not os.path.exists(output_dir):
        print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Directory...")
        os.makedirs(output_dir)    

    print(datetime.now().strftime("%H:%M:%S>"), "Saving output...")
    np.savetxt(output_dir + "matrix.tsv", ICs, delimiter = "\t") 
    genes.to_csv(output_dir + "genes.tsv", sep = "\t", index = False, header = False)
    barcodes.to_csv(output_dir + "barcodes.tsv", sep = "\t", index = False, header = False)

    print(datetime.now().strftime("%H:%M:%S>"), "sca_ICA.py terminated successfully")
    
    
    
    








