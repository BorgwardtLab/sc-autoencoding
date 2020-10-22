# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 23:52:35 2020

@author: Mike Toreno II
"""


# %% Load Data

import argparse


parser = argparse.ArgumentParser(description = "calculates a UMAP embedding")  #required
parser.add_argument("-n","--num_components", help="the number of coordinates to calculate", type = int, default = 2)
parser.add_argument("-d","--dimensions", type = int, default = 0, help="enter a value here to restrict the number of input dimensions to consider, otherwise all available PC's will be used")
parser.add_argument("-i","--input_dir", help="input directory", default = "../inputs/baseline_data/scaPCA_output/")
parser.add_argument("-o","--output_dir", help="output directory", default = "../inputs/baseline_data/scaUMAP_output/")
parser.add_argument("-p","--outputplot_dir", help="plot directory", default = "../outputs/baseline_data/scaUMAP_output/")
parser.add_argument("-v","--verbosity", help="level of verbosity", default = 3, choices = [0, 1, 2, 3], type = int)
parser.add_argument("--mode", help="chose k-split, unsplit or both", choices=['complete','split','nosplit'], default = "complete")
args = parser.parse_args() #required




source_input_dir = args.input_dir
source_output_dir = args.output_dir
source_outputplot_dir = args.outputplot_dir
component_name = "UMAP"
num_components = args.num_components

dims = args.dimensions



import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import matplotlib.cm as cm # colourpalette
import umap
import sys


try:
    os.chdir(os.path.dirname(sys.argv[0]))
except:
    pass


# args.mode == "nosplit"


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


    






# %% KFOLD = TRUE    
    
if split == True:
    print(datetime.now().strftime("%H:%M:%S>"), "Starting sca_UMAP.py (split) with num_components = {numcom:d}".format(numcom = num_components))    
 
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
        
        
        # load genes (for last task, finding most important genes)
        file = open(input_dir + "genes.tsv", "r")
        genes = file.read().split("\n")
        file.close()
        genes.remove("") 
        
        
        barcodes = pd.read_csv(input_dir + "barcodes.tsv", delimiter = "\t", header = None)
        labels = barcodes.iloc[:,1]
        
        
        test_index = np.loadtxt(fname = input_dir + "test_index.tsv", dtype = bool)
        train_index = np.logical_not(test_index)
        
        
        
        
        # %%
        
        original_data = data
        
        testdata = data[test_index]
        traindata = data[train_index]
        
        
        
        
        
        # %%
        
        
        print(datetime.now().strftime("%H:%M:%S>"), "scaling data...")
        myscaler = StandardScaler()
        traindata =  myscaler.fit_transform(traindata)
        testdata = myscaler.transform(testdata)
        
        
        print(datetime.now().strftime("%H:%M:%S>"), "calculating UMAP...")
        reducer = umap.UMAP(verbose = args.verbosity, n_components = num_components)
        new_traindata = reducer.fit_transform(traindata)
        new_testdata = reducer.transform(testdata)
        
        
        
        
        
        #%% Plots
     
        
        if not os.path.exists(outputplot_dir):
            print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Directory...")
            os.makedirs(outputplot_dir)
    
        ### Create Plot
        print(datetime.now().strftime("%H:%M:%S>"), "drawing plots...")
        targets = set(labels) # what it will draw in plot, previously it was targets = ['b_cells' ... 'cytotoxic_t'], now its dynamic :*
                
           
        # construct dataframe for 2d plot
        train_df = pd.DataFrame(data = new_traindata[:,[0,1]], columns = [ component_name + ' 1', component_name + ' 2'])
        train_df['celltype'] = np.array(labels[train_index])
        
    
        
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(1,1,1) 
        ax.set_xlabel(component_name + " 1 ", fontsize = 15)
        ax.set_ylabel(component_name + " 2 ", fontsize = 15)
        ax.set_title('Most Powerful '+ component_name +'s', fontsize = 20)
        colors = cm.rainbow(np.linspace(0, 1, len(targets)))
        for target, color in zip(targets,colors):
            indicesToKeep = train_df['celltype'] == target
            ax.scatter(train_df.loc[indicesToKeep, component_name + ' 1']
                       , train_df.loc[indicesToKeep, component_name + ' 2']
                       , c = color.reshape(1,-1)
                       , s = 5)
        ax.legend(targets)
        ax.grid()
        plt.savefig(outputplot_dir + "/UMAP_plot_train.png")
        
        
        
        # construct dataframe for 2d plot
        test_df = pd.DataFrame(data = new_testdata[:,[0,1]], columns = [ component_name + ' 1', component_name + ' 2'])
        test_df['celltype'] = np.array(labels[test_index])
        
    
        
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(1,1,1) 
        ax.set_xlabel(component_name + " 1 ", fontsize = 15)
        ax.set_ylabel(component_name + " 2 ", fontsize = 15)
        ax.set_title('Most Powerful '+ component_name +'s', fontsize = 20)
        colors = cm.rainbow(np.linspace(0, 1, len(targets)))
        for target, color in zip(targets,colors):
            indicesToKeep = test_df['celltype'] == target
            ax.scatter(test_df.loc[indicesToKeep, component_name + ' 1']
                       , test_df.loc[indicesToKeep, component_name + ' 2']
                       , c = color.reshape(1,-1)
                       , s = 5)
        ax.legend(targets)
        ax.grid()
        plt.savefig(outputplot_dir + "/UMAP_plot_train.png")
        
        
        
        
        
        # import seaborn as sns
        # colourdictionary = dict(zip(list(targets), range(len(targets))))
        
        # plt.scatter(
        #     newdata[:, 0],
        #     newdata[:, 1],
        #     s = 1,
        #     alpha = 0.5,
        #     marker = ",",
        #     c=[sns.color_palette()[x] for x in df.celltype.map(colourdictionary)])
        # plt.gca().set_aspect('equal', 'datalim')
        # plt.title('UMAP projection of the single cells with sns', fontsize=24)
        # plt.savefig(outputplot_dir + "/UMAP_plot_scatter.png")   
        
        
    
    
        # %% recombine data
        
        
        outdata = np.zeros(shape = (original_data.shape[0], num_components))
        
        outdata[train_index] = new_traindata
        outdata[test_index] = new_testdata
        
    
    
        # %% output
      
        new_genes =  []
        for i in range(num_components):
            new_genes.append("UMAP_Component_" + str(i + 1) + "_(from_" + str(dims) + "PCs)")
        genes = pd.DataFrame(new_genes)
        
    
        if not os.path.exists(output_dir):
            print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Directory...")
            os.makedirs(output_dir)    
        
        print(datetime.now().strftime("%H:%M:%S>"), "Saving output...")
        np.savetxt(output_dir + "matrix.tsv", outdata, delimiter = "\t")
        genes.to_csv(output_dir + "genes.tsv", sep = "\t", index = False, header = False)
        barcodes.to_csv(output_dir + "barcodes.tsv", sep = "\t", index = False, header = False)
        np.savetxt(output_dir + "test_index.tsv", test_index, fmt = "%d")
        
        
        print(datetime.now().strftime("%H:%M:%S>"), "sca_UMAP terminated successfully\n")
        
        
    
    
    
    
    
    




     

# %% NO SPLIT
if nosplit == True:
    
    print(datetime.now().strftime("%H:%M:%S>"), "Starting sca_UMAP.py (nosplit) with num_components = {numcom:d}".format(numcom = num_components))    
 
    input_dir = source_input_dir + "no_split/"
    output_dir = source_output_dir + "no_split/"
    outputplot_dir = source_outputplot_dir + "no_split/"
    
    
    
    
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
    
    assert os.path.isfile(input_dir + "test_index.tsv") == False


    
    if dims < 0:
        data = data[:,dims]
    elif dims == 0:
        dims = data.shape[1]
    



    # %% do 

    print(datetime.now().strftime("%H:%M:%S>"), "scaling data...")
    myscaler = StandardScaler()
    data =  myscaler.fit_transform(data)

    
    
    print(datetime.now().strftime("%H:%M:%S>"), "calculating UMAP...")
    reducer = umap.UMAP(verbose = args.verbosity, n_components = num_components)
    newdata = reducer.fit_transform(data)
   
    
    
    #%% Plots

    if not os.path.exists(outputplot_dir):
        print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Directory...")
        os.makedirs(outputplot_dir)
        
    
    # construct dataframe for 2d plot
    df = pd.DataFrame(data = newdata[:,[0,1]], columns = [ component_name + ' 1', component_name + ' 2'])
    df['celltype'] = labels
    
    
    
    
    ### Create Plot
    print(datetime.now().strftime("%H:%M:%S>"), "drawing plots...")
    targets = set(labels) # what it will draw in plot, previously it was targets = ['b_cells' ... 'cytotoxic_t'], now its dynamic :*
    
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel(component_name + " 1 ", fontsize = 15)
    ax.set_ylabel(component_name + " 2 ", fontsize = 15)
    ax.set_title('Most Powerful '+ component_name +'s', fontsize = 20)
    colors = cm.rainbow(np.linspace(0, 1, len(targets)))
    for target, color in zip(targets,colors):
        indicesToKeep = df['celltype'] == target
        ax.scatter(df.loc[indicesToKeep, component_name + ' 1']
                   , df.loc[indicesToKeep, component_name + ' 2']
                   , c = color.reshape(1,-1)
                   , s = 5)
    ax.legend(targets)
    ax.grid()
    plt.savefig(outputplot_dir + "/UMAP_plot.png")
    

    
    import seaborn as sns
    colourdictionary = dict(zip(list(targets), range(len(targets))))
    
    plt.scatter(
        newdata[:, 0],
        newdata[:, 1],
        s = 1,
        alpha = 0.5,
        marker = ",",
        c=[sns.color_palette()[x] for x in df.celltype.map(colourdictionary)])
    plt.gca().set_aspect('equal', 'datalim')
    plt.title('UMAP projection of the single cells with sns', fontsize=24)
    plt.savefig(outputplot_dir + "/UMAP_plot_scatter.png")   
    
    
    
    # %% output
    
    new_genes =  []
    for i in range(num_components):
        new_genes.append("UMAP_Component_" + str(i + 1) + "_(from_" + str(dims) + "PCs)")
    genes = pd.DataFrame(new_genes)
    

    
    if not os.path.exists(output_dir):
        print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Directory...")
        os.makedirs(output_dir)    
    
    print(datetime.now().strftime("%H:%M:%S>"), "Saving output...")
    np.savetxt(output_dir + "matrix.tsv", newdata, delimiter = "\t")
    genes.to_csv(output_dir + "genes.tsv", sep = "\t", index = False, header = False)
    barcodes.to_csv(output_dir + "barcodes.tsv", sep = "\t", index = False, header = False)

    print(datetime.now().strftime("%H:%M:%S>"), "sca_UMAP terminated successfully")
    
    
    


