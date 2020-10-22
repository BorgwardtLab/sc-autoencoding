# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 23:52:35 2020

@author: Mike Toreno II
"""


# %% Load Data

import argparse


parser = argparse.ArgumentParser(description = "Do Latent Semantic Analysis")  #required
parser.add_argument("-n","--num_components", help="the number of LSA components to calculate", type = int, default = 30)
parser.add_argument("-i","--input_dir", help="input directory", default = "../inputs/data/preprocessed_data/")
parser.add_argument("-o","--output_dir", help="output directory", default = "../inputs/baseline_data/scaLSA_output/")
parser.add_argument("-p","--outputplot_dir", help="plot directory", default = "../outputs/baseline_data/scaLSA_output/")
parser.add_argument("--kfold", help="", action="store_true")
parser.add_argument("--mode", help="chose k-split, unsplit or both", choices=['complete','split','nosplit'], default = "complete")
args = parser.parse_args() #required




import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import matplotlib.cm as cm # colourpalette
import sys
from sklearn.decomposition import TruncatedSVD



try:
    os.chdir(os.path.dirname(sys.argv[0]))
except:
    pass



source_input_dir = args.input_dir
source_output_dir = args.output_dir
source_outputplot_dir = args.outputplot_dir
component_name = "LS"
num_lsa = args.num_components





#args.mode = "nosplit"



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









    
    
# %% SPLIT  
    
    
if split == True:
    print(datetime.now().strftime("%H:%M:%S>"), "Starting sca_LSA.py (split) with num_components = {numcom:d}".format(numcom = num_lsa))    
 

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
        
        
        data = np.loadtxt(open(input_dir + "matrix.tsv"), delimiter="\t")

        file = open(input_dir + "genes.tsv", "r")
        genes = file.read().split("\n")
        file.close()
        genes.remove("") 
        
        barcodes = pd.read_csv(input_dir + "barcodes.tsv", delimiter = "\t", header = None)
        labels = barcodes.iloc[:,1]
        

        test_index = np.loadtxt(fname = input_dir + "test_index.tsv", dtype = bool)
        train_index = np.logical_not(test_index)
        
        
        
        # %% replacing data with testdata only
        
        original_data = data.copy()
        
        testdata = data[test_index]
        traindata = data[train_index]
        
        
        
        # %% doing LSA
        
        print(datetime.now().strftime("%H:%M:%S>"), "scaling data...")
        
        myscaler = StandardScaler()
        traindata =  myscaler.fit_transform(traindata)
        testdata = myscaler.transform(testdata)
        
        
        print(datetime.now().strftime("%H:%M:%S>"), "Calculating LSA...")
        svd = TruncatedSVD(n_components = num_lsa)
        svd.fit(traindata)
        
        #lsa = latent semantic analysis
        lsa_train = svd.transform(traindata)
        lsa_test = svd.transform(testdata)
        
        
        explained_variance = svd.explained_variance_ratio_
        
        
        
        #%% Plots

        if not os.path.exists(outputplot_dir):
            print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Plot Directory...")
            os.makedirs(outputplot_dir)
            
        
        
        
        ### Create Plot
        print(datetime.now().strftime("%H:%M:%S>"), "drawing plots...")
        targets = set(labels) # 
        
        
        # construct dataframe for 2d plot # TRAIN
        df_train = pd.DataFrame(data = lsa_train[:,[0,1]], columns = [component_name + '_1', component_name + '_2'])
        df_train['celltype'] = np.array(labels[train_index])
        
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(1,1,1) 
        ax.set_xlabel(component_name + '1 (' + str(round(explained_variance[0]*100, 3)) + "% of variance)", fontsize = 15)
        ax.set_ylabel(component_name + '2 (' + str(round(explained_variance[1]*100, 3)) + "% of variance)", fontsize = 15)
        ax.set_title('Most Powerful LSAs', fontsize = 20)
        
        colors = cm.rainbow(np.linspace(0, 1, len(targets)))
        for target, color in zip(targets,colors):
            indicesToKeep = df_train['celltype'] == target
            ax.scatter(df_train.loc[indicesToKeep, component_name + '_1']
                       , df_train.loc[indicesToKeep, component_name + '_2']
                       , c = color.reshape(1,-1)
                       , s = 1)
        ax.legend(targets)
        ax.grid()
        plt.savefig(outputplot_dir + "LSA_result_train.png")
        
        
         # construct dataframe for 2d plot TEST
        df_test = pd.DataFrame(data = lsa_test[:,[0,1]], columns = [component_name + '_1', component_name + '_2'])
        df_test['celltype'] = np.array(labels[test_index])
        
        fig = plt.figure(figsize = (8,8))
        ax = fig.add_subplot(1,1,1) 
        ax.set_xlabel(component_name + '1 (' + str(round(explained_variance[0]*100, 3)) + "% of variance)", fontsize = 15)
        ax.set_ylabel(component_name + '2 (' + str(round(explained_variance[1]*100, 3)) + "% of variance)", fontsize = 15)
        ax.set_title('Most Powerful LSAs', fontsize = 20)
        
        colors = cm.rainbow(np.linspace(0, 1, len(targets)))
        for target, color in zip(targets,colors):
            indicesToKeep = df_train['celltype'] == target
            ax.scatter(df_train.loc[indicesToKeep, component_name + '_1']
                       , df_train.loc[indicesToKeep, component_name + '_2']
                       , c = color.reshape(1,-1)
                       , s = 1)
        ax.legend(targets)
        ax.grid()
        plt.savefig(outputplot_dir + "LSA_result_test.png")
        
        
                
        
        
        ### Save Variances
        print(datetime.now().strftime("%H:%M:%S>"), "saving explained variances...")
        explained_sum = np.cumsum(explained_variance)
        
        file = open(outputplot_dir + 'explained_variances_train.log', 'w')
        for i in range(len(explained_variance)):
            text = (str(i + 1) + "\t" + str(explained_variance[i]) + "\t" + str(explained_sum[i]) + "\n")
            file.write(text)
        file.close()
            
            
            
        ### Scree Plots
        
        perc_var = (explained_variance * 100)
        
        labelz = [str(x) for x in range(1, len(perc_var)+1)]
        
        
        plt.figure(figsize=[16,8])
        plt.bar(x = range(1, len(perc_var)+1), height = perc_var, tick_label = labelz)
        plt.ylabel('Percentage of explained variance')
        plt.xlabel('Principal component')
        plt.title('Scree plot')
        plt.show()    
        plt.savefig(outputplot_dir + "LSA_scree_plot_train_all.png")
            
            
            
            
            
        if num_lsa > 50:
            how_many = 30;
            perc_var = (explained_variance * num_lsa)
            perc_var = perc_var[0:how_many]
        
            labelz = [str(x) for x in range(1, len(perc_var)+1)]
            
            plt.figure(figsize=[16,8])
            plt.bar(x = range(1, len(perc_var)+1), height = perc_var, tick_label = labelz)
            plt.ylabel('Percentage of explained variance')
            plt.xlabel('Linear components (?)')
            plt.title('Scree plot')
            plt.show()    
            plt.savefig(outputplot_dir + "LSA_scree_plot_train_top30.png")    
            
            
            
         
            
        # Loading scores for PC1
        
        how_many = 10
        
        loading_scores = pd.Series(svd.components_[0], index = genes)
        
        
        
        sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
        top_genes = sorted_loading_scores[0:how_many].index.values
            
        
        file = open(outputplot_dir + 'most_important_genes_train.log', 'w')
        for i in range(how_many):
            text = (str(top_genes[i]) + "\t" + str(sorted_loading_scores[i]) + "\n")
            file.write(text)
        file.close()
        
        
        
        # %% Saving Data
        
        # Preparations
        
        outdata = np.zeros(shape = (original_data.shape[0], num_lsa))
        outdata[train_index] = lsa_train
        outdata[test_index] = lsa_test
        
        new_genes =  []
        for i in range(num_lsa):
            new_genes.append("LSA_" + str(i + 1))
        genes = pd.DataFrame(new_genes)
        

        
        # saving data
  
        if not os.path.exists(output_dir):
            print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Directory...")
            os.makedirs(output_dir)
            
        print(datetime.now().strftime("%H:%M:%S>"), "Saving output...")
        
        np.savetxt(output_dir + "matrix.tsv", outdata, delimiter = "\t")
        genes.to_csv(output_dir + "genes.tsv", sep = "\t", index = False, header = False)
        barcodes.to_csv(output_dir + "barcodes.tsv", sep = "\t", index = False, header = False)
        np.savetxt(output_dir + "test_index.tsv", test_index, fmt = "%d")

            
        print(datetime.now().strftime("%H:%M:%S>"), "sca_LSA.py (split " + str(split) + ") terminated successfully\n")
        
    








# %% NO SPLIT
if nosplit == True:
    
    print(datetime.now().strftime("%H:%M:%S>"), "Starting sca_LSA.py (nosplit) with num_components = {numcom:d}".format(numcom = num_lsa))    
 
    input_dir = source_input_dir + "no_split/"
    output_dir = source_output_dir + "no_split/"
    outputplot_dir = source_outputplot_dir + "no_split/"
    

    
    
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
    
    assert os.path.isfile(input_dir + "test_index.tsv") == False




    
    # %% doing LSA
    
    print(datetime.now().strftime("%H:%M:%S>"), "scaling data...")
    
    myscaler = StandardScaler()
    data =  myscaler.fit_transform(data)
    
    
    print(datetime.now().strftime("%H:%M:%S>"), "Calculating LSA...")
    svd = TruncatedSVD(n_components = num_lsa)
    svd.fit(data)
    

    lsa = svd.transform(data)

    explained_variance = svd.explained_variance_ratio_
    
    
    
    #%% Outputs
    
    if not os.path.exists(outputplot_dir):
        print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Directory...")
        os.makedirs(outputplot_dir)

    
    ### Create Plot
    print(datetime.now().strftime("%H:%M:%S>"), "drawing plots...")
    targets = set(labels) # 
    
    
    # construct dataframe for 2d plot
    df = pd.DataFrame(data = lsa[:,[0,1]], columns = [component_name + '_1', component_name + '_2'])
    df['celltype'] = labels
    
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel(component_name + '1 (' + str(round(explained_variance[0]*100, 3)) + "% of variance)", fontsize = 15)
    ax.set_ylabel(component_name + '2 (' + str(round(explained_variance[1]*100, 3)) + "% of variance)", fontsize = 15)
    ax.set_title('Most Powerful LSAs', fontsize = 20)
    
    colors = cm.rainbow(np.linspace(0, 1, len(targets)))
    for target, color in zip(targets,colors):
        indicesToKeep = df['celltype'] == target
        ax.scatter(df.loc[indicesToKeep, component_name + '_1']
                   , df.loc[indicesToKeep, component_name + '_2']
                   , c = color.reshape(1,-1)
                   , s = 1)
    ax.legend(targets)
    ax.grid()
    plt.savefig(outputplot_dir + "LSA_plot.png")
    
    
    
    
    
    ### Save Variances
    print(datetime.now().strftime("%H:%M:%S>"), "saving explained variances...")
    explained_sum = np.cumsum(explained_variance)
    
    file = open(outputplot_dir + 'explained_variances.log', 'w')
    for i in range(len(explained_variance)):
        text = (str(i + 1) + "\t" + str(explained_variance[i]) + "\t" + str(explained_sum[i]) + "\n")
        file.write(text)
    file.close()
        
        
        
        
    ### Scree Plots
    perc_var = (explained_variance * 100)
    labelz = [str(x) for x in range(1, len(perc_var)+1)]
    
    plt.figure(figsize=[16,8])
    plt.bar(x = range(1, len(perc_var)+1), height = perc_var, tick_label = labelz)
    plt.ylabel('Percentage of explained variance')
    plt.xlabel('Principal component')
    plt.title('Scree plot')
    plt.show()    
    plt.savefig(outputplot_dir + "LSA_scree_plot_all.png")
        
        
        
        
        
    if num_lsa > 50:
        how_many = 30;
        perc_var = (explained_variance * num_lsa)
        perc_var = perc_var[0:how_many]
    
        labelz = [str(x) for x in range(1, len(perc_var)+1)]
        
        plt.figure(figsize=[16,8])
        plt.bar(x = range(1, len(perc_var)+1), height = perc_var, tick_label = labelz)
        plt.ylabel('Percentage of explained variance')
        plt.xlabel('Linear components (?)')
        plt.title('Scree plot')
        plt.show()    
        plt.savefig(outputplot_dir + "LSA_scree_plot_top30.png")    
        
        
        
     
        
    # Loading scores for PC1
    
    how_many = 10
    loading_scores = pd.Series(svd.components_[0], index = genes)
    
    sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
    top_genes = sorted_loading_scores[0:how_many].index.values
        
    
    file = open(outputplot_dir + 'most_important_genes.log', 'w')
    for i in range(how_many):
        text = (str(top_genes[i]) + "\t" + str(sorted_loading_scores[i]) + "\n")
        file.write(text)
    file.close()
    
    
       
    
    
    # %% saving data
    
    new_genes =  []
    for i in range(num_lsa):
        new_genes.append("LSA_" + str(i + 1))
    genes = pd.DataFrame(new_genes)


    if not os.path.exists(output_dir):
        print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Directory...")
        os.makedirs(output_dir)
        
        
    print(datetime.now().strftime("%H:%M:%S>"), "Saving output...")
    
    np.savetxt(output_dir + "matrix.tsv", lsa, delimiter = "\t")
    genes.to_csv(output_dir + "genes.tsv", sep = "\t", index = False, header = False)
    barcodes.to_csv(output_dir + "barcodes.tsv", sep = "\t", index = False, header = False)

    print(datetime.now().strftime("%H:%M:%S>"), "sca_LSA.py (nosplit) terminated successfully")
    





