# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 14:20:07 2020

@author: Mike Toreno II
"""




import argparse

parser = argparse.ArgumentParser(description = "program to preprocess the raw singlecell data")  
parser.add_argument("--title", default = "placeholder", type = str, help = "the overlying name of the analysis (e.g. baseline, autoencoder, SCA, BCA...). This will change the names of the output plots, and their folder")

parser.add_argument("--dbscan_results", help="directory of the data - enter <skip> (without brackets) to skip.", default = "skip")
parser.add_argument("--kmcluster_results", help="directory of the data - enter <skip> (without brackets) to skip.", default = "skip")
parser.add_argument("--random_forest_results", help="directory of the data - enter <skip> (without brackets) to skip.", default = "skip")
parser.add_argument("--general_input", default = "skip", help="instead of entering all directories individually, respect the data structure and only give the overlying directory. Will overwrite all individual directories when entered")

parser.add_argument("-o","--output_dir", help="output directory", default = "../outputs/visualized_results/")
args = parser.parse_args()


title = args.title
dbscan_dir = args.dbscan_results
kmclust_dir = args.kmcluster_results

randfor_dir = args.random_forest_results



if args.general_input != "skip":
    dbscan_dir = args.general_input + "dbscan/"
    kmclust_dir = args.general_input + "kmcluster/"
    #classification_dir = args.general_input + "ova_classification/"
    randfor_dir = args.general_input + "random_forest/"



output_dir = args.output_dir + title + "/"

import os
import matplotlib.pyplot as plt
import pandas as pd
import glob
import re
import numpy as np



#randfor_dir = "../outputs/random_forest/"
kmclust_dir = "../outputs/kmcluster/"





# %%
if not dbscan_dir == "skip":
    
    with open(dbscan_dir + "dbscan_clustering_results.txt", "r") as file:
        string = file.read()
    count = int(string.count("######")/2)
 
    purities = []
    recalls = []
    names = []
    
    
    for i in range(count):
        
        pointer1 = string.find("######")
        pointer2 = string.find("#", pointer1 + 6)
        
        
        name = string[pointer1 + 6:pointer2]
        names.append(name)
        print(name)
    
    
        # purity
        pointer1 = string.find("Average Purity:")
        pointer2 = string.find("(", pointer1)
        
        purity = float(string[pointer1+15:pointer2])
        purities.append(purity)
        print(purity)
        
        
        # recall
        pointer1 = string.find("Average Recall:")
        pointer2 = string.find("(", pointer1)
        
        recall = float(string[pointer1+15:pointer2])
        recalls.append(recall)
        print(recall)
        
        
        # shorten string
        string = string[pointer2+30:]





    ## grouped barplot

    # create pandas dataframe
    data = {"Purity": purities,
            "Recall": recalls}
    panda = pd.DataFrame(data, index = names)
    
    # plot barplot
    panda.plot.bar(rot=0, ylim = [0,1])
    plt.savefig(output_dir + title + "_dbscan")
    



else: 
    print("dbscan was skipped")
    
    
    
    
    
    
    
    
# %%
if not kmclust_dir == "skip":
    
    filelist = []
    names = []
    dataframes = []
    
    
    for filepath in glob.iglob(randfor_dir + "dataframes/randomforest_*.tsv"):
        filepath = filepath.replace('\\' , "/") # for some reason, it changes the last slash to backslash
        print(filepath)
        
        filelist.append(filepath)
        
        
        search = re.search("randomforest_(.*).tsv", filepath)
        if search:
            name = search.group(1) # to get only the matched charactesr
            names.append(name)


        accuracies[name] = pd.read_csv(filepath, delimiter = "\t", header = 0, index_col = 0)

    
    
    
    
    
    
    
    
    
    
    
    # %%
    
    accuracies.T.plot.bar(rot = 0, color={"Split_1": "crimson", "Split_2": "firebrick", "Split_3": "lightcoral"})
    plt.title("Random Forests")
    plt.ylabel("Accuracies")
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_dir + "random_forest_result.png")
    

 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

else: 
    print("kmclust was skipped")
    
    
    # with open(kmclust_dir + "km_clustering_results.txt", "r") as file:
    #     string = file.read()
    # count = int(string.count("###############################")/2)
 
    # purities = []
    # recalls = []
    # names = []
    
    
    # for i in range(count):
        
    #     pointer1 = string.find("###############################")
    #     pointer2 = string.find("#", pointer1 + 39)
        
        
    #     name = string[pointer1 + 39:pointer2]
    #     names.append(name)
    #     print(name)
    
    #     # purity
    #     pointer1 = string.find("Average Purity:")
    #     pointer2 = string.find("(", pointer1)
        
    #     purity = float(string[pointer1+15:pointer2])
    #     purities.append(purity)
    #     print(purity)
        
    #     # recall
    #     pointer1 = string.find("Average Recall:")
    #     pointer2 = string.find("(", pointer1)
        
    #     recall = float(string[pointer1+15:pointer2])
    #     recalls.append(recall)
    #     print(recall)
        
        
    #     # shorten string
    #     string = string[pointer2:]
    

    # ### grouped barplot

    # # create pandas dataframe
    # data = {"Purity": purities,
    #         "Recall": recalls}
    # panda = pd.DataFrame(data, index = names)
    
    # # plot barplot
    # panda.plot.bar(rot=0, ylim = [0,1])
    # plt.savefig(output_dir + title + "_km_clustering")
    













    
    
    
    









# %%
if not randfor_dir == "skip":
    
    filelist = []
    names = []
    accuracies = pd.DataFrame(index = ["Split_1", "Split_2", "Split_3"])
    
    
    for filepath in glob.iglob(randfor_dir + "dataframes/randomforest_*.tsv"):
        filepath = filepath.replace('\\' , "/") # for some reason, it changes the last slash to backslash
        print(filepath)
        
        filelist.append(filepath)
        
        
        search = re.search("randomforest_(.*).tsv", filepath)
        if search:
            name = search.group(1) # to get only the matched charactesr
            names.append(name)


        accuracies[name] = pd.read_csv(filepath, delimiter = "\t", header = 0, index_col = 0)

    
    
    
    
    accuracies.T.plot.bar(rot = 0, color={"Split_1": "crimson", "Split_2": "firebrick", "Split_3": "lightcoral"})
    plt.title("Random Forests")
    plt.ylabel("Accuracies")
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_dir + "random_forest_result.png")
    




else: 
    print("random_forest was skipped")
     
































