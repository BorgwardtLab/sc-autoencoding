# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 14:20:07 2020

@author: Mike Toreno II
"""




import argparse

parser = argparse.ArgumentParser(description = "program to preprocess the raw singlecell data")  
parser.add_argument("--title", default = "Placeholder", type = str, help = "the overlying name of the analysis (e.g. baseline, autoencoder, SCA, BCA...). This will change the names of the output plots")

parser.add_argument("--dbscan_results", help="directory of the data - enter <skip> (without brackets) to skip.", default = "skip")
parser.add_argument("--kmcluster_results", help="directory of the data - enter <skip> (without brackets) to skip.", default = "skip")
parser.add_argument("--classification_results", help="directory of the data - enter <skip> (without brackets) to skip.", default = "skip")
parser.add_argument("--random_forest_results", help="directory of the data - enter <skip> (without brackets) to skip.", default = "skip")
#parser.add_argument("--general_input", default = "off", help="instead of entering all directories individually, respect the data structure and only give the overlying directory. Will overwrite all individual directories when entered")
parser.add_argument("--general_input", default = "../outputs/baselines/", help="instead of entering all directories individually, respect the data structure and only give the overlying directory. Will overwrite all individual directories when entered")

parser.add_argument("-o","--output_dir", help="output directory", default = "../outputs/baselines/")
args = parser.parse_args()


title = args.title
dbscan_dir = args.dbscan_results
kmclust_dir = args.kmcluster_results
classification_dir = args.classification_results
randfor_dir = args.random_forest_results


if args.general_input != "off":
    dbscan_dir = args.general_input + "dbscan/"
    kmclust_dir = args.general_input + "kmcluster/"
    classification_dir = args.general_input + "ova_classification/"
    randfor_dir = args.general_input + "random_forest/"



output_dir = args.output_dir + "figures/"
import os
os.makedirs(output_dir, exist_ok=True)






import matplotlib.pyplot as plt
import pandas as pd






# %%
if not dbscan_dir == "skip":
    
    print("yolo")
    
    



else: 
    print("dbscan was skipped")
    
    
    
    
    
    
    
    
# %%
if not kmclust_dir == "skip":
    
    with open(kmclust_dir + "km_clustering_results.txt", "r") as file:
        string = file.read()
    count = int(string.count("###############################")/2)
 
    purities = []
    recalls = []
    names = []
    
    
    for i in range(count):
        
        pointer1 = string.find("###############################")
        pointer2 = string.find("#", pointer1 + 39)
        
        
        name = string[pointer1 + 39:pointer2]
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
        string = string[pointer2:]
    

    ### grouped barplot

    # create pandas dataframe
    data = {"Purity": purities,
            "Recall": recalls}
    panda = pd.DataFrame(data, index = names)
    
    # plot barplot
    panda.plot.bar(rot=0, ylim = [0,1])
    plt.savefig(output_dir + title + "_km_clustering")
    

else: 
    print("kmclust was skipped")
    
    
    
    
    

# %%
if not classification_dir == "skip":
    print("yolo")
    
    



else: 
    print("ova_classification was skipped")
    
    








# %%
if not randfor_dir == "skip":
    
    with open(randfor_dir + "random_forest_mult.txt", "r") as file:
        string = file.read()
        
    count = int(string.count("####")/2)
 

    accuracies = []
    names = []
    
    for i in range(count):
        pointer1 = string.find("######")
        pointer2 = string.find("######", pointer1+1)
        
        name = string[pointer1 + 6:pointer2]
        

        print(name)
        names.append(name)
        
        pointer1 = string.find("=")
        pointer2 = string.find("(", pointer1)


        accuracy = float(string[pointer1+1:pointer2])
        accuracies.append(accuracy)
        print(accuracy)

        pointer1 = string.find(")")
        string = string[pointer1 + 1:]  #+1, otherwise it would always find the same ")"


    
    # Barplot the thing
    plt.figure()
    plt.bar(names, accuracies)
    plt.ylabel("correctly asigned fraction")
    plt.title(title + ": evaluated by random forest" )
    plt.ylim(0,1)
    plt.show()
    plt.savefig(output_dir + title + "_random_forest")




else: 
    print("random_forest was skipped")
    
    











































