# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 14:20:07 2020

@author: Mike Toreno II
"""





import argparse

parser = argparse.ArgumentParser(description = "program to preprocess the raw singlecell data")  
parser.add_argument("--title", default = "placeholder", type = str, help = "prefix for filenames")
parser.add_argument("--plottitle", default = "placeholder", type = str, help = "prefix for plottitles")

parser.add_argument("--hierarch_results", help="directory of the data - enter <skip> (without brackets) to skip.", default = "skip")
parser.add_argument("--dbscan_results", help="directory of the data - enter <skip> (without brackets) to skip.", default = "skip")
parser.add_argument("--kmcluster_results", help="directory of the data - enter <skip> (without brackets) to skip.", default = "skip")
parser.add_argument("--random_forest_results", help="directory of the data - enter <skip> (without brackets) to skip.", default = "skip")
parser.add_argument("--svm_results", help="directory of the data - enter <skip> (without brackets) to skip.", default = "skip")


parser.add_argument("--general_input", default = "skip", help="instead of entering all directories individually, respect the data structure and only give the overlying directory. Will overwrite all individual directories when entered")
parser.add_argument("--unsorted", action='store_true', help="to avoid the internal ordering according to the custom order")
parser.add_argument("--dbscan_loose", action = "store_true")

parser.add_argument("-o","--output_dir", help="output directory", default = "D:/Dropbox/Internship/gitrepo/outputs/results/visualized_results/")
args = parser.parse_args()



if args.title == "placeholder":
    fileext = ""
else:
    fileext = "_" + args.title

if args.plottitle == "placeholder":
    titleext = ""
else:
    titleext = " " + args.plottitle



dbscan_dir = args.dbscan_results
kmclust_dir = args.kmcluster_results
randfor_dir = args.random_forest_results
svm_dir = args.svm_results
hierarch_dir = args.hierarch_results 


if args.general_input != "skip":
    #dbscan_dir = args.general_input + "dbscan/"
    kmclust_dir = args.general_input + "kmcluster/"
    hierarch_dir = args.general_input + "hierarchcluster/"
    #classification_dir = args.general_input + "ova_classification/"
    randfor_dir = args.general_input + "random_forest/"
    svm_dir = args.svm_results


output_dir = args.output_dir # + args.title + "/"

import os
import matplotlib.pyplot as plt
import pandas as pd
import glob
import re
import numpy as np
import seaborn as sns
from datetime import datetime
import statistics




custom_order = ["PCA", "LSA", "ICA", "tSNE", "UMAP", "DCA", "SCA", "BCA", "original_data", "denoised_data_DCA"]

# dbscan_dir = "M:/Projects/simon_streib_internship/sc-autoencoding/outputs/dbscan/"

# kmclust_dir = "M:/Projects/simon_streib_internship/sc-autoencoding/outputs/optimization/tsne_nimput/tsne_kmclresult/"
# kmclust_dir = "M:/Projects/simon_streib_internship/sc-autoencoding/outputs/results/kmcluster/"
# kmclust_dir = "D:/Dropbox/Internship/gitrepo/outputs/kmcluster/"

# randfor_dir = "M:/Projects/simon_streib_internship/sc-autoencoding/outputs/experiments/losses/randomforest_result/"

# svm_dir = "D:/Dropbox/Internship/gitrepo/outputs/results/svm/"

# hierarch_dir = "D:/Dropbox/Internship/gitrepo/outputs/results/hierarchical/"




    
    
    
    
    
    
    
    
# %%
if not kmclust_dir == "skip":
    print(datetime.now().strftime("%H:%M:%S>"), "Visualizing kmcluster...")
    names = []
    dataframes = []
    
    for filepath in sorted(glob.iglob(kmclust_dir + "dataframes/kmcluster_*.tsv")):
        filepath = filepath.replace('\\' , "/") # for some reason, it changes the last slash to backslash
        #print(filepath)
        search = re.search("dataframes/kmcluster_(.*?).tsv", filepath)
        if search:
            name = search.group(1) # to get only the matched charactesr
            names.append(name)
            #print(name)
            
            newframe = pd.read_csv(filepath, delimiter = "\t", header = 0, index_col = 0)
            newframe["Technique"] = name
            
            dataframes.append(newframe)
        else:
            print("some error with the input files of kmcluster visualizer")
            
    ##### sorting code for custom sort order. (maybe I should deactivate sorting?)    
    
    
    if len(dataframes) == 0:
        print("no datasets were found. Please recheck your input directory")
        import sys
        sys.exit()
    
    
     ##### sort the accuracies. I'm sorry, but it has to b. It's much nicer, and I don't know a better way to sort than this.
    if args.unsorted == False:
        ordered = []
        new_names = []
        
        for name in custom_order:
            if name in names:
                idx = names.index(name)
                ordered.append(dataframes[idx])
                new_names.append(names[idx])
            else:
                pass
            
        # make sure, that all names were present in the custom_order source variable
        if len(new_names) != len(names):
            for name in names:
                if name in custom_order:
                    pass
                else:
                    idx = names.index(name)
                    ordered.append(dataframes[idx])
                    new_names.append(names[idx])
                    #print("please add {:s} to the custom order variable".format(name))

        names = new_names
        dataframes = ordered

    
# %% Prepare DF for seaborn boxplots

    names_fold = []

    # boxplot
    redundancy_frame = pd.DataFrame()

    # lineplots
    f1weight = []
    f1weight_stdv = []    
    nmi_scores = []
    nmi_scores_stdv = []
    
    
    # pieplot
    sizes_pie = []
    names_pie = []

    # num_celltypes barplot
    num_ct = []
    num_ct_low = []
    num_ct_total = []


    for i in range(len(dataframes)):        
        print("Sanity Check: {:s} / {:s}".format(np.array(dataframes[i].loc[:,"Technique"])[0], names[i]))
        
        # extract num_reps
        highest_fold = max(np.array(dataframes[i].loc[:,"Fold"]))
        newname = names[i] + "\n({:d}reps)".format(highest_fold)
        names_fold.append(newname)        
        
        # boxplot
        purities = dataframes[i].loc[:,"Purity"]
        recalls = dataframes[i].loc[:,"Recall"]
        combined = np.array(purities.append(recalls))
        df1 = dataframes[i].copy()
        df2 = dataframes[i].copy()
        df1["PuRicall"] = purities
        df1["Metric"] = "-Purity"
        df2["PuRicall"] = recalls
        df2["Metric"] = "-Recall"
        new_df = pd.concat([df1, df2])
        redundancy_frame = pd.concat([redundancy_frame, new_df])
        
        # # pieplot
        # sizes_pie.append(sizes)
        # names_pie.append(np.array(dataframes[i].loc[:,"Most common label"]))


        df = dataframes[i]
        nfolds = max(df.loc[:,"Fold"])
            
        # lienplots
        nmis_per_fold = []
        f1weight_per_fold = []
        
        for fold in range(1,nfolds + 1):
            is_myfold = df["Fold"]==fold
            dffold = df[is_myfold]

            nmis_per_fold.append(dffold.loc[:,"NMI"][0])
            fscores = np.array(dffold.loc[:,"F1-score"])
            sizes = np.array(dffold.loc[:,"Size"])
            sum_sizes = sum(dffold.loc[:,"Size"])
            weighted_F1 = 0
            for j in range(len(sizes)):
                curr = fscores[j] * sizes[j]
                weighted_F1 += curr
            weighted_F1 = weighted_F1/sum_sizes
            f1weight_per_fold.append(weighted_F1)
            
        mean = statistics.mean(np.array(nmis_per_fold))
        stdv = statistics.pstdev(np.array(nmis_per_fold))
        nmi_scores.append(mean)
        nmi_scores_stdv.append(stdv)
        
        mean = statistics.mean(np.array(f1weight_per_fold))
        stdv = statistics.pstdev(np.array(f1weight_per_fold))
        f1weight.append(mean)
        f1weight_stdv.append(stdv)
    
           
    
        # barplot numct
        sum_k = 0
        sum_uniques = 0
        sum_lows = 0
        
        for fold in range(1,nfolds + 1):

            is_myfold = df["Fold"]==fold
            dffold = df[is_myfold]
            
            sum_k += len(dffold)

            
            celltypes = np.array(dffold.loc[:,"Most common label"])
            unique = np.unique(celltypes, return_counts=False)
            sum_uniques += len(unique)
        
            sizes_2 = np.array(dffold.loc[:,"Size"])
            sum_lows += sum(sizes_2 < 50)

    
    
            
        num_ct.append(sum_uniques/nfolds)
        num_ct_low.append(sum_lows/nfolds)
        num_ct_total.append(sum_k/nfolds)




    # %% first_figure

    n_rows = 3

    fig, axs = plt.subplots(nrows = n_rows, ncols = 1, figsize = [1.2*len(dataframes) + 2, 4.0 * n_rows], sharex=True)
    fig.subplots_adjust(hspace=0.5) 
    fig.suptitle("km-Clustering result of {:d} clusterings".format(len(dataframes)) + titleext, size = "xx-large", weight = "black")
    
    # boxplots
    sns.set_style("whitegrid")
    sns.boxplot(ax = axs[0], x="Technique", y="PuRicall", hue="Metric", data=redundancy_frame, palette="Set2")
    sns.stripplot(ax = axs[0], x="Technique", y="PuRicall", hue="Metric", data=redundancy_frame, palette="Set2", dodge = True, edgecolor = "black", linewidth = 0.3)
    
    handles, labels = axs[0].get_legend_handles_labels() # legend, use to only show half the legend
    axs[0].set_ylabel("Purity & Recall")
    axs[0].set_xlabel("")
    axs[0].legend(handles = handles[0:2], labels = ["Purity", "Recall"], loc = "lower right")
    axs[0].set_title("Purity and Recall Boxplots")
    axs[0].tick_params(labelbottom = True)
    
    
    
    # lineplot
    axs[1].plot(names_fold, f1weight, color = "b", linestyle ="-", marker = "o", markersize = 4)
    axs[1].errorbar(x = names_fold, y = f1weight, yerr = f1weight_stdv, capsize = 10, elinewidth = 0.5, capthick = 1)
    axs[1].set_ylabel("Average F1-Score (normalized for clustersizes)")
    axs[1].set_title("Average F1 Score")
    axs[1].tick_params(labelbottom = True)
    try:
        axs[1].plot(names_fold, nmi_scores, color = "r", linestyle ="-", marker = "D", markersize = 4)
        axs[1].errorbar(x = names_fold, y = nmi_scores, yerr = nmi_scores_stdv, capsize = 10, elinewidth = 0.5, capthick = 1)
        axs[1].legend(labels = ["F1-score","NMI"])
    except:
        print("no NMI info available")
        pass



    # baprlot num_unique CT
    #axs[2].set_yscale("log")

    diff = np.array(num_ct)-np.array(num_ct_low)
    # avoid negatives in diffs
    zeros = len(diff)*[0]
    diff = np.maximum(diff, zeros)
    
    axs[2].bar(x = names_fold, height = num_ct_total, width = 0.2, color = "black", alpha = 0.1) # k
    axs[2].bar(x = names_fold, height = diff, width = 0.45)                                      # the non-low ones
    axs[2].bar(x = names_fold, height = num_ct_low, bottom = diff, width = 0.45, color = "red")  # the low ones
    
    
    #handl, leggy = axs[2].get_legend_handles_labels()
    axs[2].legend(labels = ["k","unique clusters (>50)","unique clusters (<50)"])
    
    
    
    #axs[2].axhline(10, alpha = 0.5, c = "red")
    axs[2].set_title("Unique cluster labels (red = cluster < 50 cells)")
    axs[2].set_ylabel("Number of unique cluster labels")
    axs[2].tick_params(labelbottom = True)








    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_dir + "kmcluster_result" + fileext + ".png")
 
    


else: 
    print("kmclust was skipped")
    
    














print(datetime.now().strftime("%H:%M:%S>"), "Visualizer has finished successfully")






