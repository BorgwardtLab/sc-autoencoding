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
# kmclust_dir = "D:/Dropbox/Internship/gitrepo/outputs/kmcluster/"

randfor_dir = "M:/Projects/simon_streib_internship/sc-autoencoding/outputs/experiments/losses/randomforest_result/"

# svm_dir = "D:/Dropbox/Internship/gitrepo/outputs/results/svm/"

# hierarch_dir = "D:/Dropbox/Internship/gitrepo/outputs/results/hierarchical/"



# %%
if not dbscan_dir == "skip":
    print(datetime.now().strftime("%H:%M:%S>"), "Visualizing DBScan...")
    names = []
    dataframes = []
    
    for filepath in sorted(glob.iglob(dbscan_dir + "dataframes/dbscan_*.tsv")):
        filepath = filepath.replace('\\' , "/") # for some reason, it changes the last slash to backslash
        search = re.search("dataframes/dbscan_(.*?).tsv", filepath)
        if search:
            name = search.group(1) # to get only the matched charactesr
            names.append(name)
            
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
    
    
    
    # %%
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
                    print("please add {:s} to the custom order variable".format(name))

        names = new_names
        dataframes = ordered
        
    outliers = []
    
    

    
##### remove outlier line
    dataframes_outliers = dataframes.copy()
    for i in range(len(dataframes)):
        df = dataframes[i]
        if df.iloc[-1,:].name =="outliers":
            
            temp = int(df.iloc[-1,:].loc["Size"])
            outliers.append(temp)
            
            df = df.drop(index = "outliers")  
            dataframes[i] = df
        else:
            outliers.append(0)            
        
                
    
# %% Prepare DF for seaborn boxplots

    # boxplot
    redundancy_frame = pd.DataFrame()
    
    # lineplots
    f1weight = []
    nmi_scores = []
    
    # pieplot
    sizes_pie = []
    names_pie = []
    
    # num_cluster bar plot
    num_clusters = []
    num_clusters_low = []
    
    # outliers pie plot
    sum_sizes_arr = []


    for i in range(len(dataframes)):
        
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
        
        # lineplots
        try:
            nmiscore = dataframes[i].loc[:,"NMI"][0]
            nmiscore = nmiscore[~np.isnan(nmiscore)]
            nmi_scores.append(nmiscore)
        except:
            pass
    
    
        fscores = np.array(dataframes[i].loc[:,"F1-score"])
        fscores = fscores[~np.isnan(fscores)]               # filter out Nan
        sizes = np.array(dataframes[i].loc[:,"Size"])
        sum_sizes = sum(dataframes[i].loc[:,"Size"])
        weighted_F1 = 0
        for j in range(len(sizes)):
            curr = fscores[j] * sizes[j]
            weighted_F1 += curr
        weighted_F1 = weighted_F1/sum_sizes
        f1weight.append(weighted_F1)
        
        # pieplot
        sizes_pie.append(sizes)
        names_pie.append(np.array(dataframes[i].loc[:,"Most common label"]))
        
        # num clusters bar plot
        num_clusters.append(len(dataframes[i]))
        sizes_2 = np.array(dataframes[i].loc[:,"Size"])
        num_clusters_low.append(sum(sizes_2 < 50))
        
        # outlier pie
        sum_sizes_arr.append(sum_sizes)
        

    
    # %%
    dbscan_tight = False
    
    if dbscan_tight:
        n_rows = 5
    else:    
        n_rows = 6
    plotnumber=5 # number of the cluster pie charts
    
        
    fig, axs = plt.subplots(nrows = n_rows, ncols = 1, figsize = [1.2*len(dataframes) + 2, 4.0 * n_rows], sharex = True)
    fig.subplots_adjust(hspace=0.5) 

    fig.suptitle("DBScan_Clustering result"  + titleext, size = "xx-large", weight = "black")
    
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
    axs[1].plot(names, f1weight, color = "b", linestyle ="-", marker = "o", markersize = 4)
    axs[1].set_ylabel("Average F1-Score (normalized for clustersizes)")
    axs[1].set_title("Average F1 Score")
    axs[1].tick_params(labelbottom = True)
    try:
        axs[1].plot(names, nmi_scores, color = "r", linestyle ="-", marker = "D", markersize = 4)
        axs[1].legend(labels = ["F1-score","NMI"])
    except:
        print("no NMI info available")
        pass



    # baprlot num_clusters
    #axs[2].set_yscale("log")
    diff = np.array(num_clusters)-np.array(num_clusters_low)
    axs[2].bar(x = names, height = diff, width = 0.45)    
    axs[2].bar(x = names, height = num_clusters_low, bottom = diff, width = 0.45, color = "red") 
    
    
    #axs[2].axhline(10, alpha = 0.5, c = "red")
    axs[2].set_title("Number of clusters (red = cluster < 50 cells)")
    
    axs[2].axhline(10, alpha = 0.5, c = "red")
    axs[2].set_ylabel("number of clusters")
    axs[2].tick_params(labelbottom = True)

    # outulier fraction:
    axs[3].axis("off")
    axs[3].set_title("Outlier fraction")
    for j in range(len(sizes_pie)):
        fig.add_subplot(n_rows, len(sizes_pie),((3*len(sizes_pie)+j+1)))
        plt.pie(x = [outliers[j], sum_sizes_arr[j]], explode = [0.2,0], labels = None, radius = 1.2, shadow = False, colors=["k","orange"])
        plt.legend(labels=["outliers"], loc = "upper center", bbox_to_anchor= (0.5, 0))
        plt.title(names[j])


    # pieplot
    ############# Variant "tight"
    axs[plotnumber-1].axis("off")
    axs[plotnumber-1].set_title("Cells per cluster [Total: {:d}]".format(max(sum_sizes_arr)), pad = 30)
    axs[plotnumber-1].set_xlabel("Sizes of the found clusters. \n(watch out for very small clusters)")
    for j in range(len(sizes_pie)):
        fig.add_subplot(n_rows*2, len(sizes_pie),((plotnumber - 1)*2*len(sizes_pie)+j+1)) # note, i do double the slpits, that are actually there.
        plt.pie(x = sizes_pie[j], explode = np.ones(len(sizes_pie[j])) * 0.01, labels = None, radius = 1.2, shadow = False)
        if dbscan_tight:
            legdist = -2
            limit = 20
        else:
            legdist = -4
            limit = 40
            axs[5].axis("off")
            
        if len(names_pie[j]) > limit:
                names_pie[j] = names_pie[j][:limit]
                plt.legend(names_pie[j], prop={'size': 8}, loc = "center", bbox_to_anchor = (0.5, legdist), facecolor = "gray", edgecolor = "red")

        else:
            plt.legend(names_pie[j], prop={'size': 8}, loc = "center", bbox_to_anchor = (0.5, legdist))
    

    os.makedirs(output_dir, exist_ok=True)
    if dbscan_tight:
        plt.savefig(output_dir + "dbscan_result_tight" + fileext + ".png")
    else:
        plt.savefig(output_dir + "dbscan_result" + fileext + ".png")

    
# %%
else:
    print("dbscan was skipped")
    
    # with open(dbscan_dir + "dbscan_clustering_results.txt", "r") as file:
    #     string = file.read()
    # count = int(string.count("######")/2)
 
    # purities = []
    # recalls = []
    # names = []
    
    # for i in range(count):
        
    #     pointer1 = string.find("######")
    #     pointer2 = string.find("#", pointer1 + 6)
        
    #     name = string[pointer1 + 6:pointer2]
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
    #     string = string[pointer2+30:]

    # ## grouped barplot
    # # create pandas dataframe
    # data = {"Purity": purities,
    #         "Recall": recalls}
    # panda = pd.DataFrame(data, index = names)
    
    # # plot barplot
    # panda.plot.bar(rot=0, ylim = [0,1])
    # plt.savefig(output_dir + title + "_dbscan")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
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
        for fold in range(1,nfolds):
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
        
        for fold in range(1,nfolds):
            is_myfold = df["Fold"]==fold
            dffold = df[is_myfold]
            
            sum_k += len(dffold)
            
            celltypes = np.array(dffold.loc[:,"Most common label"])
            unique = np.unique(celltypes, return_counts=False)
            sum_uniques += len(unique)
        
            sizes_2 = np.array(dataframes[i].loc[:,"Size"])
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





    
    # %% Pieplots no longer supported :(

    # n_rows = 2
    # plotnumber= 2   # number of the pieplots
    
    
    # fig, axs = plt.subplots(nrows = n_rows, ncols = 1, figsize = [1.2*len(dataframes), 4.0 * n_rows], sharex=True)
    # fig.subplots_adjust(hspace=0.5) 
    # fig.suptitle("km-Clustering result of {:d} clusterings".format(len(dataframes)) + titleext, size = "xx-large", weight = "black")
    
    

    # # pieplot
    # axs[plotnumber-1].axis("off")
    # axs[plotnumber-1].set_title("Cells per cluster [Total: {:d}]".format(sum_sizes), pad = 30)
    # axs[plotnumber-1].set_xlabel("Sizes of the found clusters. \n(watch out for very small clusters)")
    # for j in range(len(sizes_pie)):
    #     fig.add_subplot(n_rows*3, len(sizes_pie),(((plotnumber - 1)*3 +1)*len(sizes_pie)+j+1)) # note, i do double the slpits, that are actually there.
    #     #fig.add_subplot(n_rows, len(sizes_pie),((plotnumber-1)*len(sizes_pie)+j+1))         
    #     plt.pie(x = sizes_pie[j], explode = np.ones(len(sizes_pie[j])) * 0.01, labels = None, radius = 1.2, shadow = False)
    #     plt.title(names[j])
    #     if len(names_pie[j]) > 30:
    #            names_pie[j] = names_pie[j][:30]
    #            plt.legend(names_pie[j], prop={'size': 8}, loc = "center", bbox_to_anchor = (0.5, -1.5), facecolor = "gray", edgecolor = "red")

    #     else:
    #         plt.legend(names_pie[j], prop={'size': 8}, loc = "center", bbox_to_anchor = (0.5, -1.5))

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_dir + "kmcluster_result" + fileext + ".png")
 
    

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
if not hierarch_dir == "skip":
    print(datetime.now().strftime("%H:%M:%S>"), "Visualizing hierarchical clustering...")
    names = []
    dataframes = []
    
    for filepath in sorted(glob.iglob(hierarch_dir + "dataframes/hierarchical_*.tsv")):
        filepath = filepath.replace('\\' , "/") # for some reason, it changes the last slash to backslash
        #print(filepath)
        search = re.search("dataframes/hierarchical_(.*?).tsv", filepath)
        if search:
            name = search.group(1) # to get only the matched charactesr
            names.append(name)
            print(name)
            
            newframe = pd.read_csv(filepath, delimiter = "\t", header = 0, index_col = 0)
            newframe["Technique"] = name
            
            dataframes.append(newframe)
        else:
            print("some error with the input files of hierarch visualizer")
            
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
        for fold in range(1,nfolds):
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
        
        for fold in range(1,nfolds):
            is_myfold = df["Fold"]==fold
            dffold = df[is_myfold]
            
            sum_k += len(dffold)
            
            celltypes = np.array(dffold.loc[:,"Most common label"])
            unique = np.unique(celltypes, return_counts=False)
            sum_uniques += len(unique)
        
            sizes_2 = np.array(dataframes[i].loc[:,"Size"])
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
    axs[2].set_ylabel("Number of unique cluster labels (normalized for nfolds)")
    axs[2].tick_params(labelbottom = True)





    
    # %% Pieplots no longer supported :(

    # n_rows = 2
    # plotnumber= 2   # number of the pieplots
    
    
    # fig, axs = plt.subplots(nrows = n_rows, ncols = 1, figsize = [1.2*len(dataframes), 4.0 * n_rows], sharex=True)
    # fig.subplots_adjust(hspace=0.5) 
    # fig.suptitle("km-Clustering result of {:d} clusterings".format(len(dataframes)) + titleext, size = "xx-large", weight = "black")
    
    

    # # pieplot
    # axs[plotnumber-1].axis("off")
    # axs[plotnumber-1].set_title("Cells per cluster [Total: {:d}]".format(sum_sizes), pad = 30)
    # axs[plotnumber-1].set_xlabel("Sizes of the found clusters. \n(watch out for very small clusters)")
    # for j in range(len(sizes_pie)):
    #     fig.add_subplot(n_rows*3, len(sizes_pie),(((plotnumber - 1)*3 +1)*len(sizes_pie)+j+1)) # note, i do double the slpits, that are actually there.
    #     #fig.add_subplot(n_rows, len(sizes_pie),((plotnumber-1)*len(sizes_pie)+j+1))         
    #     plt.pie(x = sizes_pie[j], explode = np.ones(len(sizes_pie[j])) * 0.01, labels = None, radius = 1.2, shadow = False)
    #     plt.title(names[j])
    #     if len(names_pie[j]) > 30:
    #            names_pie[j] = names_pie[j][:30]
    #            plt.legend(names_pie[j], prop={'size': 8}, loc = "center", bbox_to_anchor = (0.5, -1.5), facecolor = "gray", edgecolor = "red")

    #     else:
    #         plt.legend(names_pie[j], prop={'size': 8}, loc = "center", bbox_to_anchor = (0.5, -1.5))

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_dir + "hierarchical_result" + fileext + ".png")
 
    

else: 
    print("hierarch_dir was skipped")
    







































# %%
if not randfor_dir == "skip":
    print(datetime.now().strftime("%H:%M:%S>"), "Visualizing Random Forest...")
    #filelist = []
    names = []
    accuracies = None
    #print("randfor_dir = {:s}".format(randfor_dir + "dataframes/randomforest_*.tsv"))
    
    for filepath in sorted(glob.iglob(randfor_dir + "dataframes/randomforest_*.tsv")):
        filepath = filepath.replace('\\' , "/") # for some reason, it changes the last slash to backslash
        #print(filepath)
        #filelist.append(filepath)
        
        search = re.search("dataframes/randomforest_(.*?).tsv", filepath)
        if search:
            name = search.group(1) # to get only the matched charactesr
            names.append(name)

        df = pd.read_csv(filepath, delimiter = "\t", header = 0, index_col = 0)
        df = df.loc[:,["Accuracy"]] # double brackets to ensure return of a DF and not of a series
        df = df.rename(columns={"Accuracy": name})

        accuracies = pd.concat([accuracies, df], axis = 1)
    
    if accuracies is None:
        print("no datasets were found. Please recheck your input directory")
        import sys
        sys.exit()
        
    
    # first sort alphabetically
    accuracies = accuracies.reindex(sorted(accuracies.columns, key = str), axis=1)

     ##### sort the accuracies. I'm sorry, but it has to b. It's much nicer, and I don't know a better way to sort than this.
    if args.unsorted == False:
        ordered = pd.DataFrame()
        newnames = []
        for name in custom_order:
            if name in names:
                ordered[name] = accuracies.loc[:,name]
                newnames.append(name)
            else:
                pass
            
        # make sure, that all names were present in the custom_order source variable
        if accuracies.shape[1] != len(newnames):
            for name in names:
                if name in custom_order:
                    pass
                else:
                    ordered[name] = accuracies.loc[:,name]
                    newnames.append(name)
                    print("please add {:s} to the custom order variable".format(name))
        names = newnames
        accuracies = ordered
    
    
    
    
    
    # %% GROUPED BARPLOT
    
    # accuracies.T.plot.bar(rot = 45, figsize = [1*accuracies.shape[1], 6.4]) # , color={"Split_1": "crimson", "Split_2": "firebrick", "Split_3": "lightcoral"}
    # plt.title("Random Forests" + titleext)
    # plt.ylabel("Accuracies")
    # plt.grid(which = "both", axis = "y")
    # plt.legend(loc = "lower right")
    # plt.subplots_adjust(bottom=0.15)
    
    # os.makedirs(output_dir, exist_ok=True)
    # plt.savefig(output_dir + "random_forest_result" + fileext + ".png")


# %% Plot
    avgs = []
    stde = []

    for i in range(accuracies.shape[1]):
        values = np.array(accuracies.iloc[:,i])
        avgs.append(np.mean(values))
        stde.append(np.std(values, ddof = 1))
                    
    plt.figure(figsize = [1*accuracies.shape[1] + 2, 6.4*1.8])
    
    plt.subplot(2,1,1)
    plt.bar(x = names, height = avgs, yerr = stde, alpha = 0.5, color = "red")
    plt.xticks(rotation = 45)
    plt.title("Random Forests" + titleext)
    plt.ylabel("Accuracies")
    plt.grid(which = "both", axis = "x")
    #plt.legend(loc = "lower right")
    plt.subplots_adjust(bottom=0.15)



#  output_table
    a = np.array(avgs)
    b = np.array(stde)
    tabledata = np.vstack((a,b))
    tabledata = np.round(tabledata, decimals = 6)
    

    plt.subplot(2,1,2)
    plt.axis("off")
    tabulo = plt.table(cellText=tabledata, rowLabels=["Average", "Stdev(P)"], colLabels=names, loc='center', fontsize = 7)    
    tabulo.auto_set_font_size(False)
    tabulo.set_fontsize(6)
    
        
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_dir + "random_forest_result" + fileext + ".png")
    

    
else: 
    print("random_forest was skipped")
     



























# %%
if not svm_dir == "skip":
    print(datetime.now().strftime("%H:%M:%S>"), "Visualizing Support Vector Machine...")
    #filelist = []
    names = []
    accuracies = None

    for filepath in sorted(glob.iglob(svm_dir + "dataframes/svm_*.tsv")):
        filepath = filepath.replace('\\' , "/") # for some reason, it changes the last slash to backslash
        #print(filepath)
        #filelist.append(filepath)
        
        search = re.search("dataframes/svm_(.*?).tsv", filepath)
        if search:
            name = search.group(1) # to get only the matched charactesr
            names.append(name)

        #accuracies[name] = pd.read_csv(filepath, delimiter = "\t", header = 0, index_col = 0)
        df = pd.read_csv(filepath, delimiter = "\t", header = 0, index_col = 0)
        df = df.loc[:,["Accuracy"]] # double brackets to ensure return of a DF and not of a series
        df = df.rename(columns={"Accuracy": name})
        
        accuracies = pd.concat([accuracies, df], axis = 1)
    
    if accuracies is None:
        print("no datasets were found. Please recheck your input directory")
        import sys
        sys.exit()

        
    accuracies = accuracies.reindex(sorted(accuracies.columns, key = str), axis=1)

     ##### sort the accuracies. I'm sorry, but it has to b. It's much nicer, and I don't know a better way to sort than this.
    if args.unsorted == False:
        ordered = pd.DataFrame()
        newnames = []
        for name in custom_order:
            if name in names:
                ordered[name] = accuracies.loc[:,name]
                newnames.append(name)
            else:
                pass
            
        # make sure, that all names were present in the custom_order source variable
        if accuracies.shape[1] != len(newnames):
            for name in names:
                if name in custom_order:
                    pass
                else:
                    ordered[name] = accuracies.loc[:,name]
                    newnames.append(name)
                    print("please add {:s} to the custom order variable".format(name))
        names = newnames
        accuracies = ordered
    
    
    


# %% Plot
    avgs = []
    stde = []

    for i in range(accuracies.shape[1]):
        values = np.array(accuracies.iloc[:,i])
        avgs.append(np.mean(values))
        stde.append(np.std(values, ddof = 1))
                    
    plt.figure(figsize = [1*accuracies.shape[1] + 2, 6.4*1.8])
    
    plt.subplot(2,1,1)
    plt.bar(x = names, height = avgs, yerr = stde, alpha = 0.5, color = "red")
    plt.xticks(rotation = 45)
    plt.title("Support Vector Machine" + titleext)
    plt.ylabel("Accuracies")
    plt.grid(which = "both", axis = "x")
    #plt.legend(loc = "lower right")
    plt.subplots_adjust(bottom=0.15)



#  output_table
    a = np.array(avgs)
    b = np.array(stde)
    tabledata = np.vstack((a,b))
    tabledata = np.round(tabledata, decimals = 6)
    

    plt.subplot(2,1,2)
    plt.axis("off")
    tabulo = plt.table(cellText=tabledata, rowLabels=["Average", "Stdev(P)"], colLabels=names, loc='center', fontsize = 7)    
    tabulo.auto_set_font_size(False)
    tabulo.set_fontsize(6)
    
        
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_dir + "svm_result" + fileext + ".png")
    

    
# %%

else: 
    print("svm was skipped")
     


















print(datetime.now().strftime("%H:%M:%S>"), "Visualizer has finished successfully")






