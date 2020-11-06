# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 15:11:23 2020

@author: Mike Toreno II
"""

'''Please note, the clustering is based on all the received dimensions, however, plotted are only the first 2 (of course)'''



import argparse

parser = argparse.ArgumentParser(description = "clustering data")  #required
parser.add_argument("-k","--k", default = 10, help="the number of clusters to find", type = int)
parser.add_argument("-i","--input_dir", help="input directory", default = "../inputs/baseline_data/scaPCA_output/")
parser.add_argument("-o","--output_dir", help="output directory", default = "../outputs/kmcluster/")
#parser.add_argument("-p","--outputplot_dir", help="plot directory", default = "../outputs/kmcluster/")
parser.add_argument("--limit_dims", default = 0, help="number of input dimensions to consider", type = int)
parser.add_argument("-v","--verbosity", help="level of verbosity", default = 0, choices = [0, 1, 2, 3], type = int)
parser.add_argument("-e", "--elbow", help="helptext", action="store_true")
parser.add_argument("--elbowrange", help="the elobow will try all k's from 1-elbowrange", type = int, default = 11)
parser.add_argument("-t","--title", help="title that will be written into the output file", default = "title")
parser.add_argument("--num_reps", type = int, default = 100, help="how many repetitions it should do")
#parser.add_argument("-r", "--reset", help="if this is called, the previous results file will be overwritten, otherwise results are appended", action="store_true")
args = parser.parse_args() #required




# def sca_kmcluster(k = 5,
#                   dimensions = 0,
#                   input_dir = "../inputs/baselines/baseline_data/scaPCA_output/",
#                   output_dir = "../outputs/baselines/kmcluster/scaPCA_output/",
#                   verbosity = 0,
#                   elbow = False,
#                   elbowrange = 11,
#                   title = "title_placeholder",
#                   reset = False):

    



import sys
import os

from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import statistics
from sklearn.metrics.cluster import normalized_mutual_info_score

input_dir = args.input_dir + "no_split/"
output_dir = args.output_dir
outputplot_dir = output_dir + "plots/" #+ args.title + "/"
data_dir = output_dir + "dataframes/"



k = args.k
verbosity = args.verbosity
elbow = args.elbow
title = args.title
elbowrange = args.elbowrange







print(datetime.now().strftime("%d. %b %Y, %H:%M:%S>"), "Starting sca_kmcluster.py")
print(input_dir)


try:
    os.chdir(os.path.dirname(sys.argv[0]))
except:
    pass
         



# I think this is still used in the textoutput.
tech_start = input_dir.find("/sca")
tech_end = input_dir.find("_output/")


technique_name = input_dir[tech_start + 4 : tech_end]









# %% start of the looping



superpanda = pd.DataFrame()


for fold in range(1,args.num_reps+1):
    print(datetime.now().strftime("%H:%M:%S>"), "Starting Fold Nr {:d}...".format(fold))
    
    plt.close("all")
        
        
        
        
    
    # %% Read Input data
    print(datetime.now().strftime("%H:%M:%S>"), "loading data...")
    data = np.loadtxt(open(input_dir + "matrix.tsv"), delimiter="\t")
    
    
    # load barcodes
    barcodes = pd.read_csv(input_dir + "barcodes.tsv", delimiter = "\t", header = None)
    truelabels = barcodes.iloc[:,1]
    
    
    
    print(datetime.now().strftime("%H:%M:%S>"), "Clustering...")
    
    if args.limit_dims > 0:
        if args.limit_dims <= data.shape[1]:
            data = data[:,0:args.limit_dims]
            print("restricting input dimensions")
        else:
            print("cannot restrict dims. Limit dims = {:d}, input dimension = {:d}".format(args.limit_dims, data.shape[1]))
    
    
    
    
    
    
    
    # %% Clustering    
    
    km = KMeans(
        n_clusters=k, init="random", # 'k-means++',
        n_init=10, max_iter=300, 
        tol=1e-04, verbose = verbosity
    ) # default values
    
    
    
    predicted_cluster = km.fit_predict(data)
    
    
    
    
    # %% Plotting first simple plot
    
    
    
    if not os.path.exists(outputplot_dir):
        print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Plot Directory...")
        os.makedirs(outputplot_dir)
        
        
     
    
    
    print(datetime.now().strftime("%H:%M:%S>"), "Plotting Clusters...")
    
    import random
    
    import matplotlib.cm as cm 
    colors = cm.rainbow(np.linspace(0, 1, k))
    shapes = [".","o","v","^","<",">","8","s","p","P","*","h","H","X","D","d"]
    
    
    
    
    
    # %% Elbow
    
    if elbow:
        print(datetime.now().strftime("%H:%M:%S>"), "Calculating Elbow...")
        
        # calculate distortion for a range of number of cluster
        distortions = []
        for i in range(1, elbowrange):
            km = KMeans(
                n_clusters=k, init="random", #'k-means++',
                n_init=10, max_iter=300, 
                tol=1e-04, verbose = verbosity
            ) # default values
            km.fit(data)
            distortions.append(km.inertia_)
        
        # plot
        plt.figure()
        plt.plot(range(1, 11), distortions, marker='o')
        plt.xlabel('Number of clusters')
        plt.ylabel('Distortion')
        plt.title("k-search (on traindata)")
        plt.show()
        plt.savefig(outputplot_dir + title + "_Elbowplot.png")
        
    
    
    
    
    # %% Evaluate Purity
    from collections import Counter
    print(datetime.now().strftime("%H:%M:%S>"), "Evaluate Clustering...")
    
    clusterlabels = []
    clustersizes = []
    purity_per_cluster = []
    recall_per_cluster = []
    multiassigned = np.zeros(k, dtype=bool)
    global_counts = Counter(truelabels)
    
    
    
    COUNTS_PER_CLUSTER = "\n\nCounts per Cluster:"
    
    
    
    for cluster in range(k):
        indexes = np.where(predicted_cluster == cluster)[0] 
        
        # only used for creating dataframe
        clustersizes.append(len(indexes))
        #
        
        truelabels_in_cluster = truelabels[indexes]   
        counts = Counter(truelabels_in_cluster)
        
        most_common_str = ((counts.most_common(1))[0])[0]
        most_common_cnt = ((counts.most_common(1))[0])[1]
        
    
        ### remove this section if all runs well
        #print("\ncounts for cluster nr {0:d}:".format(cluster))
        #print(counts.most_common())
        
        COUNTS_PER_CLUSTER += "\ncounts for cluster nr {0:d}:\n".format(cluster)
        COUNTS_PER_CLUSTER += str(counts.most_common())
        #COUNTS_PER_CLUSTER += "\n"
    
        
        # find "multiple assigned celltypes"
        if most_common_str in clusterlabels:
            idx = clusterlabels.index(most_common_str)   
            multiassigned[cluster] = True
            multiassigned[idx] = True
    
    
        clusterlabels.append(most_common_str)              
            
     
        
        # calculate purity
        purity = most_common_cnt/len(truelabels_in_cluster)
        purity_per_cluster.append(purity)
        
        # calculate recall
        # the percentage of all cells of this type, that are in the cluster
        recall = most_common_cnt / global_counts[most_common_str]
        recall_per_cluster.append(recall)
    
    # add cluster number to multiassigneds, to mark them e.g. on the plot
    
    
    
    # create clusterlabels dictionary for the truefalseplot
    clusterlabels_dictionary = {}
    for i in range(len(clusterlabels)):
        clusterlabels_dictionary[i] = clusterlabels[i]
    
    
    clusterlabel_original = clusterlabels.copy()
    
    for idx in range(len(multiassigned)):
        if multiassigned[idx]:
            clusterlabels[idx] = clusterlabels[idx] + " (Cluster " + str(idx) + ")"
    
    
    predicted_labels_text = [clusterlabels_dictionary[i] for i in predicted_cluster]
    
    
    # %%
    
    
    nmi = normalized_mutual_info_score(labels_true = list(truelabels), labels_pred = predicted_labels_text)
    
    
    # %% Construct dataframe
    
    
    purity_per_cluster = np.array(purity_per_cluster)
    recall_per_cluster = np.array(recall_per_cluster)
    f1score = 2*purity_per_cluster*recall_per_cluster/(purity_per_cluster + recall_per_cluster)
    
    
    
    panda = pd.DataFrame(index = range(k))
    panda["Purity"] = purity_per_cluster
    panda["Size"] = clustersizes
    panda["Recall"] = recall_per_cluster
    panda["F1-score"] = f1score
    panda["NMI"] = nmi
    panda["Most common label"] = clusterlabel_original
    panda["Fold"] = fold
    
    # # Machine Output
    # os.makedirs(data_dir, exist_ok=True)
    # panda.to_csv(data_dir + "kmcluster_" + args.title + ".tsv", sep = "\t", index = True, header = True)

    superpanda = pd.concat([superpanda, panda])


    
        
    # %% Plotting
    # replot with labels
    
    plt.figure()
    for cluster in range(k):
        plt.scatter(
        x = data[predicted_cluster == cluster, 0], 
        y = data[predicted_cluster == cluster, 1],
        s=7, 
        c=colors[cluster,].reshape(1,-1),
        marker=random.choice(shapes), 
        edgecolor=[0, 0, 0, 0.3],
        label= clusterlabels[cluster],
        )
            
    plt.title(technique_name + " Clustering Prediction")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.show()
    plt.savefig(outputplot_dir + title + "_clusterplot_prediction_fold{:d}.png".format(fold))
        
    
    
    
    
    
    
    colors = cm.rainbow(np.linspace(0, 1, len(set(truelabels))))
    
    
    
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    # ax.set_xlabel(component_name + '_1 (' + str(round(explained_variance[0]*100, 3)) + "% of variance)", fontsize = 15)
    # ax.set_ylabel(component_name + '_2 (' + str(round(explained_variance[1]*100, 3)) + "% of variance)", fontsize = 15)
    ax.set_title('Real Labels', fontsize = 20)
    
    for target, color in zip(set(truelabels),colors):
        
        
        indicesToKeep = truelabels == target
        
        ax.scatter(data[indicesToKeep, 0]
                    , data[indicesToKeep, 1]
                    , c = color.reshape(1,-1)
                    , s = 5)
    ax.legend(set(truelabels))
    ax.grid()
    plt.savefig(outputplot_dir + title + "_truelabel_plot_fold{:d}.png".format(fold))
    
    
    
    
    
    
    # replot truefalse plot
    
    
    # has not worked always???
    #correct_indexes = np.array(predicted_labels_text) != np.array(truelabels).all()
    correct_indexes = np.zeros(len(predicted_labels_text), dtype = bool)
    for i in range(len(predicted_labels_text)):
        if predicted_labels_text[i] == truelabels[i]:
            correct_indexes[i] = True
    
    
    
    plt.figure()
    
    plt.scatter(
    x = data[correct_indexes, 0], 
    y = data[correct_indexes, 1],
    s=1, 
    c=np.array([1, 0, 0, 0]).reshape(1,-1),
    marker="o", 
    edgecolor='black',
    label= "correct ones",
    )
    
    plt.scatter(
    x = data[~correct_indexes, 0], 
    y = data[~correct_indexes, 1],
    s=1, 
    c=np.array([1, 0, 0, 0.5]).reshape(1,-1),
    marker="o", 
    edgecolor="face", # identical to face
    label= "incorrect ones",
    )
    
            
    plt.title(technique_name)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(scatterpoints=1)
    plt.grid()
    plt.show()
    plt.savefig(outputplot_dir + title + "_clusterplot_mistakes_fold{:d}.png".format(fold))
    
    
    
    # Barplot
    unique, counts = np.unique(predicted_cluster, return_counts=True)
    
    
    plt.figure()
    plt.bar(x = unique, height = counts)
    
    for i, y in enumerate(counts):
        plt.text(i, y+5, str(y), color='blue', fontweight='bold')
    
    
    plt.savefig(outputplot_dir + title + "_cluster_histogram_fold{:d}.png".format(fold))
    
    
    
    
    # %% Saving result
    print(datetime.now().strftime("%H:%M:%S>"), "Saving Results...")
    
    
    purity_per_cluster = np.round(purity_per_cluster, 4)       
    recall_per_cluster = np.round(recall_per_cluster, 4)  
    
    
    
    if not os.path.exists(output_dir):
        print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Directory...")
        os.makedirs(output_dir)
    
    
    
    
    filename = output_dir + "km_clustering_results.txt"  
    separator = ""
    if os.path.exists(filename):
        separator = "\n\n\n\n"
    
    
    file = open(filename, "a")
    
    file.write(separator)
    file.write("###############################\n")
    file.write(datetime.now().strftime("%d. %b %Y, %H:%M:%S - " + title ))
    #file.write("#######" + title + "#######\n")
    file.write("\n###############################\n")
    file.write("Fold number = {:d}".format(fold))
    file.write(", input_data from " + input_dir + "\n")
    file.write("\nAverage Purity: \t" + '{:.4f}'.format(statistics.mean(purity_per_cluster)))
    file.write("\t(" + str(purity_per_cluster).strip("[]") + ")")
    
    file.write("\nAverage Recall: \t" + '{:.4f}'.format(statistics.mean(recall_per_cluster)))
    file.write("\t(" + str(recall_per_cluster).strip("[]") + ")")
    
    file.write("\nCluster labels: \t" + str(clusterlabels).strip("[]") + ")")
    
    file.write(COUNTS_PER_CLUSTER)
    
    file.close() 
    
    
    # with open(output_dir + "counts_per_cluster.tsv", "w") as outfile:
    #     outfile.write(COUNTS_PER_CLUSTER)
    
    
    beenzcount = 0
    for i in range(len(truelabels)):
        if truelabels[i] == predicted_labels_text[i]:
            beenzcount = beenzcount + 1
            
    global_purity = beenzcount/len(truelabels)            
    
    
    print(datetime.now().strftime("%H:%M:%S>"), "sca_kmcluster.py terminated successfully")
    print("global purity is: {0:.4f}".format(global_purity))   
    
    
   
# Machine Output
os.makedirs(data_dir, exist_ok=True)
superpanda.to_csv(data_dir + "kmcluster_" + args.title + ".tsv", sep = "\t", index = True, header = True)
   

   

   
    
   
    
   
    
   
    
   
    
   
    
   
    
        
    # %%
      
    # if __name__ == "__main__":
    #     sca_kmcluster(k = args.k, dimensions = args.dimensions, input_dir = args.input_dir, 
    #                   output_dir = args.output_dir, 
    #                   verbosity = args.verbosity, elbow = args.elbow, elbowrange = args.elbowrange, 
    #                   title = args.title, reset = args.reset)
    




