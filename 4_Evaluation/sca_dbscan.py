# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 16:25:00 2020

@author: Mike Toreno II
"""





import sys
import os
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import statistics





print(datetime.now().strftime("%H:%M:%S>"), "Starting sca_dbscan.py")


try:
    os.chdir(os.path.dirname(sys.argv[0]))
except:
    pass
         


parser = argparse.ArgumentParser(description = "clustering data")  #required
#parser.add_argument("-k","--k", default = 6, help="the number of clusters to find", type = int)
parser.add_argument("-d","--dimensions", help="enter a value here to restrict the number of input dimensions to consider", type = int, default = 0)
parser.add_argument("-i","--input_dir", help="input directory", default = "../inputs/baselines/baseline_data/scaPCA_output/")
parser.add_argument("-o","--output_dir", help="output directory", default = "../inputs/baselines/dbscan/")
parser.add_argument("-p","--outputplot_dir", help="plot directory", default = "../outputs/baselines/dbscan/")
parser.add_argument("-v","--verbosity", help="level of verbosity", default = 0, choices = [0, 1, 2, 3], type = int)
parser.add_argument("-t","--title", help="title that will be written into the output file", default = "title placeholder")
parser.add_argument("-r", "--reset", help="if this is called, the previous results file will be overwritten, otherwise results are appended", action="store_true")

parser.add_argument("-e","--eps", help="The maximum distance between two samples for one to be considered as in the neighborhood of the other.", type = float, default = 30)
parser.add_argument("-m","--min_samples", help="The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.", type = int, default = 5)
args = parser.parse_args() #required


input_path = args.input_dir
output_dir = args.output_dir
outputplot_dir = args.outputplot_dir


tech_start = input_path.find("/sca")
tech_end = input_path.find("_output/")
technique_name = input_path[tech_start + 4 : tech_end]




# %% Read Input data
print(datetime.now().strftime("%H:%M:%S>"), "loading data...")
data = np.loadtxt(open(input_path + "matrix.tsv"), delimiter="\t")

# load barcodes
barcodes = pd.read_csv(input_path + "barcodes.tsv", delimiter = "\t", header = None)
truelabels = barcodes.iloc[:,1]




if args.dimensions == 0:
    dims = data.shape[1]
    #print("dims was set to {0:d}".format(dims))
else:
    dims = args.dimensions
    #print("dims was set to {0:d}".format(dims))
    


# %% Clustering

print(datetime.now().strftime("%H:%M:%S>"), "Clustering...")



args.eps = 15
args.min_samples = 2


data = data[:,range(dims)]
dbscanner = DBSCAN(eps=args.eps, min_samples=args.min_samples)



predicted_labels = dbscanner.fit_predict(data)

n_clusters = len(set(predicted_labels)) - (1 if -1 in predicted_labels else 0)
n_labels = len(set(predicted_labels))   # will always include -1









# %% Evaluate Purity
from collections import Counter
print(datetime.now().strftime("%H:%M:%S>"), "Evaluate Clustering...")


clusterlabels = []
purity_per_cluster = []
recall_per_cluster = []
  
multiassigned = np.zeros(n_labels, dtype=bool)

global_counts = Counter(truelabels)



for cluster in (range(n_clusters)):

    indexes = np.where(predicted_labels == cluster)[0] 
    truelabels_in_cluster = truelabels[indexes]   
    counts = Counter(truelabels_in_cluster)
    most_common_str = ((counts.most_common(1))[0])[0]
    most_common_cnt = ((counts.most_common(1))[0])[1]
    
    
    # find "multiple assigned celltypes"
    if most_common_str in clusterlabels:
        idx = clusterlabels.index(most_common_str)
        multiassigned[cluster] = True
        multiassigned[idx] = True
        
        
    clusterlabels.append(most_common_str)         
    # clusterlabels_noedit    
    
    
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




# add cluster to each name that is doubly
for idx in range(len(multiassigned)):
    if multiassigned[idx]:
        clusterlabels[idx] = clusterlabels[idx] + " (Cluster " + str(idx) + ")"



purity_per_cluster = np.round(purity_per_cluster, 4)       
recall_per_cluster = np.round(recall_per_cluster, 4)  






# %% Plotting

if not os.path.exists(outputplot_dir):
    print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Plot Directory...")
    os.makedirs(outputplot_dir)
    
    
print(datetime.now().strftime("%H:%M:%S>"), "Plotting Clusters...")

import random

import matplotlib.cm as cm 
colors = cm.rainbow(np.linspace(0, 1, n_labels))
shapes = [".","o","v","^","<",">","8","s","p","P","*","h","H","X","D","d"]






    
# %%
# replot with labels

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
plt.savefig(outputplot_dir + "truelabels.png")







# %% plot

colors = cm.rainbow(np.linspace(0, 1, n_labels))


plt.figure()
for cluster in range(n_clusters):
    plt.scatter(
    x = data[predicted_labels == cluster, 0], 
    y = data[predicted_labels == cluster, 1],
    s=5, 
    c=colors[cluster,].reshape(1,-1),
    marker=random.choice(shapes), 
    edgecolor=colors[cluster,].reshape(1,-1),
    label= clusterlabels[cluster],
    )
        
plt.title(technique_name + " Clustering Prediction")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend(scatterpoints=1)
plt.grid()
plt.show()
plt.savefig(outputplot_dir + "clusterplot_prediction.png")
    


# %% plot errors

if -1 in set(predicted_labels):
    clusterlabels_dictionary[-1] = "Not assigned"


predicted_labels_text = [clusterlabels_dictionary[i] for i in predicted_labels]


correct_indexes = np.zeros(len(predicted_labels_text), dtype = bool)
for i in range(len(predicted_labels_text)):
    if predicted_labels_text[i] == truelabels[i]:
        correct_indexes[i] = True




correct_indexes = np.array(predicted_labels_text) == np.array(truelabels).all()




truedata = data[correct_indexes, 0]



plt.figure()

plt.scatter(
x = data[correct_indexes, 0], 
y = data[correct_indexes, 1],
s=10, 
c=np.array([1, 0, 0, 0]).reshape(1,-1),
marker="o", 
edgecolor='black',
label= "correct ones",
)

plt.scatter(
x = data[~correct_indexes, 0], 
y = data[~correct_indexes, 1],
s=5, 
c=np.array([1, 0, 0, 0.5]).reshape(1,-1),
marker="o", 
edgecolor=np.array([1, 0, 0, 0.5]).reshape(1,-1),
label= "incorrect ones",
)

        
plt.title(technique_name)
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend(scatterpoints=1)
plt.grid()
plt.show()
plt.savefig(outputplot_dir + "clusterplot_mistakes.png")
    








# %% Saving result
print(datetime.now().strftime("%H:%M:%S>"), "Saving Results...")


if not os.path.exists(output_dir):
    print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Directory...")
    os.makedirs(output_dir)


if args.reset:
    file = open(output_dir + "dbscan_clustering_results.txt", "w")
else:
    file = open(output_dir + "dbscan_clustering_results.txt", "a")
    file.write("\n")
    file.write("\n")
    file.write("\n")
    file.write("\n")
    

file.write("######" + args.title + "######\n")
file.write("input_data from " + input_path + "\n")
file.write("\nAverage Purity: \t" + '{:.4f}'.format(statistics.mean(purity_per_cluster)))
file.write("\t(" + str(purity_per_cluster).strip("[]") + ")")

file.write("\nAverage Recall: \t" + '{:.4f}'.format(statistics.mean(recall_per_cluster)))
file.write("\t(" + str(recall_per_cluster).strip("[]") + ")")

file.write("\nCluster labels: \t" + str(clusterlabels).strip("[]") + ")")

file.close()

print("average purity: {0:.6f}".format(statistics.mean(purity_per_cluster)))
print(statistics.mean(purity_per_cluster))
print("number of clusters found: {0:02d}".format(n_clusters))

# %%

print(datetime.now().strftime("%H:%M:%S>"), "sca_kmcluster.py terminated successfully\n")










