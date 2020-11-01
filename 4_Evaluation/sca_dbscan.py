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
from sklearn.metrics.cluster import normalized_mutual_info_score




print(datetime.now().strftime("\n\n%d. %b %Y, %H:%M:%S>"), "Starting sca_dbscan.py")


try:
    os.chdir(os.path.dirname(sys.argv[0]))
except:
    pass
         


parser = argparse.ArgumentParser(description = "clustering data")  #required
#parser.add_argument("-k","--k", default = 6, help="the number of clusters to find", type = int)
parser.add_argument("-i","--input_dir", help="input directory", default = "../inputs/baseline_data/scaPCA_output/")
parser.add_argument("-o","--output_dir", help="output directory", default = "../outputs/dbscan/")
parser.add_argument("--limit_dims", default = 0, help="number of input dimensions to consider", type = int)
#parser.add_argument("-p","--outputplot_dir", help="plot directory", default = "../outputs/dbscan/scaPCA_output/")
parser.add_argument("-v","--verbosity", help="level of verbosity", default = 0, choices = [0, 1, 2, 3], type = int)
parser.add_argument("-t","--title", help="title that will be written into the output file", default = "title")
#parser.add_argument("-r", "--reset", help="if this is called, the previous results file will be overwritten, otherwise results are appended", action="store_true")

parser.add_argument("-e","--eps", default = 7, help="The maximum distance between two samples for one to be considered as in the neighborhood of the other.", type = float)
parser.add_argument("-m","--min_samples", default = 5, help="The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.", type = int)
args = parser.parse_args() #required



input_dir = args.input_dir + "no_split/"
output_dir = args.output_dir
outputplot_dir = output_dir + "plots/" #+ args.title + "/"
data_dir = output_dir + "dataframes/"
title = args.title

print(input_dir)

tech_start = input_dir.find("/sca")
tech_end = input_dir.find("_output/")
technique_name = input_dir[tech_start + 4 : tech_end]


# %% Read Input data


print(datetime.now().strftime("%H:%M:%S>"), "loading data...")
data = np.loadtxt(open(input_dir + "matrix.tsv"), delimiter="\t")

# load barcodes
barcodes = pd.read_csv(input_dir + "barcodes.tsv", delimiter = "\t", header = None)
truelabels = barcodes.iloc[:,1]



if args.limit_dims > 0:
    if args.limit_dims <= data.shape[1]:
        data = data[:,0:args.limit_dims]
        print("restricting input dimensions")
    else:
        print("cannot restrict dims. Limit dims = {:d}, input dimension = {:d}".format(args.limit_dims, data.shape[1]))





# %% Clustering


print(datetime.now().strftime("%H:%M:%S>"), "Clustering...")


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
clustersizes = []
  
multiassigned = np.zeros(n_labels, dtype=bool)
global_counts = Counter(truelabels)


for cluster in (range(n_clusters)):

    # only look at data with this prediction
    indexes = np.where(predicted_labels == cluster)[0] 
    truelabels_in_cluster = truelabels[indexes]   
    
    # find the most common in cluster
    counts = Counter(truelabels_in_cluster)
    most_common_str = ((counts.most_common(1))[0])[0]
    most_common_cnt = ((counts.most_common(1))[0])[1]
    clustersizes.append(len(indexes))

    
    
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


clusterlabel_original = clusterlabels.copy()

# add cluster to each name that is doubly
for idx in range(len(multiassigned)):
    if multiassigned[idx]:
        clusterlabels[idx] = clusterlabels[idx] + " (# " + str(idx) + ")"





# %% create dataframe for machine export
# plot errors
if -1 in set(predicted_labels):
    clusterlabels_dictionary[-1] = "Outlier"
predicted_labels_text = [clusterlabels_dictionary[i] for i in predicted_labels]


nmi = normalized_mutual_info_score(labels_true = list(truelabels), labels_pred = predicted_labels_text)



purity_per_cluster = np.array(purity_per_cluster)
recall_per_cluster = np.array(recall_per_cluster)

f1score = 2*purity_per_cluster*recall_per_cluster/(purity_per_cluster + recall_per_cluster)






panda = pd.DataFrame(index = range(n_clusters))
panda["Purity"] = purity_per_cluster
panda["Size"] = clustersizes
panda["Recall"] = recall_per_cluster
panda["F1-score"] = f1score
panda["NMI"] = nmi
panda["Most common label"] = clusterlabel_original
panda["eps"] = args.eps
panda["minpts"] = args.min_samples





thing = panda.index


if n_clusters < n_labels:
    # appending rows: more difficult
    unique, counts = np.unique(predicted_labels, return_counts=True)    
    
    dictionary = {"Size": counts[0], "Most common label": "Outliers"}
    
    newrow = pd.Series(dictionary).rename("outliers")
    
    ### decide if you want outliers to stand in the index 
    #panda = panda.append(newrow, ignore_index= True)
    panda = panda.append(newrow, ignore_index= False)
    
    



# Machine Output
os.makedirs(data_dir, exist_ok=True)
panda.to_csv(data_dir + "dbscan_" + args.title + ".tsv", sep = "\t", index = True, header = True)




# %% Plotting

if not os.path.exists(outputplot_dir):
    print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Plot Directory...")
    os.makedirs(outputplot_dir)
    
    
print(datetime.now().strftime("%H:%M:%S>"), "Plotting Clusters...")

import random

import matplotlib.cm as cm 
colors = cm.rainbow(np.linspace(0, 1, n_labels))
shapes = [".","o","v","^","<",">","8","s","p","P","*","h","H","X","D","d"]







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







# plot
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
    




correct_indexes = np.zeros(len(predicted_labels_text), dtype = bool)
for i in range(len(predicted_labels_text)):
    if predicted_labels_text[i] == truelabels[i]:
        correct_indexes[i] = True




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
    





# Histogram
unique, counts = np.unique(predicted_labels, return_counts=True)

bp_labels = clusterlabels.copy()
if n_clusters != n_labels:
    bp_labels.insert(0, "Outliers")

plt.figure()
plt.bar(x = bp_labels, height = counts)

# add values
for i, y in enumerate(counts):
    plt.text(i, y+5, str(y), color='blue', fontweight='bold')

plt.xticks(bp_labels, rotation=90)
plt.subplots_adjust(bottom=0.55, top=0.9)

plt.show()
plt.savefig(outputplot_dir + "cluster_histogram.png")












# %% Saving result


purity_per_cluster = np.round(purity_per_cluster, 4)       
recall_per_cluster = np.round(recall_per_cluster, 4)  



print(datetime.now().strftime("%H:%M:%S>"), "Saving Results...")


if not os.path.exists(output_dir):
    print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Directory...")
    os.makedirs(output_dir)


filename = output_dir + "dbscan_clustering_results.txt"
separator = ""

if os.path.exists(filename):
    separator = "\n\n\n\n\n"


file = open(output_dir + "dbscan_clustering_results.txt", "a")

file.write(separator)
file.write("######### " + args.title + " #########\n")
file.write(datetime.now().strftime("%d. %b %Y, %H:%M:%S\n"))
file.write("input_data from " + input_dir + "\n")
file.write("\nAverage Purity: \t" + '{:.4f}'.format(statistics.mean(purity_per_cluster)))
file.write("\t(" + str(purity_per_cluster).strip("[]") + ")")

file.write("\nAverage Recall: \t" + '{:.4f}'.format(statistics.mean(recall_per_cluster)))
file.write("\t(" + str(recall_per_cluster).strip("[]") + ")")

file.write("\nCluster labels: \t" + str(clusterlabels).strip("[]") + ")")

file.close()

print("\taverage purity: {0:.6f}".format(statistics.mean(purity_per_cluster)))
print("\tnumber of clusters found: {0:02d}".format(n_clusters))




if n_clusters != n_labels:
    outlier_fraction = counts[0]/len(predicted_labels)
    print("\tOutlier Fraction is {:.4f}".format(outlier_fraction))
else:
    print("\tOutlier Fraction is 0.0000")
    
        
    

# %%

print(datetime.now().strftime("%H:%M:%S>"), "sca_dbscan.py terminated successfully\n")











