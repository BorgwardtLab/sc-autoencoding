# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 15:11:23 2020

@author: Mike Toreno II
"""


import sys
import os
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans





print(datetime.now().strftime("%H:%M:%S>"), "Starting sca_kmcluster.py")


try:
    os.chdir(os.path.dirname(sys.argv[0]))
except:
    pass
         


parser = argparse.ArgumentParser(description = "clustering data")  #required
parser.add_argument("-k","--k", help="the number of clusters to find", type = int, default = 3)
parser.add_argument("-d","--dimensions", help="enter a value here to restrict the number of input dimensions to consider", type = int, default = 0)
parser.add_argument("-i","--input_dir", help="input directory", default = "../inputs/baseline_data/scaPCA_output/")
parser.add_argument("-o","--output_dir", help="output directory", default = "../outputs/kmcluster/")
parser.add_argument("-p","--outputplot_dir", help="plot directory", default = "../outputs/kmcluster_plots/")
args = parser.parse_args() #required



input_path = args.input_dir
output_dir = args.output_dir
outputplot_dir = args.outputplot_dir
k = args.k







# %% Read Input data
print(datetime.now().strftime("%H:%M:%S>"), "loading data...")
data = np.loadtxt(open(input_path + "coordinates.tsv"), delimiter="\t")


# load barcodes
barcodes = pd.read_csv(input_path + "barcodes.tsv", delimiter = "\t", header = None)


# %% Clustering

if args.dimensions == 0:
    dims = data.shape[1]
    #print("dims was set to {0:d}".format(dims))
else:
    dims = args.dimensions
    #print("dims was set to {0:d}".format(dims))
    

data = data[:,range(dims)]


km = KMeans(
    n_clusters=k, init='k-means++',
    n_init=10, max_iter=300, 
    tol=1e-04, verbose = 3
) # default values


cluster_labels = km.fit_predict(data)





# %% Plotting
if not os.path.exists(outputplot_dir):
    print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Plot Directory...")
    os.makedirs(outputplot_dir)
    



print(datetime.now().strftime("%H:%M:%S>"), "Plotting Results...")

import random

import matplotlib.cm as cm 
colors = cm.rainbow(np.linspace(0, 1, k))
shapes = [".","o","v","^","<",">","8","s","p","P","*","h","H","X","D","d"]


plt.figure()

for i in range(k):
    plt.scatter(
    data[cluster_labels == i, 0], data[cluster_labels == i, 1],
    s=50, 
    c=colors[i,],
    marker=random.choice(shapes), 
    edgecolor='black',
    label='cluster {0:d}'.format(i)
    )
    stuff = colors[i,].reshape(1,-1)

plt.legend(scatterpoints=1)
plt.grid()
plt.show()
plt.savefig(outputplot_dir + "clusterplot.png")

# %% Elbow


# calculate distortion for a range of number of cluster
distortions = []
for i in range(1, 11):
    km = KMeans(
        n_clusters=k, init='k-means++',
        n_init=10, max_iter=300, 
        tol=1e-04, verbose = 3
    ) # default values
    km.fit(data)
    distortions.append(km.inertia_)

# plot
plt.figure()
plt.plot(range(1, 11), distortions, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()
plt.savefig(outputplot_dir + "Elbowplot.png")

















