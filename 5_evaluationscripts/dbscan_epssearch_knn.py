# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 15:15:18 2020

@author: Mike Toreno II
"""



# %%


import sys
import os
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import statistics

from sklearn.neighbors import NearestNeighbors



print(datetime.now().strftime("%H:%M:%S>"), "Starting dbscan_epssearch_knn.py")


try:
    os.chdir(os.path.dirname(sys.argv[0]))
except:
    pass




parser = argparse.ArgumentParser(description = "clustering data")  #required
parser.add_argument("-i","--input_dir", help="input directory", default = "../inputs/baseline_data/scaPCA_output/")
parser.add_argument("-p","--outputplot_dir", help="plot directory", default = "../outputs/hyperparameter/sca_DBScan/")
args = parser.parse_args() #required


input_path = args.input_dir
outputplot_dir = args.outputplot_dir



# %% Read Input data
print(datetime.now().strftime("%H:%M:%S>"), "loading data...")
data = np.loadtxt(open(input_path + "matrix.tsv"), delimiter="\t")

# load barcodes
barcodes = pd.read_csv(input_path + "barcodes.tsv", delimiter = "\t", header = None)
truelabels = barcodes.iloc[:,1]




# %%
print(datetime.now().strftime("%H:%M:%S>"), "Drawing Knee...")


neigh = NearestNeighbors(n_neighbors=4)
nbrs = neigh.fit(data)
distances, indices = nbrs.kneighbors(data)

distances = distances[:,1:]  # first column is always 0


# %%

distances = np.mean(distances, axis = 1)
distances = np.sort(distances, axis=0)


# %%


plt.figure()
plt.plot(distances)

plt.yticks(np.arange(0,80,5))

plt.grid(which = "both", axis = "y")


plt.savefig("epsplot.png")





