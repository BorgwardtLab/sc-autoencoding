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



from sklearn.neighbors import NearestNeighbors



print(datetime.now().strftime("%H:%M:%S>"), "Starting dbscan_epssearch_knn.py")


try:
    os.chdir(os.path.dirname(sys.argv[0]))
except:
    pass




parser = argparse.ArgumentParser(description = "clustering data")  #required
parser.add_argument("-i","--input_dir", help="input directory", default = "../inputs/baseline_data/scaPCA_output/")
parser.add_argument("-t","--title", help="title that will be written into the output file", default = "title")
parser.add_argument("-p","--outputplot_dir", help="plot directory", default = "../outputs/optimization/dbscan_epsearch/")
args = parser.parse_args() #required






input_path = args.input_dir + "no_split/"
outputplot_dir = args.outputplot_dir


if not os.path.exists(outputplot_dir):
    print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Directory...")
    os.makedirs(outputplot_dir, exist_ok=True)




# %% Read Input data
print(datetime.now().strftime("%H:%M:%S>"), "loading data...")
data = np.loadtxt(open(input_path + "matrix.tsv"), delimiter="\t")

# load barcodes
barcodes = pd.read_csv(input_path + "barcodes.tsv", delimiter = "\t", header = None)
truelabels = barcodes.iloc[:,1]




# %%
print(datetime.now().strftime("%H:%M:%S>"), "Calculating Distances...")


neigh = NearestNeighbors(n_neighbors=10)
nbrs = neigh.fit(data)
distances, indices = nbrs.kneighbors(data)


distances = distances[:,1:]  # first column is always 0


# %%

avgdistances = np.mean(distances, axis = 1)
avgdistances = np.sort(avgdistances, axis=0)

first = distances[:,0]
first = np.sort(first, axis=0)

sec = distances[:,1]
sec = np.sort(sec, axis=0)

third = distances[:,2]
third = np.sort(third, axis=0)

fifth = distances[:,4]
fifth = np.sort(fifth, axis=0)

# %%

matplotlib_colours = {'b': 'blue', 'g': 'green', 'r': 'red', 'c': "cyan", 'm': "magenta", 'y': "yellow", 'k': "black", 'w': "white"}


print(datetime.now().strftime("%H:%M:%S>"), "Drawing Knee...")

plt.figure()

plt.plot(avgdistances, 'k')
plt.plot(first, 'y')
plt.plot(sec, 'b')
plt.plot(third, 'r')
plt.plot(fifth, 'g')

ticks = np.linspace(0, max(avgdistances)*1.1, num = 30) 
# for i in range(len(ticks)):
#    ticks[i] = np.format_float_positional(x = ticks[i], precision = 3, fractional = False, trim = "-") # round to 2 "significant" digits (not decimals) -> because we know nothing about the scale of the data, and still want the plot to look nice

plt.yticks(ticks)

plt.grid(which = "both", axis = "x")
plt.title("distances to the xth-nearest neighbour \n(individually ordered in ascending order)")
plt.xlabel("point")
plt.ylabel("distance to the xth-nearest neighbour")


plt.legend(["average (1-9)", 'first', 'second', 'third', 'fifth'])

plt.savefig(outputplot_dir + "epsplot_" + args.title + ".png")


print(datetime.now().strftime("%H:%M:%S>"), "dbscan_epssearch_knn finished successfully")

