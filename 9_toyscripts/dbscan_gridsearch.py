# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 03:20:37 2020

@author: Mike Toreno II
"""




# %%


import sys
import os
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN





print(datetime.now().strftime("%H:%M:%S>"), "Starting gridsearch.py")


try:
    os.chdir(os.path.dirname(sys.argv[0]))
except:
    pass
         


parser = argparse.ArgumentParser(description = "clustering data")  #required
#parser.add_argument("-k","--k", default = 6, help="the number of clusters to find", type = int)
parser.add_argument("-d","--dimensions", help="enter a value here to restrict the number of input dimensions to consider", type = int, default = 0)
parser.add_argument("-i","--input_dir", help="input directory", default = "../inputs/baseline_data/scaPCA_output/")
parser.add_argument("-o","--output_dir", help="output directory", default = "../outputs/kmcluster/")
parser.add_argument("-p","--outputplot_dir", help="plot directory", default = "../outputs/kmcluster/")
parser.add_argument("-v","--verbosity", help="level of verbosity", default = 0, choices = [0, 1, 2, 3], type = int)
parser.add_argument("-t","--title", help="title that will be written into the output file", default = "title placeholder")
parser.add_argument("-r", "--reset", help="if this is called, the previous results file will be overwritten, otherwise results are appended", action="store_true")

parser.add_argument("-e","--eps", help="The maximum distance between two samples for one to be considered as in the neighborhood of the other.", type = int, default = 30)
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




data = data[:,range(dims)]
dbscanner = DBSCAN(eps=args.eps, min_samples=args.min_samples)

#predicted_labels = dbscanner.fit_predict(data)




# %%

import numpy as np
from sklearn.model_selection import GridSearchCV


#potential_eps = np.linspace(start = 0.1, stop = 50, num = 30)
#potential_min_samples = np.append(arr = np.linspace(start = 1, stop = 20, num = 10, dtype = int), values = np.linspace(start = 21, stop = 100, num = 10, dtype = int))

potential_eps = np.linspace(start = 0.1, stop = 50, num = 5)
potential_min_samples = np.append(arr = np.linspace(start = 1, stop = 20, num = 2, dtype = int), values = np.linspace(start = 21, stop = 100, num = 2, dtype = int))


parameters = {'eps': potential_eps, 'min_samples': potential_min_samples}

# %%



gridsearcher = GridSearchCV(estimator = dbscanner, param_grid = parameters)

gridsearcher.fit(data)



# %%

print(gridsearcher)

print(gridsearcher.best_score_)

print(gridsearcher.best_estimator_.alpha)
















