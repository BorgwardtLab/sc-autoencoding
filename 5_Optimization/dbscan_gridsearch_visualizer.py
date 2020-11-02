# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 15:11:23 2020

@author: Mike Toreno II
"""

'''Please note, the clustering is based on all the received dimensions, however, plotted are only the first 2 (of course)'''



import argparse

parser = argparse.ArgumentParser(description = "clustering data")  #required
parser.add_argument("-i","--input_dir", help="input directory", default = "M:/Projects/simon_streib_internship/sc-autoencoding/outputs/optimization/technique_evaluation/dbscan_gridsearch_bash/")
parser.add_argument("-o","--output_dir", help="output directory", default = "M:/Projects/simon_streib_internship/sc-autoencoding/outputs/optimization/technique_evaluation/dbscan_gridsearch_bash/")
#parser.add_argument("-p","--outputplot_dir", help="plot directory", default = "../outputs/kmcluster/")
args = parser.parse_args() #required





import sys
import os

from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import re



input_dir = args.input_dir + "dataframes/"
output_dir = args.output_dir




print(datetime.now().strftime("%d. %b %Y, %H:%M:%S>"), "Starting dbscan_gridsearch_visualizer.py")
print(input_dir)


try:
    os.chdir(os.path.dirname(sys.argv[0]))
except:
    pass
         


# %% readin dataframes


names = []
dataframes = []

for filepath in sorted(glob.iglob(input_dir + "dbscan_*.tsv")):
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
        





# %% now lets find out what kinda dataframes we have here: 
minpts = []
eps = []



for df in dataframes:
    eps.append(df.loc[:,"eps"][0])
    minpts.append(df.loc[:,"minpts"][0])



# rows = minpts
# columns = eps


minpts = np.unique(minpts, return_counts = False)
eps = np.unique(eps, return_counts=False)

nmis = pd.DataFrame(index = eps, columns = minpts, dtype = float)
outlier_fractions = pd.DataFrame(index = eps, columns = minpts, dtype = float)
f1scores = pd.DataFrame(index = eps, columns = minpts, dtype = float)
numclust50 = pd.DataFrame(index = eps, columns = minpts, dtype = int)


for df in dataframes:
    curr_ep = df.loc[:,"eps"][0]
    curr_minpts = df.loc[:,"minpts"][0]

    nmi_score = df.loc[:,"NMI"].iloc[0]
    
    # nmis
    nmis.loc[curr_ep, curr_minpts] = nmi_score

    # outlier fraction
    if df.iloc[-1,:].loc["Most common label"] == "Outliers":
        outlier_size = df.loc["outliers","Size"]
        sum_sizes = sum(np.array(df.loc[:,"Size"]))
        outlier_fraction = outlier_size / sum_sizes
        df2 = df.copy()
        df2 = df2.iloc[0:-1,:]
    else:
        outlier_fraction = 0
        df2 = df.copy()
    outlier_fractions.loc[curr_ep, curr_minpts] = outlier_fraction

    # f1 scores
    weighted_F1 = 0
    sizes = np.array(df2.loc[:,"Size"])
    fscores = np.array(df2.loc[:,"F1-score"])
    for j in range(len(sizes)):
        weighted_F1 += fscores[j] * sizes[j]
    weighted_F1 = weighted_F1/sum(sizes)
    f1scores.loc[curr_ep, curr_minpts] = weighted_F1

    # numclust50
    sizes = np.array(df2.loc[:,"Size"])
    numclusts = sum(sizes > 50)
    numclust50.loc[curr_ep, curr_minpts] = numclusts



# %%
fig, ax = plt.subplots(1,1)

plt.title("NMI score per combination\n(axis NOT evenly spaced)")
img = ax.imshow(nmis, interpolation = "bilinear", origin = "lower", cmap = "RdYlGn")
fig.colorbar(img)

ax.set_xlabel("minpts")
ax.set_ylabel("eps")
ax.set_title("NMI Score")

plt.xticks(np.arange(len(eps)), eps)   
plt.yticks(np.arange(len(minpts)), minpts)

plt.savefig(output_dir + "NMI_scores.png")









fig, ax = plt.subplots(1,1)

plt.title("Outlier Fractions per combination\n(axis NOT evenly spaced)")
img = ax.imshow(f1scores, interpolation = "bilinear", origin = "lower", cmap = "RdYlGn")
fig.colorbar(img)

ax.set_xlabel("minpts")
ax.set_ylabel("eps")
ax.set_title("weighted F1 score")

plt.xticks(np.arange(len(eps)), eps)   
plt.yticks(np.arange(len(minpts)), minpts)

plt.savefig(output_dir + "F1_scores.png")








fig, ax = plt.subplots(1,1)

plt.title("Outlier Fractions per combination\n(axis NOT evenly spaced)")
img = ax.imshow(outlier_fractions, interpolation = "bilinear", origin = "lower", cmap = "jet")
fig.colorbar(img)

ax.set_xlabel("minpts")
ax.set_ylabel("eps")
ax.set_title("Outlier Fraction")

plt.xticks(np.arange(len(eps)), eps)   
plt.yticks(np.arange(len(minpts)), minpts)

plt.savefig(output_dir + "Outlierfractions.png")







fig, ax = plt.subplots(1,1)

plt.title("Outlier Fractions per combination\n(axis NOT evenly spaced)")
img = ax.imshow(numclust50, interpolation = "bilinear", origin = "lower", cmap = "ocean")
fig.colorbar(img)

ax.set_xlabel("minpts")
ax.set_ylabel("eps")
ax.set_title("Number of clusters found with at least 50+ cells")

plt.xticks(np.arange(len(eps)), eps)   
plt.yticks(np.arange(len(minpts)), minpts)

plt.savefig(output_dir + "num_clusts50plus.png")











print(datetime.now().strftime("%H:%M:%S>"), "Gridsearch terminated successfully")






























