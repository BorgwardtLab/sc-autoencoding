# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 23:52:35 2020

@author: Mike Toreno II
"""

# %% Load Data
import scipy.io
import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import matplotlib.cm as cm # colourpalette
from sklearn.manifold import TSNE



### Get Matrix
print(datetime.now().strftime("%H:%M:%S>"), "reading input matrix...")
mtx_file = "./input/filtered_matrices_mex/hg19/matrix.mtx"
coomatrix = scipy.io.mmread(mtx_file)
data = np.transpose(coomatrix) # samples must be rows, variables = columns


### Get Labels
print(datetime.now().strftime("%H:%M:%S>"), "reading labels...")
lbl_file = "./input/filtered_matrices_mex/hg19/celltype_labels.tsv"
file = open(lbl_file, "r")
labels = file.read().split("\n")
file.close()
labels.remove("") #last, empty line is also removed


# load genes (for last task, finding most important genes)
file = open("./input/filtered_matrices_mex/hg19/genes.tsv", "r")
genes = file.read().split("\n")
file.close()
genes.remove("") 


# %%  for local execution (remove for full picture)


# print(datetime.now().strftime("%H:%M:%S>"), "deleting random data pieces...")
# genes_uplimit = 30000
# genes_downlimit = 25000
# cells_uplimit = 25000
# cells_downlimit = 10000

# # prev_element = "gulligulli"
# # for index in range(len(labels)):
# #     if labels[index] != prev_element:
# #         print(index)
# #     prev_element = labels[index]
# coomatrix = data #so that it is transposed, sorry for uglyness, this shouldn't be visible to outside. 
# labels = labels[cells_downlimit:cells_uplimit]

# reduced = coomatrix.tocsr()
# data = reduced[cells_downlimit:cells_uplimit, genes_downlimit:genes_uplimit]

# genes = genes[genes_downlimit:genes_uplimit]


# %%

print(datetime.now().strftime("%H:%M:%S>"), "scaling data...")
data = StandardScaler(with_mean= False).fit_transform(data) # Standardizing the features

# %%


print(datetime.now().strftime("%H:%M:%S>"), "Calculating tSNE...")
tsne = TSNE(n_components=2, verbose = 10)
tsnedata = tsne.fit_transform(data)


embed = tsne.embedding_
kldiver = tsne.kl_divergence_
niter = tsne.n_iter_




#%% Outputs

output_dir = "./scaTSNE_output"
component_name = "t-SNE"




# construct dataframe for 2d plot
df = pd.DataFrame(data = tsnedata[:,[0,1]], columns = [ component_name + ' 1', component_name + ' 2'])
df['celltlype'] = labels



if not os.path.exists(output_dir):
    print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Directory...")
    os.makedirs(output_dir)
    


### Create Plot
print(datetime.now().strftime("%H:%M:%S>"), "drawing plots...")
targets = set(labels) # what it will draw in plot, previously it was targets = ['b_cells' ... 'cytotoxic_t'], now its dynamic :*

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel(component_name + " 1", fontsize = 15)
ax.set_ylabel(component_name + " 2", fontsize = 15)
#ax.set_xlabel(component_name + ' 1 (' + str(round(explained_variance[0]*100, 3)) + "% of variance)", fontsize = 15)
#ax.set_ylabel(component_name + ' 2 (' + str(round(explained_variance[1]*100, 3)) + "% of variance)", fontsize = 15)
ax.set_title(component_name +' Plot (KL-divergence = ' + str(round(tsne.kl_divergence_, 2)) + ')', fontsize = 20)
colors = cm.rainbow(np.linspace(0, 1, len(targets)))
for target, color in zip(targets,colors):
    indicesToKeep = df['celltlype'] == target
    ax.scatter(df.loc[indicesToKeep, component_name + ' 1']
               , df.loc[indicesToKeep, component_name + ' 2']
               , c = color.reshape(1,-1)
               , s = 5)
ax.legend(targets)
ax.grid()
plt.savefig(output_dir + "/tSNE_Plot.png")




# %% Diagnostics





