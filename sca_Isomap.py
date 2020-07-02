# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 23:52:35 2020

@author: Mike Toreno II
"""


# %% Load Data
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import matplotlib.cm as cm # colourpalette
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler


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




# %%


print(datetime.now().strftime("%H:%M:%S>"), "scaling data...")
data = StandardScaler(with_mean= False).fit_transform(data) # Standardizing the features



print(datetime.now().strftime("%H:%M:%S>"), "calculating Isomap...")
embedding = Isomap(n_components = 2)
reduced = embedding.fit_transform(data)


# %%


embeddingdata = embedding.embedding_
kernel_pca = embedding.kernel_pca_
nbrs = embedding.nbrs_
dist_matrix = embedding.dist_matrix_




#%% Outputs

output_dir = "./scaIsomap_output"
component_name = "Isomap"


# construct dataframe for 2d plot
df = pd.DataFrame(data = reduced[:,[0,1]], columns = [ component_name + ' 1', component_name + ' 2'])
df['celltlype'] = labels




if not os.path.exists(output_dir):
    print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Directory...")
    os.makedirs(output_dir)
    


### Create Plot
print(datetime.now().strftime("%H:%M:%S>"), "drawing plots...")
targets = set(labels) # what it will draw in plot, previously it was targets = ['b_cells' ... 'cytotoxic_t'], now its dynamic :*

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
# ax.set_xlabel(component_name + ' 1 (' + str(round(explained_variance[0]*100, 3)) + "% of variance)", fontsize = 15)
# ax.set_ylabel(component_name + ' 2 (' + str(round(explained_variance[1]*100, 3)) + "% of variance)", fontsize = 15)
ax.set_xlabel(component_name + " 1 (??% of variance)", fontsize = 15)
ax.set_ylabel(component_name + " 2 (??% of variance)", fontsize = 15)
ax.set_title('Most Powerful '+ component_name +'s', fontsize = 20)
colors = cm.rainbow(np.linspace(0, 1, len(targets)))
for target, color in zip(targets,colors):
    indicesToKeep = df['celltlype'] == target
    ax.scatter(df.loc[indicesToKeep, component_name + ' 1']
               , df.loc[indicesToKeep, component_name + ' 2']
               , c = color.reshape(1,-1)
               , s = 5)
ax.legend(targets)
ax.grid()
plt.savefig(output_dir + "/Isomap_plot.png")




# %% Diagnostics





