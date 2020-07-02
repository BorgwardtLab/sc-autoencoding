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
from sklearn.manifold import Isomap
import umap




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
labels.remove("") 


# load genes (for last task, finding most important genes)
file = open("./input/filtered_matrices_mex/hg19/genes.tsv", "r")
genes = file.read().split("\n")
file.close()
genes.remove("") 


# %%  for local execution (remove for full picture)
print(datetime.now().strftime("%H:%M:%S>"), "deleting random data pieces...")
genes_uplimit = 30000
genes_downlimit = 25000
cells_uplimit = 25000
cells_downlimit = 10000

# prev_element = "gulligulli"
# for index in range(len(labels)):
#     if labels[index] != prev_element:
#         print(index)
#     prev_element = labels[index]
labels = labels[cells_downlimit:cells_uplimit]

csrdata = data.tocsr()
data = csrdata[cells_downlimit:cells_uplimit, genes_downlimit:genes_uplimit]
data = data.tocoo()

genes = genes[genes_downlimit:genes_uplimit]



# # %%
# import seaborn as sns

# print(data.shape)



# component_name = "Isomap"
# df = pd.DataFrame(data = data[:,[0,1]], columns = ["pc1", "pc2"])
# df['celltype'] = labels

# sns.pairplot(df, hue = "celltype")




# %%






# %%


print(datetime.now().strftime("%H:%M:%S>"), "scaling data...")
data = StandardScaler(with_mean=False).fit_transform(data) # Standardizing the features


print(datetime.now().strftime("%H:%M:%S>"), "calculating UMAP...")
reducer = umap.UMAP(verbose = 1)
newdata = reducer.fit_transform(data)





# %%



#%% Outputs

output_dir = "./scaUMAP_output"
component_name = "UMAP"

# construct dataframe for 2d plot
df = pd.DataFrame(data = newdata[:,[0,1]], columns = [ component_name + ' 1', component_name + ' 2'])
df['celltype'] = labels



if not os.path.exists(output_dir):
    print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Directory...")
    os.makedirs(output_dir)
    








### Create Plot
print(datetime.now().strftime("%H:%M:%S>"), "drawing plots...")
targets = set(labels) # what it will draw in plot, previously it was targets = ['b_cells' ... 'cytotoxic_t'], now its dynamic :*

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel(component_name + " 1 ", fontsize = 15)
ax.set_ylabel(component_name + " 2 ", fontsize = 15)
ax.set_title('Most Powerful '+ component_name +'s', fontsize = 20)
colors = cm.rainbow(np.linspace(0, 1, len(targets)))
for target, color in zip(targets,colors):
    indicesToKeep = df['celltype'] == target
    ax.scatter(df.loc[indicesToKeep, component_name + ' 1']
               , df.loc[indicesToKeep, component_name + ' 2']
               , c = color.reshape(1,-1)
               , s = 5)
ax.legend(targets)
ax.grid()
plt.savefig(output_dir + "/UMAP_result.png")




# %% Diagnostics


# so for the colours this might be a bit weird, but i'm just adapting to what they gave me in this other tutorial. the first plot is better anyways
targetlist = list(targets)
colourindexes = range(len(targets))
colourdictionary = dict(zip(targetlist, colourindexes))

import seaborn as sns

plt.scatter(
    newdata[:, 0],
    newdata[:, 1],
    s = 1,
    alpha = 0.5,
    marker = ",",
    c=[sns.color_palette()[x] for x in df.celltype.map(colourdictionary)])
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the single cells', fontsize=24)
plt.savefig(output_dir + "/UMAP_plot_scatter.png")   






print(datetime.now().strftime("%H:%M:%S>"), "Script terminated successfully")



