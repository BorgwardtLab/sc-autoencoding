# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 01:25:34 2020

@author: Simon Streib
"""


# %% Load Data
import scipy.io
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime






print(datetime.now().strftime("%H:%M:%S>"), "reading input matrix...")
### Get Matrix
mtx_file = "./input/filtered_matrices_mex/hg19/matrix.mtx"
coomatrix = scipy.io.mmread(mtx_file)

coomatrix = np.transpose(coomatrix) # samples must be rows, variables = columns



print(datetime.now().strftime("%H:%M:%S>"), "converting sparse matrix to dense...")
data = coomatrix.toarray()


print(datetime.now().strftime("%H:%M:%S>"), "reading labels...")
### Get Labels
lbl_file = "./input/filtered_matrices_mex/hg19/celltype_labels.tsv"
file = open(lbl_file, "r")
labels = file.read().split("\n")
file.close()
labels.remove("") #last, empty line is also removed



# load genes (for last task, finding most important genes)
file = open("./input/filtered_matrices_mex/hg19/genes.tsv", "r")
genes = file.read().split("\n")
file.close()


# %%  Cut back data for handlability lmao

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

# labels = labels[cells_downlimit:cells_uplimit]


# reduced = coomatrix.tocsr()

# data = reduced[cells_downlimit:cells_uplimit, genes_downlimit:genes_uplimit]
# data = data.toarray()



# genes = genes[genes_downlimit:genes_uplimit]

# %% do PCA

print(datetime.now().strftime("%H:%M:%S>"), "scaling data...")
data = StandardScaler().fit_transform(data) # Standardizing the features


print(datetime.now().strftime("%H:%M:%S>"), "calculating principal components...")
myPCA = PCA()
PCs = myPCA.fit_transform(data)


# construct dataframe for 2d plot
df = pd.DataFrame(data = PCs[:,[0,1]], columns = ['principal component 1', 'principal component 2'])
df['celltlype'] = labels




explained_variance = myPCA.explained_variance_ratio_


#%% Outputs
if not os.path.exists("./scaPCA_output"):
    print("Creating Output Directory...")
    os.makedirs("./scaPCA_output")
    


### Create Plot
print(datetime.now().strftime("%H:%M:%S>"), "drawing plots...")
targets = set(labels) # what it will draw in plot, previously it was targets = ['b_cells' ... 'cytotoxic_t'], now its dynamic :*

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('PC1 (' + str(round(explained_variance[0]*100, 3)) + "% of variance)", fontsize = 15)
ax.set_ylabel('PC2 (' + str(round(explained_variance[1]*100, 3)) + "% of variance)", fontsize = 15)
ax.set_title('Most Powerful PCAs', fontsize = 20)
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = df['celltlype'] == target
    ax.scatter(df.loc[indicesToKeep, 'principal component 1']
               , df.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 5)
ax.legend(targets)
ax.grid()
plt.savefig("./scaPCA_output/PCA_result.png")





### Save Variances
print(datetime.now().strftime("%H:%M:%S>"), "saving explained variances...")
explained_sum = np.cumsum(explained_variance)

file = open('./scaPCA_output/explained_variances.log', 'w')
for i in range(len(explained_variance)):
    text = (str(i + 1) + "\t" + str(explained_variance[i]) + "\t" + str(explained_sum[i]) + "\n")
    file.write(text)
file.close()
    
    
    
    
    
### Scree Plots
how_many = -1;

perc_var = (explained_variance * 100)
perc_var = perc_var[0:how_many]

labelz = [str(x) for x in range(1, len(perc_var)+1)]


plt.figure(figsize=[16,8])
plt.bar(x = range(1, len(perc_var)+1), height = perc_var, tick_label = labelz)
plt.ylabel('Percentage of explained variance')
plt.xlabel('Principal component')
plt.title('Scree plot')
plt.show()    
plt.savefig("./scaPCA_output/PCA_scree_plot_all.png")
    
    
    
    
how_many = 50;

perc_var = (explained_variance * 100)
perc_var = perc_var[0:how_many]

labelz = [str(x) for x in range(1, len(perc_var)+1)]


plt.figure(figsize=[16,8])
plt.bar(x = range(1, len(perc_var)+1), height = perc_var, tick_label = labelz)
plt.ylabel('Percentage of explained variance')
plt.xlabel('Principal component')
plt.title('Scree plot')
plt.show()    
plt.savefig("./scaPCA_output/PCA_scree_plot_top50.png")    
    
    
    
    
    
# Loading scores for PC1


how_many = 10

loading_scores = pd.Series(myPCA.components_[0], index = genes)
sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
top_genes = sorted_loading_scores[0:how_many].index.values
    

file = open('./scaPCA_output/most_important_genes.log', 'w')
for i in range(how_many):
    text = (str(top_genes[i]) + "\t" + str(sorted_loading_scores[i]) + "\n")
    file.write(text)
file.close()

print(datetime.now().strftime("%H:%M:%S>"), "Script terminated successfully")

# %% Diagnostics





