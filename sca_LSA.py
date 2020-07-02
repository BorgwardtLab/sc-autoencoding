# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 23:52:35 2020

@author: Mike Toreno II
"""


# %% Load Data
import scipy.io
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import matplotlib.cm as cm # colourpalette




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

num_lsa = data.shape[1]-1


# %%  for local execution (remove for full picture)

# num_lsa = 100




# %%



print(datetime.now().strftime("%H:%M:%S>"), "scaling data...")
data = StandardScaler(with_mean= False).fit_transform(data) # Standardizing the features



print(datetime.now().strftime("%H:%M:%S>"), "Calculating LSA...")
svd = TruncatedSVD(n_components = num_lsa)
svd.fit(data)

#lsa = latent semantic analysis
lsa = svd.transform(data)


# %%



# construct dataframe for 2d plot
df = pd.DataFrame(data = lsa[:,[0,1]], columns = ['LS 1', 'LS 2'])
df['celltlype'] = labels

df2 = df


explained_variance = svd.explained_variance_ratio_


#%% Outputs
if not os.path.exists("./scaLSA_output"):
    print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Directory...")
    os.makedirs("./scaLSA_output")
    




### Create Plot
print(datetime.now().strftime("%H:%M:%S>"), "drawing plots...")
targets = set(labels) # what it will draw in plot, previously it was targets = ['b_cells' ... 'cytotoxic_t'], now its dynamic :*

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('LS1 (' + str(round(explained_variance[0]*100, 3)) + "% of variance)", fontsize = 15)
ax.set_ylabel('LS2 (' + str(round(explained_variance[1]*100, 3)) + "% of variance)", fontsize = 15)
ax.set_title('Most Powerful LSAs', fontsize = 20)

colors = cm.rainbow(np.linspace(0, 1, len(targets)))
for target, color in zip(targets,colors):
    indicesToKeep = df['celltlype'] == target
    ax.scatter(df.loc[indicesToKeep, 'LS 1']
               , df.loc[indicesToKeep, 'LS 2']
               , c = color.reshape(1,-1)
               , s = 1)
ax.legend(targets)
ax.grid()
plt.savefig("./scaLSA_output/LSA_result.png")





### Save Variances
print(datetime.now().strftime("%H:%M:%S>"), "saving explained variances...")
explained_sum = np.cumsum(explained_variance)

file = open('./scaLSA_output/explained_variances.log', 'w')
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
plt.savefig("./scaLSA_output/LSA_scree_plot_all.png")
    
    
    
    
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
plt.savefig("./scaLSA_output/LSA_scree_plot_top50.png")    
    
    
    
    
    
    
# %%
    
    
# Loading scores for PC1

how_many = 10

loading_scores = pd.Series(svd.components_[0], index = genes)



sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
top_genes = sorted_loading_scores[0:how_many].index.values
    

file = open('./scaLSA_output/most_important_genes.log', 'w')
for i in range(how_many):
    text = (str(top_genes[i]) + "\t" + str(sorted_loading_scores[i]) + "\n")
    file.write(text)
file.close()

print(datetime.now().strftime("%H:%M:%S>"), "Script terminated successfully")

# %% Diagnostics












