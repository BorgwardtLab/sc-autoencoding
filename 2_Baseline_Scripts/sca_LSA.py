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
import argparse
import sys
from sklearn.decomposition import TruncatedSVD




try:
    os.chdir(os.path.dirname(sys.argv[0]))
except:
    pass



#os.chdir(os.path.dirname(sys.argv[0]))
input_path = "../inputs/raw_input_combined/filtered_matrices_mex/hg19/"



parser = argparse.ArgumentParser(description = "calculate PCAs")  #required
parser.add_argument("-n","--num_components", help="the number of LSA components to calculate", type = int)
args = parser.parse_args() #required





# %% Read Input data

print(datetime.now().strftime("%H:%M:%S>"), "reading input matrix...")
### Get Matrix
mtx_file = input_path + "matrix.mtx"
coomatrix = scipy.io.mmread(mtx_file)
data = np.transpose(coomatrix)


### Get Labels
print(datetime.now().strftime("%H:%M:%S>"), "reading labels...")
lbl_file = input_path + "celltype_labels.tsv"
file = open(lbl_file, "r")
labels = file.read().split("\n")
file.close()
labels.remove("") #last, empty line is also removed


# load genes (for last task, finding most important genes)
file = open(input_path + "genes.tsv", "r")
genes = file.read().split("\n")
file.close()
genes.remove("") 


# load barcodes
file = open(input_path + "barcodes.tsv", "r")
barcodes = file.read().split("\n")
file.close()
barcodes.remove("") 



# %%  Cut back data for handlability lmao

# print(datetime.now().strftime("%H:%M:%S>"), "deleting random data pieces...")
# genes_uplimit = 30000
# genes_downlimit = 25000
# cells_uplimit = 15000
# cells_downlimit = 10000


# labels = labels[cells_downlimit:cells_uplimit]

# genes = genes[genes_downlimit:genes_uplimit]

# csrmatrix = data.tocsr()
# data = csrmatrix[cells_downlimit:cells_uplimit, genes_downlimit:genes_uplimit]




# %%  for local execution (remove for full picture)

if args.num_components == None:
    num_lsa = data.shape[1]-1
else:
    num_lsa = args.num_components
    



# %% doing LSA

print(datetime.now().strftime("%H:%M:%S>"), "scaling data...")
data = StandardScaler(with_mean= False).fit_transform(data) # Standardizing the features


print(datetime.now().strftime("%H:%M:%S>"), "Calculating LSA...")
svd = TruncatedSVD(n_components = num_lsa)
svd.fit(data)

#lsa = latent semantic analysis
lsa = svd.transform(data)



#%% Outputs

output_dir = "../outputs/scaLSA_output/"
component_name = "LS"

explained_variance = svd.explained_variance_ratio_


if not os.path.exists(output_dir):
    print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Directory...")
    os.makedirs(output_dir)
    





### Create Plot
print(datetime.now().strftime("%H:%M:%S>"), "drawing plots...")
targets = set(labels) # 


# construct dataframe for 2d plot
df = pd.DataFrame(data = lsa[:,[0,1]], columns = [component_name + '_1', component_name + '_2'])
df['celltype'] = labels

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel(component_name + '1 (' + str(round(explained_variance[0]*100, 3)) + "% of variance)", fontsize = 15)
ax.set_ylabel(component_name + '2 (' + str(round(explained_variance[1]*100, 3)) + "% of variance)", fontsize = 15)
ax.set_title('Most Powerful LSAs', fontsize = 20)

colors = cm.rainbow(np.linspace(0, 1, len(targets)))
for target, color in zip(targets,colors):
    indicesToKeep = df['celltype'] == target
    ax.scatter(df.loc[indicesToKeep, component_name + '_1']
               , df.loc[indicesToKeep, component_name + '_2']
               , c = color.reshape(1,-1)
               , s = 1)
ax.legend(targets)
ax.grid()
plt.savefig(output_dir + "LSA_result.png")





### Save Variances
print(datetime.now().strftime("%H:%M:%S>"), "saving explained variances...")
explained_sum = np.cumsum(explained_variance)

file = open(output_dir + 'explained_variances.log', 'w')
for i in range(len(explained_variance)):
    text = (str(i + 1) + "\t" + str(explained_variance[i]) + "\t" + str(explained_sum[i]) + "\n")
    file.write(text)
file.close()
    
    
    
### Scree Plots

perc_var = (explained_variance * 100)

labelz = [str(x) for x in range(1, len(perc_var)+1)]


plt.figure(figsize=[16,8])
plt.bar(x = range(1, len(perc_var)+1), height = perc_var, tick_label = labelz)
plt.ylabel('Percentage of explained variance')
plt.xlabel('Principal component')
plt.title('Scree plot')
plt.show()    
plt.savefig(output_dir + "LSA_scree_plot_all.png")
    
    
    
    
    
if num_lsa > 50:
    how_many = 50;
    perc_var = (explained_variance * num_lsa)
    perc_var = perc_var[0:how_many]

    labelz = [str(x) for x in range(1, len(perc_var)+1)]
    
    plt.figure(figsize=[16,8])
    plt.bar(x = range(1, len(perc_var)+1), height = perc_var, tick_label = labelz)
    plt.ylabel('Percentage of explained variance')
    plt.xlabel('Principal component')
    plt.title('Scree plot')
    plt.show()    
    plt.savefig(output_dir + "LSA_scree_plot_top50.png")    
    
    
    
 
    
# Loading scores for PC1

how_many = 10

loading_scores = pd.Series(svd.components_[0], index = genes)



sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
top_genes = sorted_loading_scores[0:how_many].index.values
    

file = open(output_dir + 'most_important_genes.log', 'w')
for i in range(how_many):
    text = (str(top_genes[i]) + "\t" + str(sorted_loading_scores[i]) + "\n")
    file.write(text)
file.close()




# %% saving data
print(datetime.now().strftime("%H:%M:%S>"), "Saving output...")

np.savetxt(output_dir + "result_PCA.tsv", lsa, delimiter = "\t")


with open(output_dir + "result_genes.tsv", "w") as outfile:
    outfile.write("\n".join(genes))

with open(output_dir + "result_genes.tsv", "w") as outfile:
    outfile.write("\n".join(genes))

with open(output_dir + "result_barcodes.tsv", "w") as outfile:
    outfile.write("\n".join(barcodes))

with open(output_dir + "result_celltype_labels.tsv", "w") as outfile:
    outfile.write("\n".join(labels))


print(datetime.now().strftime("%H:%M:%S>"), "Script terminated successfully")











