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
import sys
import argparse

print(datetime.now().strftime("%H:%M:%S>"), "Starting sca_tSNE.py")

try:
    os.chdir(os.path.dirname(sys.argv[0]))
except:
    pass



#os.chdir(os.path.dirname(sys.argv[0]))
input_path = "../inputs/raw_input_combined/filtered_matrices_mex/hg19/"



parser = argparse.ArgumentParser(description = "calculates a tSNE embedding")  #required
parser.add_argument("-n","--num_components", default = 2, help="the number of coordinates to calculate (default = 2). For any number > 3, another algorithm (exact) is used, which hasn't been tested.", type = int)
parser.add_argument("-s", "--nosave", help="passing this flag prevents the program from saving the reduced coordinates to prevent storage issues. (plots and other output still gets saved)", action="store_true")
parser.add_argument("-i","--input_dir", help="input directory", default = "../inputs/preprocessed_data/")
parser.add_argument("-o","--output_dir", help="output directory", default = "../outputs/scaTSNE_output/")
parser.add_argument("-p","--outputplot_dir", help="plot directory", default = "../outputs/scaTSNE_output/")
args = parser.parse_args() #required



input_path = args.input_dir
output_dir = args.output_dir
outputplot_dir = args.outputplot_dir
component_name = "t-SNE"




# %% Read Input data

matrix_file = input_path + "matrix.tsv"
mat = np.loadtxt(open(matrix_file), delimiter="\t")
data = np.transpose(mat)


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
# print(datetime.now().strftime("%H:%M:%S>"), "reading input matrix...")
# ### Get Matrix
# mtx_file = input_path + "matrix.mtx"
# coomatrix = scipy.io.mmread(mtx_file)
# coomatrix_t = np.transpose(coomatrix)

# print(datetime.now().strftime("%H:%M:%S>"), "deleting random data pieces...")
# genes_uplimit = 30000
# genes_downlimit = 25000
# cells_uplimit = 15000
# cells_downlimit = 10000
# labels = labels[cells_downlimit:cells_uplimit]
# genes = genes[genes_downlimit:genes_uplimit]
# csrmatrix = coomatrix_t.tocsr()
# coomatrix_t = csrmatrix[cells_downlimit:cells_uplimit, genes_downlimit:genes_uplimit]


# Convert to dense
# print(datetime.now().strftime("%H:%M:%S>"), "converting sparse matrix to dense...")
#data = coomatrix_t.toarray()




# %% setup defaults

if isinstance(args.num_components, int) and args.num_components > 3:
    mymethod = 'exact'
else:
    mymethod = "barnes_hut"


    
num_components = args.num_components



# %%

print(datetime.now().strftime("%H:%M:%S>"), "scaling data...")
data = StandardScaler(with_mean= False).fit_transform(data) # Standardizing the features

# %%


print(datetime.now().strftime("%H:%M:%S>"), "Calculating tSNE...")
tsne = TSNE(n_components=num_components, verbose = 3, method= mymethod)
tsnedata = tsne.fit_transform(data)


embed = tsne.embedding_
kldiver = tsne.kl_divergence_
niter = tsne.n_iter_




#%% Outputs



# construct dataframe for 2d plot
df = pd.DataFrame(data = tsnedata[:,[0,1]], columns = [ component_name + ' 1', component_name + ' 2'])
df['celltype'] = labels



if not os.path.exists(outputplot_dir):
    print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Directory...")
    os.makedirs(outputplot_dir)
    


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
    indicesToKeep = df['celltype'] == target
    ax.scatter(df.loc[indicesToKeep, component_name + ' 1']
               , df.loc[indicesToKeep, component_name + ' 2']
               , c = color.reshape(1,-1)
               , s = 5)
ax.legend(targets)
ax.grid()
plt.savefig(outputplot_dir + "/tSNE_Plot.png")




# %% Diagnostics

if args.nosave == False:
    
    print(datetime.now().strftime("%H:%M:%S>"), "Saving output...")
    
    np.savetxt(output_dir + "result_tSNE_coordinates.tsv", tsnedata, delimiter = "\t")
    
    
    with open(output_dir + "result_genes.tsv", "w") as outfile:
        outfile.write("\n".join(genes))
    
    with open(output_dir + "result_barcodes.tsv", "w") as outfile:
        outfile.write("\n".join(barcodes))
    
    with open(output_dir + "result_celltype_labels.tsv", "w") as outfile:
        outfile.write("\n".join(labels))


print(datetime.now().strftime("%H:%M:%S>"), "sca_tSNE.py terminated successfully")



