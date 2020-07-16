# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 23:52:35 2020

@author: Mike Toreno II
"""

# %% Load Data
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
parser.add_argument("-i","--input_dir", help="input directory", default = "../inputs/baseline_data/scaPCA_output/")
parser.add_argument("-o","--output_dir", help="output directory", default = "../inputs/baseline_data/scaTSNE_output/")
parser.add_argument("-p","--outputplot_dir", help="plot directory", default = "../outputs/baseline_data/scaTSNE_output/")
args = parser.parse_args() #required



input_path = args.input_dir
output_dir = args.output_dir
outputplot_dir = args.outputplot_dir
component_name = "t-SNE"




# %% Read Input data
print(datetime.now().strftime("%H:%M:%S>"), "reading input data...")

matrix_file = input_path + "matrix.tsv"
data = np.loadtxt(open(matrix_file), delimiter="\t")



# load genes (for last task, finding most important genes)
file = open(input_path + "genes.tsv", "r")
genes = file.read().split("\n")
file.close()
genes.remove("") 


barcodes = pd.read_csv(input_path + "barcodes.tsv", delimiter = "\t", header = None)
labels = barcodes.iloc[:,1]




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
    
    if not os.path.exists(output_dir):
        print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Directory...")
        os.makedirs(output_dir)    
    
    print(datetime.now().strftime("%H:%M:%S>"), "Saving output...")
    
    np.savetxt(output_dir + "matrix.tsv", tsnedata, delimiter = "\t")
    
    with open(output_dir + "genes.tsv", "w") as outfile:
        outfile.write("\n".join(genes))
    
    barcodes.to_csv(output_dir + "barcodes.tsv", sep = "\t", index = False, header = False)


print(datetime.now().strftime("%H:%M:%S>"), "sca_tSNE.py terminated successfully\n")



