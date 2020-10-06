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
import umap
import sys
import argparse

print(datetime.now().strftime("%H:%M:%S>"), "Starting sca_UMAP.py")


try:
    os.chdir(os.path.dirname(sys.argv[0]))
except:
    pass



#os.chdir(os.path.dirname(sys.argv[0]))
input_dir = "../inputs/raw_input_combined/filtered_matrices_mex/hg19/"



parser = argparse.ArgumentParser(description = "calculates a UMAP embedding")  #required
parser.add_argument("-n","--num_components", help="the number of coordinates to calculate", type = int, default = 2)
parser.add_argument("-s", "--nosave", help="passing this flag prevents the program from saving the reduced coordinates to prevent storage issues. (plots and other output still gets saved)", action="store_true")
parser.add_argument("-i","--input_dir", help="input directory", default = "../inputs/baselines/baseline_data/scaPCA_output/")
parser.add_argument("-o","--output_dir", help="output directory", default = "../inputs/baselines/baseline_data/scaUMAP_output/")
parser.add_argument("-p","--outputplot_dir", help="plot directory", default = "../outputs/baselines/baseline_data/scaUMAP_output/")
parser.add_argument("-v","--verbosity", help="level of verbosity", default = 3, choices = [0, 1, 2, 3], type = int)
args = parser.parse_args() #required



input_dir = args.input_dir
output_dir = args.output_dir
outputplot_dir = args.outputplot_dir
component_name = "UMAP"
num_components = args.num_components


# %% Read Input data
print(datetime.now().strftime("%H:%M:%S>"), "reading input data...")

matrix_file = input_dir + "matrix.tsv"
data = np.loadtxt(open(matrix_file), delimiter="\t")



# load genes (for last task, finding most important genes)
file = open(input_dir + "genes.tsv", "r")
genes = file.read().split("\n")
file.close()
genes.remove("") 


barcodes = pd.read_csv(input_dir + "barcodes.tsv", delimiter = "\t", header = None)
labels = barcodes.iloc[:,1]


test_index = np.loadtxt(fname = input_dir + "test_index.tsv", dtype = bool)
train_index = np.logical_not(test_index)




# %%

original_data = data

testdata = data[test_index]
data = data[train_index]





# %%


print(datetime.now().strftime("%H:%M:%S>"), "scaling data...")
myscaler = StandardScaler()
data =  myscaler.fit_transform(data)
testdata = myscaler.transform(testdata)


print(datetime.now().strftime("%H:%M:%S>"), "calculating UMAP...")
reducer = umap.UMAP(verbose = args.verbosity, n_components = num_components)
newdata = reducer.fit_transform(data)
new_testdata = reducer.transform(testdata)





#%% Outputs


# construct dataframe for 2d plot
df = pd.DataFrame(data = newdata[:,[0,1]], columns = [ component_name + ' 1', component_name + ' 2'])
df['celltype'] = labels



if not os.path.exists(outputplot_dir):
    print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Directory...")
    os.makedirs(outputplot_dir)
    








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
plt.savefig(outputplot_dir + "/UMAP_plot.png")






import seaborn as sns
colourdictionary = dict(zip(list(targets), range(len(targets))))

plt.scatter(
    newdata[:, 0],
    newdata[:, 1],
    s = 1,
    alpha = 0.5,
    marker = ",",
    c=[sns.color_palette()[x] for x in df.celltype.map(colourdictionary)])
plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the single cells with sns', fontsize=24)
plt.savefig(outputplot_dir + "/UMAP_plot_scatter.png")   






# %% recombine data


outdata = np.zeros(shape = (original_data.shape[0], num_components))

outdata[train_index] = newdata
outdata[test_index] = new_testdata






# %% output


if args.nosave == False:
    
    if not os.path.exists(output_dir):
        print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Directory...")
        os.makedirs(output_dir)    

    print(datetime.now().strftime("%H:%M:%S>"), "Saving output...")
    
    np.savetxt(output_dir + "matrix.tsv", outdata, delimiter = "\t")
    
    
    with open(output_dir + "genes.tsv", "w") as outfile:
        outfile.write("\n".join(genes))
    
    barcodes.to_csv(output_dir + "barcodes.tsv", sep = "\t", index = False, header = False)

    np.savetxt(output_dir + "test_index.tsv", test_index, fmt = "%d")
    

print(datetime.now().strftime("%H:%M:%S>"), "sca_UMAP terminated successfully\n")





