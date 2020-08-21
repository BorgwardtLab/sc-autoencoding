# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 00:49:30 2020

@author: Mike Toreno II
"""


# %% Load Data
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import matplotlib.cm as cm # colourpalette
import argparse
import sys


print(datetime.now().strftime("%H:%M:%S>"), "Starting sca_ICA.py")


try:
    os.chdir(os.path.dirname(sys.argv[0]))
except:
    pass
         


parser = argparse.ArgumentParser(description = "calculate ICAs")  #required
parser.add_argument("-n","--num_components", help="the number of ICAs to calculate", type = int, default = 30)
parser.add_argument("-s", "--nosave", help="passing this flag prevents the program from saving the reduced coordinates to prevent storage issues. (plots and other output still gets saved)", action="store_true")
parser.add_argument("-i","--input_dir", help="input directory", default = "../inputs/baselines/baseline_data/scaPCA_output/")
parser.add_argument("-o","--output_dir", help="output directory", default = "../inputs/baselines/baseline_data/scaICA_output/")
parser.add_argument("-p","--outputplot_dir", help="plot directory", default = "../outputs/baselines/baseline_data/scaICA_output/")
args = parser.parse_args() #required



input_path = args.input_dir
output_dir = args.output_dir
outputplot_dir = args.outputplot_dir
component_name = "IC"
num_components = args.num_components




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


test_index = np.loadtxt(fname = input_path + "test_index.tsv", dtype = bool)
train_index = np.logical_not(test_index)







# %% replacing data with testdata only

original_data = data

testdata = data[test_index]
data = data[train_index]





# %% do ICA


print(datetime.now().strftime("%H:%M:%S>"), "scaling data...")
myscaler = StandardScaler()
data =  myscaler.fit_transform(data)
testdata = myscaler.transform(testdata)


print(datetime.now().strftime("%H:%M:%S>"), "calculating independant components...")
ica = FastICA(n_components=num_components)
ICs = ica.fit_transform(data)
test_ICs = ica.transform(testdata)




#%% Outputs



if not os.path.exists(outputplot_dir):
    print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Directory...")
    os.makedirs(outputplot_dir)
    


### Create Plot
print(datetime.now().strftime("%H:%M:%S>"), "drawing plots...")
targets = set(labels) # what it will draw in plot, previously it was targets = ['b_cells' ... 'cytotoxic_t'], now its dynamic :*

# construct dataframe for 2d plot
df = pd.DataFrame(data = ICs[:,[0,1]], columns = [component_name + '_1', component_name + '_2'])
df['celltype'] = labels




# %%

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel(component_name + "_1", fontsize = 15)
ax.set_ylabel(component_name + "_2", fontsize = 15)
ax.set_title('Most Powerful ICAs', fontsize = 20)
colors = cm.rainbow(np.linspace(0, 1, len(targets)))
for target, color in zip(targets,colors):
    indicesToKeep = df['celltype'] == target
    ax.scatter(df.loc[indicesToKeep, component_name + '_1']
               , df.loc[indicesToKeep, component_name + '_2']
               , c = color.reshape(1,-1)
               , s = 5)
ax.legend(targets)
ax.grid()
plt.savefig(outputplot_dir + "ICA_plot.png")


# Loading scores for PC1

how_many = 10

#generate index
index = []
for i in range(len(ica.components_[0,:])):
    index.append("PC " + str(i))
    
loading_scores = pd.Series(ica.components_[0,:], index = index)
sorted_loading_scores = loading_scores.abs().sort_values(ascending=False)
top_genes = sorted_loading_scores[0:how_many].index.values
    

file = open(outputplot_dir + 'most_important_components.log', 'w')
for i in range(how_many):
    text = (str(top_genes[i]) + "\t" + str(sorted_loading_scores[i]) + "\n")
    file.write(text)
file.close()



# %% Recombine Data
outdata = np.zeros(shape = (original_data.shape[0], num_components))


outdata[train_index] = ICs
outdata[test_index] = test_ICs





# %% Saving the data

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
    

print(datetime.now().strftime("%H:%M:%S>"), "sca_ICA.py terminated successfully\n")
































