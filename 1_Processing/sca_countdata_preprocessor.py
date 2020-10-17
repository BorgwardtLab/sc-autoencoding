# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 15:07:08 2020

@author: Mike Toreno II
"""


import os
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io

from sklearn.model_selection import train_test_split




print(datetime.now().strftime("%H:%M:%S>"), "Starting sca_countdata_preprocessor.py")


try:
    os.chdir(os.path.dirname(sys.argv[0]))
except:
    pass
         


parser = argparse.ArgumentParser(description = "program to preprocess the raw singlecell data")  #required
parser.add_argument("-i","--input_dir", help="input directory", default = "../inputs/data/raw_input_combined/filtered_matrices_mex/hg19/")
parser.add_argument("-o","--output_dir", help="output directory", default = "../inputs/sca/sca_preprocessed_data/")

parser.add_argument("-v","--verbosity", help="level of verbosity", default = 3, choices = [0, 1, 2, 3], type = int)
parser.add_argument("--test_fraction", help="enter a float between 0-1. This will be the fraction of the data, that is marked as test data.", default = 0.25, type = float)


parser.add_argument("--mingenes", help="minimal amount of genes per cell", default = 200, type = int)
parser.add_argument("--mincells", help="minimal number of cells for a gene", default = 5, type = int)

parser.add_argument("--maxfeatures", help="maximal number of genes per cell (check plot)", default = 1500, type = int)
parser.add_argument("--maxmito", help="maximal percentage of mitochondrial counts", default = 5, type = int)

parser.add_argument("--features", help="number of highly variable features to catch", default = 3000, type = int)
parser.add_argument("--limit_cells", help="only take a certain number of samples, to keep the countmatrix small", default = 0, type = int)

args = parser.parse_args() #required



input_dir = args.input_dir
output_dir = args.output_dir

min_genes_per_cell = args.mingenes
min_cells_per_gene = args.mincells

max_num_features = args.maxfeatures
max_mt_perc = args.maxmito

num_top_genes = args.features 



if not os.path.exists(output_dir):
    print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Directory...")
    os.makedirs(output_dir)
    
    
# fs_min_mean = 0.0125
# fs_max_mean = 3
# fs_min_disp = 0.5



# %% Load Data

print(datetime.now().strftime("%H:%M:%S>"), "reading input data...")


### Get Matrix
coomatrix = scipy.io.mmread(input_dir + "matrix.mtx")
coomatrix = np.transpose(coomatrix) # samples must be rows, variables = columns

genes = pd.read_csv(input_dir + "genes.tsv", delimiter = "\t", header = None)

barcodes = pd.read_csv(input_dir + "barcodes.tsv", delimiter = "\t", header = None)



# %% ScanPy Setup

print(datetime.now().strftime("%H:%M:%S>"), "Launching Scanpy")

sc.settings.verbosity = args.verbosity             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_versions()
sc.settings.set_figure_params(dpi=80, facecolor='white')

results_file = 'write/pbmc3k.h5ad'  # the file that will store the analysis results









# %% Read data into anndata object (holds slots for annotation etc)

AnnData = sc.read_10x_mtx(path = input_dir, var_names = "gene_ids", cache = False)


# %% basic filtering
print(datetime.now().strftime("%H:%M:%S>"), "Filtering Data with min_genes= {a:d} and min_cells= {b:d}...".format(a = min_genes_per_cell, b=min_cells_per_gene))


sc.pp.filter_cells(AnnData, min_genes = min_genes_per_cell) # only keep cells with at least 200 genes detecte
# could also pass counts instead of genes

sc.pp.filter_genes(AnnData, min_cells=min_cells_per_gene) # and only keep genes that are present in at least # cells


# %% Calculate numbers

# flag each gene if mitochondrial
AnnData.var['mt'] = AnnData.var['gene_symbols'].str.startswith(('MT-', 'MT.', 'MT\\'))  # annotate the group of mitochondrial genes as 'mt'
# only finds like 13 genes :/ But technically they do the same as seurat, so must be same?

sc.pp.calculate_qc_metrics(AnnData, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
# adds(for each gene) n_cells_by_count, mean counts, pct_dropbout by counts, total counts
# the qc_vars = ['mt'] does not influence the calculations of AnnData.var,
# but instead the percentage calculation is made into AnnData.obs



#%% Filter Mitochondrial Genes and those with too many genes
print(datetime.now().strftime("%H:%M:%S>"), "Filtering Data with n_genes_by_count < {a:d} and pct_counts_mt < {b:d}...".format(a = max_num_features, b = max_mt_perc))

# slicing
AnnData = AnnData[AnnData.obs.n_genes_by_counts < max_num_features, :]
AnnData = AnnData[AnnData.obs.pct_counts_mt < max_mt_perc, :]




### generate counts-only object
adatacounts = AnnData.copy()





# %% Normalize (for AnnData only, to find variable genes)
print(datetime.now().strftime("%H:%M:%S>"), "Normalizing...")
sc.pp.normalize_total(AnnData, target_sum=1e4)

# %% Logarithmize (for AnnData only, to find variable genes)
print(datetime.now().strftime("%H:%M:%S>"), "Logarithmizing...")
sc.pp.log1p(AnnData)
    





# %% Feature Selection
print(datetime.now().strftime("%H:%M:%S>"), "Doing feature selection with {a:d} highly variable genes...".format(a = num_top_genes))

sc.pp.highly_variable_genes(AnnData, n_top_genes = num_top_genes)



# remove non variable features

AnnData = AnnData[:, AnnData.var.highly_variable]
adatacounts = adatacounts[:, AnnData.var.highly_variable.index]


densematrix = scipy.sparse.csr_matrix(adatacounts.X).todense()




# %% Exporting




if args.limit_cells > 0:
    num = args.limit_cells
    assert isinstance(num, int)

    lucky_cells_idx = np.linspace(0, len(densematrix)-1, num, dtype = int)
    densematrix = densematrix[lucky_cells_idx, :]


# those are useless, only mean and dispersion etc
# AnnData.write_csvs("filename2", skip_data=False)

genes = pd.DataFrame(adatacounts.var_names)
genes["symbols"] = list(adatacounts.var["gene_symbols"])

panda = pd.DataFrame(densematrix) #obs*vars

barcodelist = list(adatacounts.obs_names)
# alright, I give up. Lets' do it cavemen style
bc_names = [item.split('\t')[0] for item in barcodelist]
bc_types = [item.split('\t')[1] for item in barcodelist]
# damn i even have to caveman the df creation, i suck so hard
barcodes = pd.DataFrame(data = bc_names)
barcodes["type"] = bc_types

''' the reason why i wrote this ugly blcok is, if i just panda'd the AnnData.obs_name, it would
write it out with quotatation marks around it, (probably due to the inclusion of the \t, that forces 
it to somehow keep the string as one (what i didn't want, as its two columns)). I've tried to just disable quotation marks
with quotin=csv.QUOTE_NONE, but then it wanted anothe rescape character, and I gave up. And then I've tried
to split an array of strings in two in a nice manner, but had to give up, and do it with these for items now
I think this equals to 2 loops, so awesome for runtime (not that it matters) 
anyway, this is a bad solution, but it fixes the problem, so meh'''


if args.limit_cells > 0:
    barcodes = barcodes.iloc[lucky_cells_idx, :]








# %% Train Test Split

print(datetime.now().strftime("%H:%M:%S>"), "Creating Train Test Split")

X_train, X_test, y_train, y_test = train_test_split(panda, bc_types, test_size=args.test_fraction)

train_indexes = list(X_train.index)
test_indexes = list(X_test.index)
test_index = np.zeros(len(bc_types), dtype = bool)

for i in test_indexes:
    test_index[i] = True

np.savetxt(output_dir + "test_index.tsv", test_index, fmt = "%d")






''' so at this point we have to filter out zerogenes again, as the train-test split 
may have "created" additional zero genes. 

I think its easiest doing this caveman-style: instead of going back to anndata, manually take them out of the pandas.
'''
# %%
# get zerogenes


rowsums = X_test.sum(axis = 0)
testzg = np.where(rowsums < 5)
testzg = np.array(testzg)

rowsums = X_train.sum(axis = 0)
trainzg = np.where(rowsums < 5)
trainzg = np.array(trainzg)



nonzeros = np.concatenate([testzg, trainzg], axis = 1)
nonzeros = nonzeros.flatten()
rowsums = panda.sum(axis = 0) #jff



panda_nz = panda.copy()
panda_nz = panda_nz.drop(nonzeros, axis = 1)

genes_nz = genes.copy()
genes_nz = genes_nz.drop(nonzeros, axis = 0)


# %%
print(datetime.now().strftime("%H:%M:%S>"), "Generating Output...")


panda_nz.to_csv(output_dir + "matrix.tsv", sep = "\t", index = False, header = False)
genes_nz.to_csv(output_dir + "genes.tsv", sep = "\t", index = False, header = False)
barcodes.to_csv(output_dir + "barcodes.tsv", sep = "\t", index = False, header = False)









# %% to generate "transposed" outdata with headers for the vanilla DCA


panda_nzt = panda_nz.copy()
panda_nzt = panda_nzt.transpose()

joined_barcodes = [x.replace("\t", "_") for x in barcodelist]
panda_nzt.columns = joined_barcodes

panda_nzt.index = genes_nz.iloc[:,0]




panda_nzt.to_csv(output_dir + "matrix_transposed.tsv", sep = "\t", index = True, header = True)
  

print(datetime.now().strftime("%H:%M:%S>"), "dca_countdata_preprocessor.py terminated successfully\n")
