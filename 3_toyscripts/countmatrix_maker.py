# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 14:09:24 2020

@author: Mike Toreno II
"""

import numpy as np
import pandas as pd
from datetime import datetime
import scipy.io




### Get Matrix
mtx_file = "../inputs/raw_input_combined/filtered_matrices_mex/hg19/matrix.mtx"
coomatrix = scipy.io.mmread(mtx_file)


genesdf = pd.read_csv("../inputs/raw_input_combined/filtered_matrices_mex/hg19/genes.tsv", delimiter = "\t", header = None)
genes = genesdf.iloc[:,0]

barcodes = pd.read_csv("../inputs/raw_input_combined/filtered_matrices_mex/hg19/barcodes.tsv", delimiter = "\t", header = None)
labels = barcodes.iloc[:,1]





# %%  Cut back data for handlability lmao

print(datetime.now().strftime("%H:%M:%S>"), "deleting random data pieces...")
num_cells = 5555
num_genes = 6666


### this block gets the indices of the num_genes highest rowcounts
rowsums = np.array(coomatrix.sum(axis = 1))
rowsums = rowsums.reshape(-1)
indices = np.argpartition(rowsums, -num_genes)[-num_genes:]


### this block gets evenly spaced indices from the array
linspacing = np.linspace(0, coomatrix.shape[1]-1, num_cells, dtype = int)


labels = labels[linspacing]
genes = genes[indices]



csrmatrix = coomatrix.tocsr()
csrmatrix = csrmatrix[indices,:]
coomatrix = csrmatrix[:,linspacing]

print(coomatrix.shape)





# %%

# Convert to dense
print(datetime.now().strftime("%H:%M:%S>"), "converting sparse matrix to dense...")
data = coomatrix.toarray()


# bring in the pandas
panda = pd.DataFrame(data, index = genes, columns = labels)


# %%

panda.to_csv("../../1. Eraslan/dca_test/countmatrix.tsv", sep = "\t")









