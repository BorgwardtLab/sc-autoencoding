# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 16:13:54 2020

@author: Mike Toreno II
"""


import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import sys
import argparse
import pandas as pd


try:
    os.chdir(os.path.dirname(sys.argv[0]))
except:
    pass


parser = argparse.ArgumentParser(description = "analyses the input data")  #required
parser.add_argument("-i","--input_dir", help="input directory", default = "../inputs/data/raw_input_combined/filtered_matrices_mex/hg19/")
parser.add_argument("-p","--outputplot_dir", help="plot directory", default = "../outputs/inspect_data")
args = parser.parse_args() #required






input_path = args.input_dir
outputplot_dir = args.outputplot_dir



input_path = "M:/Projects/simon_streib_internship/sc-autoencoding/inputs/data/raw_input_combined/filtered_matrices_mex/hg19/"


# %% Read Input data

print(datetime.now().strftime("%H:%M:%S>"), "reading input matrix...")
### Get Matrix
mtx_file = input_path + "matrix.mtx"
coomatrix = scipy.io.mmread(mtx_file)
data = np.transpose(coomatrix)


# ### Get Labels
# print(datetime.now().strftime("%H:%M:%S>"), "reading labels...")
# lbl_file = input_path + "celltype_labels.tsv"
# file = open(lbl_file, "r")
# labels = file.read().split("\n")
# file.close()
# labels.remove("") #last, empty line is also removed


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



# %% Count data


if not os.path.exists(outputplot_dir):
    print(datetime.now().strftime("%H:%M:%S>"), "Creating Output Directory...")
    os.makedirs(outputplot_dir)




bin1 = data.getnnz(axis = None)
bin_genes = data.getnnz(axis = 0) # stores the number of cells in which the gene was detected for each gene
bin_cells = data.getnnz(axis = 1) # stores the number of genes detected in each cell


# oh boy lets run this loop
rowsums = np.sum(data, axis = 0)
colsums = np.sum(data, axis = 0)












plt.figure()
plt.title("Histogram: How many cells express each gene?\n (Total: " + str(len(bin_genes) - np.count_nonzero(bin_genes)) + " zero-genes)")
plt.hist(bin_genes, log = True, bins = 100)
plt.ylabel("log(Frequency)")
plt.xlabel("Number of cells cells a gene is expressed by")
plt.show()
plt.savefig(outputplot_dir + "/genesplot.png")
print("Histogram: How many cells express each gene?, and it has the lenght of {:d}".format(len(bin_genes)))




plt.figure()
plt.title("Histogram: How many genes  were detected per cell\n (Total: " + str(len(bin_cells) - np.count_nonzero(bin_cells)) + " zero-cells)")
plt.hist(bin_cells, log = True, bins = 100)
plt.ylabel("log(Frequency)")
plt.xlabel("Number of genes detected per cell")
plt.show()
plt.savefig(outputplot_dir + "/cellplot.png")
print("Histogram: How many genes  were detected per cell, and it has the lenght of {:d}".format(len(bin_cells)))



# plt.figure()
# plt.title("Histogram: How many cells express each gene?\n (Total: " + str(len(bin_genes) - np.count_nonzero(bin_genes)) + " zero-genes)")
# plt.hist(bin_genes, log = True, bins = 100, density = True, label = "Log", alpha = 0.5)
# plt.hist(bin_genes, log = False, bins = 100, density = True, label = "normal"alpha = 0.5)
# plt.ylabel("Frequency")
# plt.xlabel("Number of cells cells a gene is expressed by")
# plt.legend()
# plt.show()
# plt.savefig(outputplot_dir + "/genesplot.png")


# plt.figure()
# plt.title("Histogram: How many genes  were detected per cell\n (Total: " + str(len(bin_cells) - np.count_nonzero(bin_cells)) + " zero-cells)")
# plt.hist(bin_cells, log = True, bins = 100, density = True, label = "Log")
# plt.hist(bin_cells, log = False, bins = 100, density = True, label = "normal")
# plt.ylabel("log(Frequency)")
# plt.xlabel("Number of genes detected per cell")
# plt.legend()
# plt.show()
# plt.savefig(outputplot_dir + "/cellplot.png")

plt.figure()
plt.title("Histogram: How many total transcripts read of a gene")
plt.hist(rowsums, log = True, bins = 100)
plt.ylabel("log(Frequency)")
plt.xlabel("Number of cells cells a gene is expressed by")
plt.show()
plt.savefig(outputplot_dir + "/tot_genesplot.png")
plt.figure()
print("the number of total transcripts is the rowsums, and it has the lenght of {:d}".format(len(rowsums)))




plt.figure()
plt.title("Histogram: How many total transcripts read per cell")
plt.hist(colsums, log = True, bins = 100)
plt.ylabel("log(Frequency)")
plt.xlabel("Number of genes detected per cell")
plt.show()
plt.savefig(outputplot_dir + "/tot_cellplot.png")
print("the number of total transcripts read per cell is the colsums, and it has the lenght of {:d}".format(len(colsums)))






plt.figure()
plt.title("Histogram: How many total transcripts read of a gene")
plt.hist(rowsums, log = False, bins = 100)
plt.ylabel("log(Frequency)")
plt.xlabel("Number of cells cells a gene is expressed by")
plt.show()
plt.savefig(outputplot_dir + "/tot_genesplot.png")
plt.figure()


plt.figure()
plt.title("Histogram: How many total transcripts read per cell")
plt.hist(colsums, log = False, bins = 100)
plt.ylabel("log(Frequency)")
plt.xlabel("Number of genes detected per cell")
plt.show()
plt.savefig(outputplot_dir + "/tot_cellplot.png")




# %%

print(data.shape)



# %% reread barcodes in order to count count per celltypes lmao


barcodes2 = pd.read_csv(input_path + "barcodes.tsv", delimiter = "\t", header = None)
celltype_counts = np.array(barcodes2.iloc[:,1])
unique, counts = np.unique(celltype_counts, return_counts=True)

for i in range(len(unique)):
    print(unique[i])
    print(counts[i])
    print()
    
