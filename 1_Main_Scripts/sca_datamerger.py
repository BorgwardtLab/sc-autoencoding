# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 22:56:14 2020

@author: Simon Streib
"""
import argparse
import scipy.io
import glob
import tarfile
import pandas as pd
import os
import csv
import sys
from datetime import datetime



os.chdir(sys.path[0])



### General Setup / input
parser = argparse.ArgumentParser(description = "this program unpacks the tar \
files, extracts the infos and \
1.) concatenates all the counts together into: combined_matrix \
2.) concatenates all the barcodes together into a dataframe: barcodes \
3.) saves the gene list as well (once) \
4.) creates the object celltype_label, which is an array of the same length \
as combined_matrix, and contains the names of the original input file \
(repeated for each cell). Depending on mode, the four files are then saved \
or also compressed into a new .tar.gz file (combined_input.tar.gz), saved in \
the output directory. (per default ./input/")

parser.add_argument("-i", "--input_dir", help = "change the input directory", default = "../inputs/raw_input") 
parser.add_argument("-o", "--output_dir", help = "change the output directory", default = "../inputs/raw_input_combined")
parser.add_argument("-m", "--mode", help = "choose between compressed and decompressed", \
    choices = ["compressed", "decompressed", "both"], default = "decompressed")
args = parser.parse_args()



# Crash if input dir doesn't exist    
if not os.path.exists(args.input_dir):
    sys.exit("input directory doesn't exist")
    
# Create output directories if they don't exist
if not os.path.exists(args.output_dir):
    print(datetime.now().strftime("%H:%M:%S>"), "created output directory called " + args.output_dir)
    os.makedirs(args.output_dir)
    




# %% Combine Inputs
combined_matrix = None
combined_barcodes = pd.DataFrame()
celltype_label = []


for filepath in glob.iglob(args.input_dir + "/*.tar.gz"):
    print(datetime.now().strftime("%H:%M:%S>"), "unpacking " + filepath[20:] + "...")
    tarfile = tarfile.open(filepath, "r:gz")
    mtx_file = tarfile.extractfile("filtered_matrices_mex/hg19/matrix.mtx")
    
    current_label = filepath[20:filepath.find("_filtered_gene_bc_matrices.tar.gz")]
    current_matrix = scipy.io.mmread(mtx_file)
    combined_matrix = scipy.sparse.hstack((combined_matrix, current_matrix))
    
    
    # also export the gene files. (will be overwritten each round but whatev)
    genes_file = tarfile.extractfile("filtered_matrices_mex/hg19/genes.tsv")
    genes = pd.read_csv(genes_file, header = None, sep = "\t")
    
    # export the cell barcodes
    barcodes_file = tarfile.extractfile("filtered_matrices_mex/hg19/barcodes.tsv")
    barcodes = pd.read_csv(barcodes_file, header = None)  
    combined_barcodes = combined_barcodes.append(barcodes)
    
    tarfile.close()
    
    
    labels = current_matrix.shape[1] * [current_label] #fill the array with identical string
    celltype_label = celltype_label + labels

    
    
# %% Create output
# print(type(combined_matrix))
# print(type(combined_barcodes))
# print(type(genes))
# print(type(celltype_label))

# print((combined_matrix.shape))
# print((combined_barcodes.shape))
# print((genes.shape))
# print(len(celltype_label))



# check if directory exists
print(datetime.now().strftime("%H:%M:%S>"), "creating output...")

output_dir = args.output_dir + "/filtered_matrices_mex/hg19"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
    
        
# write the two panda dataframes 
genes.to_csv(output_dir + "/genes.tsv", sep = "\t", index = False, header = False)
combined_barcodes.to_csv(output_dir + "/barcodes.tsv", sep = "\t", index = False, header = False)

# write the array
handle = (open (output_dir + '/celltype_labels.tsv', 'w'))
wtr = csv.writer(handle, delimiter=';', lineterminator='\n')
for line in celltype_label: wtr.writerow ([line])
handle.close()

# write the .mtx
scipy.io.mmwrite(output_dir + "/matrix.mtx", combined_matrix)


# %% formalize for tar archive

if args.mode == "decompressed":
    print(datetime.now().strftime("%H:%M:%S>"), "script has terminated successfully")
    print(datetime.now().strftime("%H:%M:%S>"), "data is found in " + output_dir)
   
else:
    print(datetime.now().strftime("%H:%M:%S>"), "building archive...")
    with tarfile.open(args.output_dir + "/combined_cells_matrices.tar.gz", "w:gz") as tar:
        tar.add(output_dir + "/genes.tsv", arcname = "/filtered_matrices_mex/hg19/genes.tsv")    
        tar.add(output_dir + "/barcodes.tsv", arcname = "/filtered_matrices_mex/hg19/barcodes.tsv") 
        tar.add(output_dir + "/celltype_labels.tsv", arcname = "/filtered_matrices_mex/hg19/celltype_labels.tsv") 
        tar.add(output_dir + "/matrix.mtx", arcname = "/filtered_matrices_mex/hg19/matrix.mtx")   
        tar.close()
        
    print(datetime.now().strftime("%H:%M:%S>"), "archive has been successfully buildt")
    
    if args.mode == "compressed":
        print(datetime.now().strftime("%H:%M:%S>"), "removing decompressed files")
        os.remove(output_dir + "/genes.tsv")
        os.remove(output_dir + "/barcodes.tsv")
        os.remove(output_dir + "/matrix.mtx")
        os.remove(output_dir + '/celltype_labels.tsv')
        
        try:
            os.rmdir(args.output_dir + "/filtered_matrices_mex/hg19")
            os.rmdir(args.output_dir + "/filtered_matrices_mex")
        except:
            print(datetime.now().strftime("%H:%M:%S>"), "couldn't delete directory: directory not empty?")
             
        
        
    print("script has terminated successfully")
    print("data is found in " + args.output_dir)
    




