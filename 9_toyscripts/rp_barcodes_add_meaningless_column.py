# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 16:41:26 2020

@author: Mike Toreno II
"""




import argparse

parser = argparse.ArgumentParser(description = "program to preprocess the raw singlecell data")  
parser.add_argument("--file", help="input directory", default = "../barcodes_original.tsv")
parser.add_argument("--outfile", help="input directory", default = "../barcodes.tsv")

args = parser.parse_args()





import pandas as pd


barcodes = pd.read_csv(args.file, 
                    delimiter = "\t",   # default None -> will read all as one column
                    header = None,         # which row will be header names, or None
                    index_col = None,      # which column should be row labels
                    )  


barcodes["extracolumn"] = "unknown"


barcodes.to_csv(args.file, sep = "\t", index = False, header = False)






