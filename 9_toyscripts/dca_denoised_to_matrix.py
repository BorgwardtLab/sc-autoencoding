# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 16:58:38 2020

@author: Mike Toreno II
"""


import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser(description = "program to preprocess the raw singlecell data")  #required
parser.add_argument("--full_input_file", help="input directory", default = "M:/Projects/simon_streib_internship/sc-autoencoding/inputs/autoencoder_data/DCA_output/no_split/mean.tsv")
parser.add_argument("--full_output_file", help="input directory", default = "M:/Projects/simon_streib_internship/sc-autoencoding/inputs/autoencoder_data/DCA_output/denoised_reconstruction/matrix.txv")
args = parser.parse_args() #required



print("starting denoised to matrix")

pandas = pd.read_csv(args.full_input_file, delimiter = "\t", header = 0, index_col = 0)
# pandas = pandas.transpose()

print("if it worked correctly, the following output is not a number")
print(pandas.columns[5])
print(pandas.index[1])
print("but the following is a number")

print(pandas.iloc[0,0])

print("shape")
print(pandas.shape)




# shorten outputfilename to only dir, so I can put it into makedir.
# this is kinda cumbersome, but very intuitive
dir = args.full_output_file
while dir[-1] != "/":
    dir = dir[:-1]
os.makedirs(dir, exist_ok=True)




pandas.to_csv(path_or_buf = args.full_output_file, sep = "\t", header = False, index = False, float_format='%.6f')


print("dca_output_to_matrix.py successfully run")




