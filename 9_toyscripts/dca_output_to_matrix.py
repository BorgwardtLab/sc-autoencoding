# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 16:58:38 2020

@author: Mike Toreno II
"""


import pandas as pd
import argparse

parser = argparse.ArgumentParser(description = "program to preprocess the raw singlecell data")  #required
parser.add_argument("--input_dir", help="input directory", default = "../inputs/sca/DCA_output/")
args = parser.parse_args() #required



input_dir = args.input_dir


pandas = pd.read_csv(input_dir + "latent.tsv", delimiter = "\t", header = None, index_col = 0)
pandas.transpose()
pandas.to_csv(path_or_buf= input_dir + "matrix.tsv", sep = "\t", header = False, index = False, float_format='%.6f')





