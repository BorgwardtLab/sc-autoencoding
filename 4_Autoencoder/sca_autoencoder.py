# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 16:14:45 2020

@author: Mike Toreno II
"""


import argparse
from sca_ae_classes import sca


parser = argparse.ArgumentParser(description = "program to preprocess the raw singlecell data") 
parser.add_argument("-i","--input_dir", help="input directory", default = "../inputs/raw_input_combined/filtered_matrices_mex/hg19/")
parser.add_argument("-o","--output_dir", help="output directory", default = "../inputs/dca/dca_preprocessed_data/")
parser.add_argument("-v","--verbosity", help="level of verbosity", default = 3, choices = [0, 1, 2, 3], type = int)
args = parser.parse_args()




def sca(input_dir, output_dir, verbosity):
    from datetime import datetime

    
    print(datetime.now().strftime("%H:%M:%S>"), "Starting sca_autoencoder.py...")








if __name__ == "__main__":
    sca(input_dir= args.input_dir, output_dir= args.output_dir, verbosity= args.verbosity)




































