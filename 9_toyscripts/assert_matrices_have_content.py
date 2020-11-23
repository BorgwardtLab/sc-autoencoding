# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 18:07:08 2020

@author: Mike Toreno II
"""






import argparse
parser = argparse.ArgumentParser(description = "clustering data")  #required
parser.add_argument("-i","--input_dir", help="input directory", default = "M:/Projects/simon_streib_internship/sc-autoencoding/outputs/experiments/AEtypes/bca_data/nb-conddisp/")
args = parser.parse_args() #required

import pandas as pd

print(args.input_dir)

folders = ["no_split/", "split_1/", "split_2/", "split_3/"]

for folder in folders:
    directory = args.input_dir + folder + "matrix.tsv"
    data = pd.read_csv(directory, delimiter = "\t", header = None)
    are_numbers = True
    
    for i in range(20):
        sample = data.sample(1, axis = 1).sample(1, axis = 0)
        sample = sample.iloc[0,0]

        if sample != sample:
            are_numbers = False
            
    if are_numbers:
        print(folder + " is okay")
    else:
        print(folder + " is only NaNs")            



