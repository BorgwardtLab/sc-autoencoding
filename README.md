# sc-autoencoding
student internship of Simon Streib to reduce single cell data
The goal of the project is to develop a neural network capable of reducing single cell data




Input Data: 
contains public single cell transcriptome data downloaded from https://support.10xgenomics.com/single-cell-gene-expression/datasets (Cellranger 1.1.0)


SCA_datamerger:
a script used to collect the transcriptome data of different cell types in a common matrix, loosing the cell type information. (cell type information is saved in a separate .tsv file). 
The program is buildt on: argparse, scipy, glob, tarfile, pandas, os, csv and sys. it takes the following optional arguments: 
--input_dir: directory which contains the different gzip compressed input data. Default: ./input_data
--output_dir: directory, which to which the combined matrices are to be saved. Default: ./input (as this will be the input data for the next script)
--mode: compressed, decompressed, both: Whether to save the combined data compressed (analog to the input_data) or decompressed. Default: Both










