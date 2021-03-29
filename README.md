# sc-autoencoding
student internship of Simon Streib to reduce single cell data


The goal was to use an autoencoder to reduce the dimensionality or sincle-cell transcriptome data, and then compare the results with baselines, such as PCA, ICA, LSA, t-SNE, UMAP. Different autoencoders were used, called BCA (Basic Count Autoencoder), DCA (Denoising Count Autoencoder) by Zheng et. al., and SCA (Simon's Count Autoencoder). 

The complete pipeline can be run by cloning the repository, downloading input data as explained below, and then run script "0_runner_bashscripts\analyse_all.sh" which will call all other scripts for preprocessing, dimensionality reduction and evaluation.

Input Data:
Was downloaded from https://support.10xgenomics.com/single-cell-gene-expression/datasets (Cellranger 1.1.0)
Downloaded were 10 datasets from Zheng. et. al, containing Data to 10 different celltypes. The data should be put to <cloned repo>\inputs\data\raw_input\cd4_t_helper_filtered_gene_bc_matrices.tar.gz", with the internal structure of .tar.gz files corresponding to the rules of 10x Genomics Example Datasets. Any number of .tar.gz files can be deposited in the directory as needed. Note however, that the data of each file will be regarded as one "celltype", with the name taken from the filename. If the filenames have different names, then edits might be necessary in <cloned repo>\1_Processing\sca_datamerger.py"



More information aboout the results and structure of the project can be found in the uploaded file "Report_RP1".


