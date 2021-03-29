# sc-autoencoding
student internship of Simon Streib to reduce single cell data

The goal was to use an autoencoder to reduce the dimensionality or sincle-cell transcriptome data, and then compare the results with baselines, such as PCA, ICA, LSA, t-SNE, UMAP. Different autoencoders were used, called BCA (Basic Count Autoencoder), DCA (Denoising Count Autoencoder, an autoencoder created by Zheng et. al.), and SCA (Simon's Count Autoencoder). 

The complete pipeline can be run by cloning the repository, downloading input data as explained below, and then run script "0_runner_bashscripts\analyse_all.sh" which will call all other scripts for preprocessing, dimensionality reduction and evaluation. Note, that a very long runtime (~days on a 100-core processor) will be expected for the whole project. Depending on the amount of available computing power, the amount of parallel computations can be edited within the .sh files inside \cloned_dirctory\0_runner_bashscripts by allowing loops to run in parallel using "&". 

Requirements of installed python packages has been supplied. Plese note, that the pipeline running Zheng's Denoising Count Autoncoder (DCA) has to be run in a different environment. It is recommended to set up environments identically to the original setup: The main conda environment, called "tf" and using the packages of "requirements_tf.txt", and the Zheng environment called "dicia2" using the packages listed in "requirements_dicia2.txt". Also note, that in order to activate conda environments from scripts, the conda.sh needs to be sourced. By default, the conda.sh is expected to be located under ~/anaconda3/etc/profile.d/conda.sh.
If any changes are made to this, then the environment handling of the .sh scripts in \cloned_dirctory\0_runner_bashscripts needs to be adjusted at the lines looking like this:

source ~/anaconda3/etc/profile.d/conda.sh   # source your own conda.sh 

conda activate dicia2                       # edit names to your own environments


Input Data:
Was downloaded from https://support.10xgenomics.com/single-cell-gene-expression/datasets (Cellranger 1.1.0)
Downloaded were 10 datasets from Zheng. et. al, containing Data to 10 different celltypes. The data should be put to cloned_dirctory\inputs\data\raw_input\cd4_t_helper_filtered_gene_bc_matrices.tar.gz", with the internal structure of .tar.gz files corresponding to the rules of 10x Genomics Example Datasets. Any number of .tar.gz files can be deposited in the directory as needed. Note however, that the data of each file will be regarded as one "celltype", with the name taken from the filename. If the filenames have different structures, then edits might be necessary in cloned_dirctory\1_Processing\sca_datamerger.py" to ensure proper labelling of the celltype.

More information about the results and structure of the project can be found in the uploaded file "Report_RP1".
