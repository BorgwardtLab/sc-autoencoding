# sc-autoencoding
student internship of Simon Streib to reduce single cell data



Input Data: 
contains public single cell transcriptome data downloaded from https://support.10xgenomics.com/single-cell-gene-expression/datasets (Cellranger 1.1.0)
All the data is uploaded into the group share /links/groups/borgwardt/Projects/simon_streib_internship/sc-autoencoding/



SCA_datamerger:
combineds the raw input matrices of different celltypes
--input_dir: Default: "../inputs/raw_input"
--output_dir:  Default: ./input 
--mode: compressed, decompressed, both: Default: decompressed
Input and output data is a sparse matrix of nxm, with n = #genes and m = #cells. 


Baseline Scripts:
Inputs (= the output of SCA_datamerger.py) are sparse matrices of nxm, with n = #genes and m = #cells.
sca_PCA.py: 






sca_PCA_local.py: A "downsampled" version, which takes the data, but only a chunk of it, meaning even my local computer can run it.











