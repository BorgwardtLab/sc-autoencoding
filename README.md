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
passing -s / --nosave prevents the program from saving the reduced coordinates. (plots and other output still gets saved)

about n_components:
sca_PCA.py: 	calculating with all PC's fails for memory issues. It is therefore necessary to cut the components somewhere after 10000. Default = 100
sca_LSA.py: 	default = 100, calculating all is possible. (use n_features - 1). Also consider setting the --nosave option. 
sca_Isomap.py:	default = 100. I didn't try more, but should be possible. 
sca_tSNE:	default = 2. Script runs on the standard barnes hut algorithm. As such, the number of components is cut to maximal 3 by default. Using more calls the algorithm with the exact method, which takes longer (and hasn't been tested). (but calling more is possible)
sac_UMAP: 	i've defaulted the n_components to 2. However, you can use as many as you want. 





toyscripts










