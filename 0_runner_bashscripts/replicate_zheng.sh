


#python ../1_Processing/sca_datamerger.py --mode both --input_dir "../inputs/replication_zheng/original/" --output_dir "../inputs/replication_zheng/combined/"


python ../9_toyscripts/rp_barcodes_add_meaningless_column.py --file "../inputs/replication_zheng/original/filtered_matrices_mex/hg19/barcodes_original.tsv" --outfile "../inputs/replication_zheng/original/filtered_matrices_mex/hg19/barcodes.tsv"

python ../1_Processing/sca_countdata_preprocessor.py --n_splits 3 --mingenes 200 --mincells 1 --maxfeatures 1500 --maxmito 5 --features 2000 --test_fraction 0.25 --input_dir "../inputs/replication_zheng/original/filtered_matrices_mex/hg19/" --output_dir "../inputs/replication_zheng/preprocessed/" --verbosity 0





