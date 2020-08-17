# conda activate tf

rm log_run_autoencoder_DCA.log



# python ../1_Processing/sca_datamerger.py --mode both --input_dir "../inputs/data/raw_input" --output_dir "../inputs/data/raw_input_combined" |& tee -a log_run_baselines

# PREPROCESSING
python ../3_Autoencoder/sca_countdata_preprocessor.py --mingenes 200 --mincells 1 --maxfeatures 1500 --maxmito 5 --features 2000 --input_dir "../inputs/data/raw_input_combined/filtered_matrices_mex/hg19/" --output_dir "../inputs/sca/sca_preprocessed_data" --verbosity 0 |& tee -a log_run_autoencoder.log


dca "../inputs/sca/sca_preprocessed_data/matrix_transposed.tsv" "../inputs/sca/DCA_output/" |& tee -a log_run_autoencoder_DCA.log
cp "../inputs/sca/sca_preprocessed_data/barcodes.tsv" "../inputs/sca/DCA_output/barcodes.tsv"
cp "../inputs/sca/sca_preprocessed_data/genes.tsv" "../inputs/sca/DCA_output/genes.tsv"
python ../9_toyscripts/dca_output_to_matrix.py --input_dir "../inputs/sca/DCA_output/" |& tee -a log_run_autoencoder_DCA.log

### Evaluate the loss history
python ../9_toyscripts/plot_dca_loss.py --input_dir "../0_runner_bashscripts/" --outputplot_dir "../outputs/sca/dca/" |& tee -a log_run_autoencoder_DCA.log



### Evaluate the baselines with Kmeans clustering
python ../4_Evaluation/sca_kmcluster.py --reset --title "DCA" --k 5 --dimensions 0 --verbosity 0 --input_dir "../inputs/sca/DCA_output/" --output_dir "../outputs/sca/dca/kmcluster/" --outputplot_dir "../outputs/sca/dca/kmcluster/" |& tee -a log_run_autoencoder_DCA.log

### Evaluate the baselines with classification
python ../4_Evaluation/sca_classification.py --reset --title "DCA" --kfold 5 --classifier "logreg" --input_dir "../inputs/sca/DCA_output/" --output_dir "../outputs/sca/dca/ova_classification/" |& tee -a log_run_autoencoder_DCA.log



echo "I have finished running DCA"





