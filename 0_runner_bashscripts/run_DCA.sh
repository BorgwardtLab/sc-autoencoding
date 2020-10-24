


source ~/anaconda3/etc/profile.d/conda.sh # to source the conda command. Check directory if it doesn't work.
conda activate dicia2
conda env list	# it should be visible in the log-textfile. I'm not saving it to anything else. 


mkdir logs
logfile=logs/3_autoencoder_DCA.log

dca "../inputs/data/preprocessed_data_autoencoder/no_split/matrix_transposed.tsv" "../inputs/autoencoder_data/DCA_output/" |& tee -a $logfile
echo "DCA is done" |& tee -a $logfile



# bring the data into "my" format (no headers etc)
cp "../inputs/data/preprocessed_data_autoencoder/no_split/barcodes.tsv" "../inputs/autoencoder_data/DCA_output/barcodes.tsv" |& tee -a $logfile
cp "../inputs/data/preprocessed_data_autoencoder/no_split/genes.tsv" "../inputs/autoencoder_data/DCA_output/genes.tsv" |& tee -a $logfile
# cp "../inputs/data/preprocessed_data_autoencoder/no_split/test_index.tsv" "../inputs/autoencoder_data/DCA_output/test_index.tsv" |& tee -a $logfile

python ../9_toyscripts/dca_output_to_matrix.py --input_dir "../inputs/autoencoder_data/DCA_output/" |& tee -a $logfile



### Evaluate the loss history
# not that the file loaded and the file appended are identical. But I think tee only ever writes after the process is done, so should probably be fine.
python ../9_toyscripts/plot_dca_loss.py --input_file "../0_runner_bashscripts/logs/3_autoencoder_DCA.log" --outputplot_dir "../outputs/autoencoders/DCA/" |& tee -a $logfile





# restore original env
source ~/anaconda3/etc/profile.d/conda.sh # to source the conda command. Check directory if it doesn't work.
conda activate tf
conda env list	# it should be visible in the log-textfile. I'm not saving it to anything else. 
