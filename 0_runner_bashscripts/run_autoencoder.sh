# conda activate tf

rm log_run_autoencoder



# python ../1_Main_Scripts/sca_datamerger.py --mode both --input_dir "../inputs/raw_input" --output_dir "../inputs/raw_input_combined" |& tee -a log_run_autoencoder


python ../4_Autoencoder/sca_countdata_preprocessor.py --mingenes 200 --mincells 1 --maxfeatures 1500 --maxmito 5 --features 2000 --input_dir "../inputs/raw_input_combined/filtered_matrices_mex/hg19/" --output_dir "../inputs/sca/sca_preprocessed_data" --verbosity 0 |& tee -a log_run_autoencoder



python ../4_Autoencoder/sca_autoencoder.py --input_dir "../inputs/sca/sca_preprocessed_data/" --output_dir "../inputs/sca/autoencoder_output/" |& tee -a log_run_autoencoder


### Evaluate the baselines with Kmeans clustering
python ../1_Main_Scripts/sca_kmcluster.py --reset --title "SCAutoencoder" --k 5 --dimensions 0 --verbosity 0 --input_dir "../inputs/sca/autoencoder_output/" --output_dir "../outputs/sca/kmcluster/" --outputplot_dir "../outputs/sca/kmcluster/PCA/" |& tee -a log_run_autoencoder


### Evaluate the baselines with classification
python ../1_Main_Scripts/sca_classification.py --reset --title "SCAutoencoder" --kfold 5 --classifier "logreg" --input_dir "../inputs/sca/autoencoder_output/" --output_dir "../outputs/sca/ova_classification/" |& tee -a log_run_autoencoder



echo "I have finished running the autoencoder"





