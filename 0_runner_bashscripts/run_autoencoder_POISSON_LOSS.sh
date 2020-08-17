# conda activate tf





# change name here
intermediate="poisson_loss"

# change to the correct autoencoder







######################################################################
#the rest is automatic



filename=log_run_autoencoder_$intermediate.log


rm $filename



ae_outdir="../inputs/sca/"$intermediate"/autoencoder_output/"

kmclust="../outputs/sca/"$intermediate"/kmcluster/"
ova="../outputs/sca/"$intermediate"/ova_classification/"





# python ../1_Processing/sca_datamerger.py --mode both --input_dir "../inputs/data/raw_input" --output_dir "../inputs/data/raw_input_combined" |& tee -a log_run_baselines


#python ../3_Autoencoder/sca_countdata_preprocessor.py --mingenes 200 --mincells 1 --maxfeatures 1500 --maxmito 5 --features 2000 --input_dir "../inputs/data/raw_input_combined/filtered_matrices_mex/hg19/" --output_dir "../inputs/sca/sca_preprocessed_data" --verbosity 0 |& tee -a log_run_autoencoder.log



python ../3_Autoencoder/sca_autoencoder.py --input_dir "../inputs/sca/sca_preprocessed_data/" --output_dir $ae_outdir |& tee -a $filename



### Evaluate the baselines with Kmeans clustering
python ../4_Evaluation/sca_kmcluster.py --reset --title "SCAutoencoder" --k 5 --dimensions 0 --verbosity 0 --input_dir $ae_outdir --output_dir $kmclust --outputplot_dir $kmclust |& tee -a $filename


### Evaluate the baselines with classification
python ../4_Evaluation/sca_classification.py --reset --title "SCAutoencoder" --kfold 5 --classifier "logreg" --input_dir $ae_outdir --output_dir $ova |& tee -a $filename



echo "I have finished running the autoencoder"





