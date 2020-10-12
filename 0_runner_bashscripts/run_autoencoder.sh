# conda activate tf

mkdir logs
filename=logs/log_run_autoencoder.log

rm $filename

source /home/sstreib/anaconda3/etc/profile.d/conda.sh
conda activate tf





# python ../1_Processing/sca_datamerger.py --mode both --input_dir "../inputs/data/raw_input" --output_dir "../inputs/data/raw_input_combined" |& tee -a $filename


python ../1_Processing/sca_countdata_preprocessor.py --mingenes 200 --mincells 1 --maxfeatures 1500 --maxmito 5 --features 2000 --test_fraction 0.25 --input_dir "../inputs/data/raw_input_combined/filtered_matrices_mex/hg19/" --output_dir "../inputs/sca/sca_preprocessed_data/" --verbosity 0 |& tee -a $filename



python ../3_Autoencoder/sca_autoencoder.py --loss poisson_loss --input_dir "../inputs/sca/sca_preprocessed_data/" --output_dir "../inputs/sca/autoencoder_output/" |& tee -a $filename



### Evaluate the baselines with Kmeans clustering
python ../4_Evaluation/sca_kmcluster.py --reset --title "SCAutoencoder" --k 5 --dimensions 0 --verbosity 0 --input_dir "../inputs/sca/autoencoder_output/test_data/" --output_dir "../outputs/sca/kmcluster/" --outputplot_dir "../outputs/sca/kmcluster/PCA/" |& tee -a $filename


### Evaluate the baselines with classification
for classifier in logreg lda forest
do
python ../4_Evaluation/sca_classification.py --reset --title "SCAutoencoder" --kfold 5 --classifier $classifier --input_dir "../inputs/sca/autoencoder_output/test_data/" --output_dir "../outputs/sca/ova_classification/" |& tee -a $filename
done




### evaluate with dbscan
eps=30
minpts=5
python ../4_Evaluation/sca_dbscan.py --reset --title "SCAutoencoder" --verbosity 3 --eps $eps --min_samples $minpts --input_dir "../inputs/sca/autoencoder_output/test_data/" --output_dir "../outputs/sca/dbscan/" --outputplot_dir "../outputs/sca/dbscan/" |& tee -a $filename



### evaluate with randomforest multiclass
ntrees=100
# maxdepth=None; minsamplesplit=2; minsamplesleaf=1; maxfeatures="auto"
python ../4_Evaluation/sca_randforest.py --reset --title "SCAutoencoder" --n_trees $ntrees --input_dir "../inputs/sca/autoencoder_output/test_data/" --output_dir "../outputs/sca/random_forest/" --outputplot_dir "../outputs/sca/random_forest/" |& tee -a $filename







echo "I have finished running the autoencoder"





