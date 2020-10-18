# conda activate tf

mkdir logs
filename=logs/log_run_autoencoder_BCA.log

rm $filename

source /home/sstreib/anaconda3/etc/profile.d/conda.sh |& tee -a $filename
conda activate tf |& tee -a $filename



### 0.) Preprocessing
# preprocessing is commented out, as it is assumed, that the previous script will already do this. If this is run "alone", please uncomment both lines.
# python ../1_Processing/sca_datamerger.py --mode both --input_dir "../inputs/data/raw_input" --output_dir "../inputs/data/raw_input_combined" |& tee -a $filename
# python ../1_Processing/sca_countdata_preprocessor.py --mingenes 200 --mincells 1 --maxfeatures 1500 --maxmito 5 --features 2000 --test_fraction 0.25 --input_dir "../inputs/data/raw_input_combined/filtered_matrices_mex/hg19/" --output_dir "../inputs/sca/sca_preprocessed_data/" --verbosity 0 |& tee -a $filename


### 1.) Run the BCA
python ../3_Autoencoder/bca_autoencoder.py --loss poisson --activation relu --optimizer Adam --input_dir "../inputs/sca/sca_preprocessed_data/" --output_dir "../inputs/sca/BCA_output/" |& tee -a $filename


### 2.1)Evaluate the baselines with Kmeans clustering
python ../4_Evaluation/sca_kmcluster.py --reset --title "BCAutoencoder" --k 5 --dimensions 0 --verbosity 0 --input_dir "../inputs/sca/BCA_output/" --output_dir "../outputs/sca/kmcluster/" --outputplot_dir "../outputs/sca/kmcluster/PCA/" |& tee -a $filename


### 2.2) Evaluate the baselines with classification
for classifier in logreg lda forest
do
python ../4_Evaluation/sca_classification.py --reset --title "BCAutoencoder" --kfold 5 --classifier $classifier --input_dir "../inputs/sca/BCA_output/" --output_dir "../outputs/sca/ova_classification/" |& tee -a $filename
done


### 2.3) evaluate with dbscan
eps=30
minpts=5
python ../4_Evaluation/sca_dbscan.py --reset --title "BCAutoencoder" --verbosity 3 --eps $eps --min_samples $minpts --input_dir "../inputs/sca/BCA_output/" --output_dir "../outputs/sca/dbscan/" --outputplot_dir "../outputs/sca/dbscan/" |& tee -a $filename


### 2.4) evaluate with randomforest multiclass
ntrees=100
# maxdepth=None; minsamplesplit=2; minsamplesleaf=1; maxfeatures="auto"
python ../4_Evaluation/sca_randforest.py --reset --title "BCAutoencoder" --n_trees $ntrees --input_dir "../inputs/sca/BCA_output/" --output_dir "../outputs/sca/random_forest/" --outputplot_dir "../outputs/sca/random_forest/" |& tee -a $filename




echo "I have finished running the BCA autoencoder"





