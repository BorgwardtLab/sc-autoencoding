# conda activate tf


mkdir logs
filename=logs/log_run_autoencoder_DCA.log

rm $filename





# python ../1_Processing/sca_datamerger.py --mode both --input_dir "../inputs/data/raw_input" --output_dir "../inputs/data/raw_input_combined" |& tee -a $filename

# PREPROCESSING
python ../1_Processing/sca_countdata_preprocessor.py --mingenes 200 --mincells 1 --maxfeatures 1500 --maxmito 5 --features 2000 --test_fraction 0.25 --input_dir "../inputs/data/raw_input_combined/filtered_matrices_mex/hg19/" --output_dir "../inputs/sca/sca_preprocessed_data/" --verbosity 0 |& tee -a $filename



source /home/sstreib/anaconda3/etc/profile.d/conda.sh
conda activate dicia2

dca "../inputs/sca/sca_preprocessed_data/matrix_transposed.tsv" "../inputs/sca/DCA_output/" |& tee -a $filename
echo "DCA is done"




# bring the data into "my" format (no headers etc)
cp "../inputs/sca/sca_preprocessed_data/barcodes.tsv" "../inputs/sca/DCA_output/barcodes.tsv" |& tee -a $filename
cp "../inputs/sca/sca_preprocessed_data/genes.tsv" "../inputs/sca/DCA_output/genes.tsv" |& tee -a $filename
cp "../inputs/sca/sca_preprocessed_data/test_index.tsv" "../inputs/sca/DCA_output/test_index.tsv" |& tee -a $filename
python ../9_toyscripts/dca_output_to_matrix.py --input_dir "../inputs/sca/DCA_output/" |& tee -a $filename



### Evaluate the loss history
python ../9_toyscripts/plot_dca_loss.py --input_dir "../0_runner_bashscripts/" --outputplot_dir "../outputs/sca/dca/" |& tee -a $filename



### Evaluate the baselines with Kmeans clustering
python ../4_Evaluation/sca_kmcluster.py --reset --title "DCA" --k 5 --dimensions 0 --verbosity 0 --input_dir "../inputs/sca/DCA_output/" --output_dir "../outputs/sca/dca/kmcluster/" --outputplot_dir "../outputs/sca/dca/kmcluster/" |& tee -a $filename


### Evaluate the baselines with classification
for classifier in logreg lda forest
do

python ../4_Evaluation/sca_classification.py --reset --title "DCA" --kfold 5 --classifier $classifier --input_dir "../inputs/sca/DCA_output/" --output_dir "../outputs/sca/dca/ova_classification/" |& tee -a $filename

done



### evaluate with dbscan
eps=30
minpts=5
python ../4_Evaluation/sca_dbscan.py --reset --title "DCA" --verbosity 3 --eps $eps --min_samples $minpts --input_dir "../inputs/sca/DCA_output/" --output_dir "../outputs/sca/dca/dbscan/" --outputplot_dir "../outputs/sca/dca/dbscan/" |& tee -a $filename



### evaluate with randomforest multiclass
ntrees=100
# maxdepth=None; minsamplesplit=2; minsamplesleaf=1; maxfeatures="auto"
python ../4_Evaluation/sca_randforest.py --reset --title "DCA" --n_trees $ntrees --input_dir "../inputs/sca/DCA_output/" --output_dir "../outputs/sca/dca/random_forest/" --outputplot_dir "../outputs/sca/dca/random_forest/" |& tee -a $filename






echo "I have finished running DCA"





