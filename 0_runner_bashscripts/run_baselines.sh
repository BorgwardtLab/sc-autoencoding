
mkdir logs
filename=./logs/log_run_baselines.log

rm $filename

source /home/sstreib/anaconda3/etc/profile.d/conda.sh
conda activate tf



python ../1_Processing/sca_datamerger.py --mode both --input_dir "../inputs/data/raw_input" --output_dir "../inputs/data/raw_input_combined" |& tee -a $filename


python ../1_Processing/sca_preprocessor.py --mingenes 200 --mincells 1 --maxfeatures 1500 --maxmito 5 --features 2000 --test_fraction 0.25 --input_dir "../inputs/data/raw_input_combined/filtered_matrices_mex/hg19/" --output_dir "../inputs/data/preprocessed_data/" --outputplot_dir "../outputs/preprocessing/preprocessed_data/" --verbosity 0 |& tee -a $filename

#Rscript 1_Main_Scripts/sca_preprocessing.R ../inputs/raw_input_combined/filtered_matrices_mex/hg19/ ../inputs/preprocessed_data/ ../outputs/preprocessed_data/ 200 1750 5 2000 |& tee -a log_run_baselines.log

python ../2_Baseline_Scripts/sca_PCA.py --num_components 50 --input_dir "../inputs/data/preprocessed_data/" --output_dir "../inputs/baselines/baseline_data/scaPCA_output/" --outputplot_dir "../outputs/baselines/baseline_data/scaPCA_output/" |& tee -a $filename
python ../2_Baseline_Scripts/sca_LSA.py --num_components 50 --input_dir "../inputs/data/preprocessed_data/" --output_dir "../inputs/baselines/baseline_data/scaLSA_output/" --outputplot_dir "../outputs/baselines/baseline_data/scaLSA_output/" |& tee -a $filename
# python ../2_Baseline_Scripts/sca_Isomap.py --num_components 50 --input_dir "../inputs/data/preprocessed_data/" --output_dir "../inputs/baselines/baseline_data/scaIsomap_output/" --outputplot_dir "../outputs/baselines/baseline_data/scaIsomap_output/" |& tee -a $filename
python ../2_Baseline_Scripts/sca_ICA.py --num_components 2 --input_dir "../inputs/baselines/baseline_data/scaPCA_output/" --output_dir "../inputs/baselines/baseline_data/scaICA_output/" --outputplot_dir "../outputs/baselines/baseline_data/scaICA_output/" |& tee -a $filename
python ../2_Baseline_Scripts/sca_tSNE.py --num_components 2 --verbosity 0 --input_dir "../inputs/baselines/baseline_data/scaPCA_output/" --output_dir "../inputs/baselines/baseline_data/scaTSNE_output/" --outputplot_dir "../outputs/baselines/baseline_data/scaTSNE_output/" |& tee -a $filename
python ../2_Baseline_Scripts/sca_UMAP.py --num_components 2 --verbosity 0 --input_dir "../inputs/baselines/baseline_data/scaPCA_output/" --output_dir "../inputs/baselines/baseline_data/scaUMAP_output/" --outputplot_dir "../outputs/baselines/baseline_data/scaUMAP_output/" |& tee -a $filename



### Evaluate the baselines with Kmeans clustering
python ../4_Evaluation/sca_kmcluster.py --reset --title "PCA" --k 5 --dimensions 0 --verbosity 0 --input_dir "../inputs/baselines/baseline_data/scaPCA_output/" --output_dir "../outputs/baselines/kmcluster/" --outputplot_dir "../outputs/baselines/kmcluster/PCA/" |& tee -a $filename
python ../4_Evaluation/sca_kmcluster.py --title "ICA" --k 5 --dimensions 0 --verbosity 0 --input_dir "../inputs/baselines/baseline_data/scaICA_output/" --output_dir "../outputs/baselines/kmcluster/" --outputplot_dir "../outputs/baselines/kmcluster/ICA/" |& tee -a $filename
python ../4_Evaluation/sca_kmcluster.py --title "LSA" --k 5 --dimensions 0 --verbosity 0 --input_dir "../inputs/baselines/baseline_data/scaLSA_output/" --output_dir "../outputs/baselines/kmcluster/" --outputplot_dir "../outputs/baselines/kmcluster/LSA/" |& tee -a $filename
# python ../4_Evaluation/sca_kmcluster.py --title "Isomap" --k 5 --dimensions 0 --verbosity 0 --input_dir "../inputs/baselines/baseline_data/scaIsomap_output/" --output_dir "../outputs/baselines/kmcluster/" --outputplot_dir "../outputs/baselines/kmcluster/Isomap/" |& tee -a $filename
python ../4_Evaluation/sca_kmcluster.py --title "t-SNE" --k 5 --dimensions 0 --verbosity 0 --input_dir "../inputs/baselines/baseline_data/scaTSNE_output/" --output_dir "../outputs/baselines/kmcluster/" --outputplot_dir "../outputs/baselines/kmcluster/tSNE/" |& tee -a $filename
python ../4_Evaluation/sca_kmcluster.py --title "UMAP" --k 5 --dimensions 0 --verbosity 0 --input_dir "../inputs/baselines/baseline_data/scaUMAP_output/" --output_dir "../outputs/baselines/kmcluster/" --outputplot_dir "../outputs/baselines/kmcluster/UMAP/" |& tee -a $filename
python ../4_Evaluation/sca_kmcluster.py --title "original_data" --k 5 --dimensions 0 --verbosity 0 --input_dir "../inputs/data/preprocessed_data/" --output_dir "../outputs/baselines/kmcluster/" --outputplot_dir "../outputs/baselines/kmcluster/original_data/" |& tee -a $filename


### Evaluate the baselines with classification


for classifier in logreg lda forest
do

echo python ../4_Evaluation/sca_classification.py --reset --title "PCA" --kfold 5 --classifier $classifier --input_dir "../inputs/baselines/baseline_data/scaPCA_output/" --output_dir "../outputs/baselines/ova_classification/" |& tee -a $filename
python ../4_Evaluation/sca_classification.py --reset --title "PCA" --kfold 5 --classifier $classifier --input_dir "../inputs/baselines/baseline_data/scaPCA_output/" --output_dir "../outputs/baselines/ova_classification/" |& tee -a $filename

echo python ../4_Evaluation/sca_classification.py --title "ICA" --kfold 5 --classifier $classifier --input_dir "../inputs/baselines/baseline_data/scaICA_output/" --output_dir "../outputs/baselines/ova_classification/" |& tee -a $filename
python ../4_Evaluation/sca_classification.py --title "ICA" --kfold 5 --classifier $classifier --input_dir "../inputs/baselines/baseline_data/scaICA_output/" --output_dir "../outputs/baselines/ova_classification/" |& tee -a $filename

echo python ../4_Evaluation/sca_classification.py --title "LSA" --kfold 5 --classifier $classifier --input_dir "../inputs/baselines/baseline_data/scaLSA_output/" --output_dir "../outputs/baselines/ova_classification/" |& tee -a $filename
python ../4_Evaluation/sca_classification.py --title "LSA" --kfold 5 --classifier $classifier --input_dir "../inputs/baselines/baseline_data/scaLSA_output/" --output_dir "../outputs/baselines/ova_classification/" |& tee -a $filename

# python ../4_Evaluation/sca_classification.py --title "Isomap" --kfold 5 --classifier $classifier --input_dir "../inputs/baselines/baseline_data/scaIsomap_output/" --output_dir "../outputs/baselines/ova_classification/" |& tee -a $filename

echo python ../4_Evaluation/sca_classification.py --title "t-SNE" --kfold 5 --classifier $classifier --input_dir "../inputs/baselines/baseline_data/scaTSNE_output/" --output_dir "../outputs/baselines/ova_classification/" |& tee -a $filename
python ../4_Evaluation/sca_classification.py --title "t-SNE" --kfold 5 --classifier $classifier --input_dir "../inputs/baselines/baseline_data/scaTSNE_output/" --output_dir "../outputs/baselines/ova_classification/" |& tee -a $filename

echo python ../4_Evaluation/sca_classification.py --title "UMAP" --kfold 5 --classifier $classifier --input_dir "../inputs/baselines/baseline_data/scaUMAP_output/" --output_dir "../outputs/baselines/ova_classification/" |& tee -a $filename
python ../4_Evaluation/sca_classification.py --title "UMAP" --kfold 5 --classifier $classifier --input_dir "../inputs/baselines/baseline_data/scaUMAP_output/" --output_dir "../outputs/baselines/ova_classification/" |& tee -a $filename

echo python ../4_Evaluation/sca_classification.py --title "original_data" --kfold 5 --classifier $classifier --input_dir "../inputs/data/preprocessed_data/" --output_dir "../outputs/baselines/ova_classification/" |& tee -a $filename
python ../4_Evaluation/sca_classification.py --title "original_data" --kfold 5 --classifier $classifier --input_dir "../inputs/data/preprocessed_data/" --output_dir "../outputs/baselines/ova_classification/" |& tee -a $filename

done




### evaluate with dbscan
eps=30
minpts=5

python ../4_Evaluation/sca_dbscan.py --reset --title "PCA" --verbosity 3 --eps $eps --min_samples $minpts --input_dir "../inputs/baselines/baseline_data/scaPCA_output/" --output_dir "../outputs/baselines/dbscan/" --outputplot_dir "../outputs/baselines/dbscan/scaPCA_output/" |& tee -a $filename
python ../4_Evaluation/sca_dbscan.py  --title "ICA" --verbosity 3 --eps $eps --min_samples $minpts --input_dir "../inputs/baselines/baseline_data/scaICA_output/" --output_dir "../outputs/baselines/dbscan/" --outputplot_dir "../outputs/baselines/dbscan/scaICA_output/" |& tee -a $filename
python ../4_Evaluation/sca_dbscan.py  --title "LSA" --verbosity 3 --eps $eps --min_samples $minpts --input_dir "../inputs/baselines/baseline_data/scaLSA_output/" --output_dir "../outputs/baselines/dbscan/" --outputplot_dir "../outputs/baselines/dbscan/scaLSA_output/" |& tee -a $filename
python ../4_Evaluation/sca_dbscan.py  --title "t-SNE" --verbosity 3 --eps $eps --min_samples $minpts --input_dir "../inputs/baselines/baseline_data/scaTSNE_output/" --output_dir "../outputs/baselines/dbscan/" --outputplot_dir "../outputs/baselines/dbscan/scaTSNE_output/" |& tee -a $filename
python ../4_Evaluation/sca_dbscan.py  --title "UMAP" --verbosity 3 --eps $eps --min_samples $minpts --input_dir "../inputs/baselines/baseline_data/scaUMAP_output/" --output_dir "../outputs/baselines/dbscan/" --outputplot_dir "../outputs/baselines/dbscan/scaUMAP_output/" |& tee -a $filename
python ../4_Evaluation/sca_dbscan.py  --title "original_data" --verbosity 3 --eps $eps --min_samples $minpts --input_dir "../inputs/data/preprocessed_data/" --output_dir "../outputs/baselines/dbscan/" --outputplot_dir "../outputs/baselines/dbscan/preprocessed_data/" |& tee -a $filename





### evaluate with randomforest multiclass
ntrees=100
# maxdepth=None; minsamplesplit=2; minsamplesleaf=1; maxfeatures="auto"
python ../4_Evaluation/sca_randforest.py --reset --title "PCA" --n_trees $ntrees --input_dir "../inputs/baselines/baseline_data/scaPCA_output/" --output_dir "../outputs/baselines/random_forest/" --outputplot_dir "../outputs/baselines/random_forest/scaPCA_output/" |& tee -a $filename
python ../4_Evaluation/sca_randforest.py  --title "ICA" --n_trees $ntrees --input_dir "../inputs/baselines/baseline_data/scaICA_output/" --output_dir "../outputs/baselines/random_forest/" --outputplot_dir "../outputs/baselines/random_forest/scaICA_output/" |& tee -a $filename
python ../4_Evaluation/sca_randforest.py  --title "LSA" --n_trees $ntrees --input_dir "../inputs/baselines/baseline_data/scaLSA_output/" --output_dir "../outputs/baselines/random_forest/" --outputplot_dir "../outputs/baselines/random_forest/scaLSA_output/" |& tee -a $filename
python ../4_Evaluation/sca_randforest.py  --title "t-SNE" --n_trees $ntrees --input_dir "../inputs/baselines/baseline_data/scaTSNE_output/" --output_dir "../outputs/baselines/random_forest/" --outputplot_dir "../outputs/baselines/random_forest/scaTSNE_output/" |& tee -a $filename
python ../4_Evaluation/sca_randforest.py  --title "UMAP" --n_trees $ntrees --input_dir "../inputs/baselines/baseline_data/scaUMAP_output/" --output_dir "../outputs/baselines/random_forest/" --outputplot_dir "../outputs/baselines/random_forest/scaUMAP_output/" |& tee -a $filename
python ../4_Evaluation/sca_randforest.py  --title "original_data" --n_trees $ntrees --input_dir "../inputs/data/preprocessed_data/" --output_dir "../outputs/baselines/random_forest/" --outputplot_dir "../outputs/baselines/random_forest/preprocessed_data/" |& tee -a $filename






echo "I have finished running all baselines"





