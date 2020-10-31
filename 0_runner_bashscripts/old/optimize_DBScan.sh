
mkdir logs
filename=logs/log_optimize_DBScan_gridsearch.log

rm $filename



#for epsearch
python ../5_Optimization/dbscan_epssearch_knn.py --input_dir "../inputs/data/preprocessed_data/" --outputplot_dir "../outputs/optimization/dbscan_epsearch/" |& tee -a $filename
date
echo "finished the epsearch"


#gridsearch

python ../5_Optimization/dbscan_gridsearch_pca.py --eps 10 15 17.5 20 22.5 25 27.5 30 32.5 35 40 45 50 --minpts 1 2 3 4 5 6 8 10 20 50 --input_dir "../inputs/baselines/baseline_data/scaPCA_output/" --output_dir "../outputs/optimization/dbscan_gridsearch/" |& tee -a $filename

date
echo "finished the gridsearch"



echo "I have finished running the DBScan"





