



rm log_run_DBScan_evaluation.log


#for epsearch
python ../5_Optimization/dbscan_epssearch_knn.py --input_dir "../inputs/data/preprocessed_data/" --outputplot_dir "../outputs/optimization/dbscan_epsearch/" |& tee -a log_run_DBScan_evaluation.log
date
echo "finished the epsearch"


#gridsearch
#python ../5_Optimization/dbscan_gridsearch_pca.py --eps 10 12.5 15 17.5 20 25 30 35 40 45 50 --minpts 1 2 3 4 5 6 7 8 9 10 15 20 30 50 100 --input_dir "../inputs/baselines/baseline_data/scaPCA_output/" --output_dir "../outputs/optimization/dbscan_gridsearch/" |& tee -a log_run_DBScan_evaluation.log
python ../5_Optimization/dbscan_gridsearch_pca.py --eps 10 15 20 35 50 --minpts 1 4 10 100 --input_dir "../inputs/baselines/baseline_data/scaPCA_output/" --output_dir "../outputs/optimization/dbscan_gridsearch/" |& tee -a log_run_DBScan_evaluation.log

date
echo "finished the gridsearch"



echo "I have finished running the DBScan"





