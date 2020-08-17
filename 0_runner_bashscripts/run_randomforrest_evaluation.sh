



rm log_run_randomforrest_evaluation.log



python ../5_Optimization/random_forrest_gridsearch.py --input_dir "../inputs/baselines/baseline_data/scaPCA_output/" --outputplot_dir "../outputs/optimization/random_forrest/" |& tee -a log_run_randomforrest_evaluation.log



echo "I have finished running the randomforrest evaluation"





