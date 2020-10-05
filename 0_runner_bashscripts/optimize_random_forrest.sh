

filename=log_optimize_random_forrest.log


rm $filename



python ../5_Optimization/random_forrest_binary_parametersearch.py --input_dir "../inputs/baselines/baseline_data/scaPCA_output/" --outputplot_dir "../outputs/optimization/random_forrest/binary/" |& tee -a $filename



echo "I have finished running the randomforrest evaluation"





