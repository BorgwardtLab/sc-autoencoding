

filename=optimize_random_forrest.log


rm $filename



python ../5_Optimization/random_forrest_gridsearch.py --input_dir "../inputs/baselines/baseline_data/scaPCA_output/" --outputplot_dir "../outputs/optimization/random_forrest/" |& tee -a $filename



echo "I have finished running the randomforrest evaluation"





