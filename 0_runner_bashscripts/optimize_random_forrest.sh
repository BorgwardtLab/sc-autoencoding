
mkdir logs
filename=logs/log_optimize_random_forrest.log


rm $filename

# note: all parameters are taken at default value. There is no need to overwrite them here, they are already pretty good I think. 


python ../5_Optimization/random_forrest_binary_parametersearch.py --input_dir "../inputs/baselines/baseline_data/scaPCA_output/" --outputplot_dir "../outputs/optimization/random_forest/binary/" |& tee -a $filename
python ../5_Optimization/random_forrest_multiclass_parametersearch.py --input_dir "../inputs/baselines/baseline_data/scaPCA_output/" --outputplot_dir "../outputs/optimization/random_forest/binary/" |& tee -a $filename


echo "I have finished running the randomforrest evaluation"





