

mkdir logs
filename=logs/log_optimize_num_k_kmcluster.log


rm $filename

python ../4_Evaluation/sca_kmcluster.py --elbow --elbowrange 25 --input_dir "../inputs/baselines/baseline_data/scaPCA_output/" --output_dir "../outputs/optimization/num_kmcluster/scaPCA_output/" --outputplot_dir "../outputs/optimization/num_kmcluster/scaPCA_output/" |& tee -a $filename


echo "I have finished running the PCA_evaluation"





