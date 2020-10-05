


filename=log_optimize_num_PCA.log


rm $filename



python ../5_Optimization/pca-kmcluster_evaluation.py --num_components 2 3 4 5 6 7 8 9 10 15 20 30 40 50 60 80 100 --num_kmclusters 5 --reps 100 --input_dir "../inputs/data/preprocessed_data/" --output_dir "../outputs/optimization/num_PCA/" |& tee -a $filename



echo "I have finished running the PCA_evaluation"





