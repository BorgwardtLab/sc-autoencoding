



rm run_pca-kmcluster_evaluation.log



python pca-kmcluster_evaluation.py --num_components 2 3 4 5 6 7 8 9 10 15 20 30 40 50 60 80 100 --num_kmclusters 5 --reps 100 --input_dir "../inputs/preprocessed_data/" --output_dir "../inputs/baseline_data/scaPCA_output/" |& tee -a run_pca-kmcluster_evaluation.log









