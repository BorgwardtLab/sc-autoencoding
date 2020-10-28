


source ~/anaconda3/etc/profile.d/conda.sh # to source the conda command. Check directory if it doesn't work.
conda activate tf
printf "[optimize_nPCA.sh ] "
conda env list	# it should be visible in the log-textfile. I'm not saving it to anything else. 




pcadir="../inputs/baseline_data/scaPCA_output/"
output_dir="../outputs/optimization/nPCA/"


for limit in {2..100}; do;
	(
	python ../4_Evaluation/sca_randforest.py --title "${limit}-PCAs" --limit_dims $limit --input_dir $pcadir --output_dir "${output_dir}random_forest/"
	) &
	(
	python ../4_Evaluation/sca_kmcluster.py --title "${limit}-PCAs" --k 10 --input_dir $pcadir --output_dir "${output_dir}kmcluster/"
	) &
	(
	python ../4_Evaluation/sca_dbscan.py  --title "${limit}-PCAs" --eps 20 --min_samples 3 --input_dir $pcadir --output_dir "${output_dir}dbscan/"
	) &
done



wait 


python ../4_Evaluation/visualize.py  --title "nPCA results" --general_input "../outputs/optimization/nPCA/" --output_dir "../outputs/optimization/nPCA/"


