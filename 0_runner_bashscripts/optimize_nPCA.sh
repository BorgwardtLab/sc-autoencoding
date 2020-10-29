


source ~/anaconda3/etc/profile.d/conda.sh # to source the conda command. Check directory if it doesn't work.
conda activate tf
printf "[optimize_nPCA.sh ] "
conda env list	# it should be visible in the log-textfile. I'm not saving it to anything else. 

mkdir logs

logfile="logs/5_optimize_nPCA.log"
timestamps="logs/5_optimize_nPCA.tmstmp"

pcadir="../inputs/baseline_data/scaPCA_output/"
output_dir="../outputs/optimization/nPCA/"



dbs_start=`date +%s`
go=`date`

echo "Evaluate nPCA" |& tee -a $timestamps
echo Starting: $go |& tee -a $timestamps

(
for limit in {002..100}; do
	(
	python ../4_Evaluation/sca_randforest.py --title "${limit}-PCAs" --limit_dims $limit --input_dir $pcadir --output_dir "${output_dir}random_forest/" |& tee -a $logfile
	) &
	(
	python ../4_Evaluation/sca_kmcluster.py --title "${limit}-PCAs" --k 10 --limit_dims $limit --input_dir $pcadir --output_dir "${output_dir}kmcluster/" |& tee -a $logfile
	) &
	#(
	#python ../4_Evaluation/sca_dbscan.py  --title "${limit}-PCAs" --eps 20 --min_samples 3 --limit_dims $limit --input_dir $pcadir --output_dir "${output_dir}dbscan/" |& tee -a $logfile
	#) &
done

wait 
)

python ../4_Evaluation/visualize.py  --title "nPCA results"  --output_dir "../outputs/optimization/nPCA/" --random_forest_results "${output_dir}random_forest/" --kmcluster_results "${output_dir}kmcluster/" |& tee -a $logfile
# --general_input "../outputs/optimization/nPCA/"

wait


dbs_end=`date +%s`


echo Finished: `date` |& tee -a $timestamps
printf "\nnPCA search took %d minutes\n\n\n" `echo "($dbs_end-$dbs_start)/60" | bc` |& tee -a $timestamps



