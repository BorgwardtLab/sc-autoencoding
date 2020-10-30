


source ~/anaconda3/etc/profile.d/conda.sh # to source the conda command. Check directory if it doesn't work.
conda activate tf
printf "[optimize_nLSA.sh ] "
conda env list	# it should be visible in the log-textfile. I'm not saving it to anything else. 

mkdir logs

logfile="logs/5_optimize_nLSA.log"
timestamps=$logfile

lsadir="../inputs/baseline_data/scaLSA_output/"
output_dir="../outputs/optimization/nLSA/"



dbs_start=`date +%s`
go=`date`

echo "Evaluate nLSA" |& tee -a $timestamps
echo Starting: $go |& tee -a $timestamps

(
for limit in {002..100}; do
	(
	python ../4_Evaluation/sca_randforest.py --title "${limit}-LSAs" --limit_dims $limit --input_dir $lsadir --output_dir "${output_dir}random_forest/" |& tee -a $logfile
	) &
	(
	python ../4_Evaluation/sca_kmcluster.py --title "${limit}-LSAs" --k 10 --limit_dims $limit --input_dir $lsadir --output_dir "${output_dir}kmcluster/" |& tee -a $logfile
	) &
	#(
	#python ../4_Evaluation/sca_dbscan.py  --title "${limit}-LSAs" --eps 20 --min_samples 3 --limit_dims $limit --input_dir $lsadir --output_dir "${output_dir}dbscan/" |& tee -a $logfile
	#) &
done

wait 
)

python ../4_Evaluation/visualize.py  --title "nLSA results"  --output_dir "../outputs/optimization/nLSA/" --random_forest_results "${output_dir}random_forest/" --kmcluster_results "${output_dir}kmcluster/" |& tee -a $logfile
# --general_input "../outputs/optimization/nLSA/"

wait


dbs_end=`date +%s`


echo Finished: `date` |& tee -a $timestamps
printf "\nnLSA search took %d minutes\n\n\n" `echo "($dbs_end-$dbs_start)/60" | bc` |& tee -a $timestamps



