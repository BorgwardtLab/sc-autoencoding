source ~/anaconda3/etc/profile.d/conda.sh # to source the conda command. Check directory if it doesn't work.
conda activate tf
printf "[optimize_dbscan_epsearch.sh ] "
conda env list	# it should be visible in the log-textfile. I'm not saving it to anything else. 


directories=(
"../inputs/baseline_data/scaPCA_output/"
"../inputs/baseline_data/scaICA_output/"
"../inputs/baseline_data/scaLSA_output/"
"../inputs/baseline_data/scaTSNE_output/"
"../inputs/baseline_data/scaUMAP_output/"
"../inputs/data/preprocessed_data/"
"../inputs/autoencoder_data/DCA_output/"
"../inputs/autoencoder_data/BCA_output/"
"../inputs/autoencoder_data/SCA_output/"
)

titles=(
"PCA"
"ICA"
"LSA"
"tSNE"
"UMAP"
"original_data"
"DCA"
"BCA"
"SCA"
)


mkdir logs
output_dir="../outputs/optimization/technique_evaluation/"

logfile="logs/5_optimize_dbscan_epsearch.log"
timestamps=$logfile




# make sure titles and directories have the same length
if [ ${#directories[@]} = ${#titles[@]} ]; then 
range=$(eval echo "{0..$[${#directories[@]}-1]}")
else
exit
fi


########################## DBScan: epsearch
###############################################################################################################################
(
dbs_start=`date +%s`
go=`date`

echo "Evaluate DBScan" |& tee -a $timestamps
echo Starting: $go |& tee -a $timestamps


for i in $range; do

	(
	python ../5_Optimization/dbscan_epssearch_knn.py --title ${titles[$i]} --input_dir ${directories[$i]} --outputplot_dir "${output_dir}dbscan/epsearch/" |& tee -a $logfile
	) &
done


wait


dbs_end=`date +%s`


echo Finished: `date` |& tee -a $timestamps
printf "\ndbscan_epsearch took %d minutes\n\n\n" `echo "($dbs_end-$dbs_start)/60" | bc` |& tee -a $timestamps

) 