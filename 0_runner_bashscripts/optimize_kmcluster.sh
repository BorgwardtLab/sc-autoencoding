


source ~/anaconda3/etc/profile.d/conda.sh # to source the conda command. Check directory if it doesn't work.
conda activate tf
printf "[optimize_kmcluster.sh ] "
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
logfile="logs/5_optimize_kmcluster.log"
timestamps=$logfile


dbs_start=`date +%s`
go=`date`



echo "Evaluate kmcluster" |& tee -a $timestamps
echo Starting: $go |& tee -a $timestamps




# make sure titles and directories have the same length
if [ ${#directories[@]} = ${#titles[@]} ]; then 
range=$(eval echo "{0..$[${#directories[@]}-1]}")
else
echo "LENGTHS OF DIRECTORIES AND TITLES IS NOT EQUAL"
exit
fi




########################### Random_forest
###############################################################################################################################
(
for i in $range; do


	for k in {02..20}; do
	(
	python ../4_Evaluation/sca_kmcluster.py --title "${k}-PCAs" --k $k --limit_dims 0 --input_dir ${directories[$i]} --output_dir "${output_dir}kmcluster_k/${titles[$i]}/" |& tee -a $logfile
	) &
	done

done
wait
)

echo "starting visualization"

for i in $range; do
	echo "${output_dir}kmcluster_k/${titles[$i]}/"
	python ../4_Evaluation/visualize.py  --title "${titles[$i]}" --plottitle ${titles[$i]} --kmcluster_results "${output_dir}kmcluster_k/${titles[$i]}/" --output_dir "${output_dir}random_forest/kmcluster_k/" |& tee -a $logfile
done

wait

dbs_end=`date +%s`


echo Finished: `date` |& tee -a $timestamps
printf "\nrand_forest optimization took %d minutes\n\n\n" `echo "($dbs_end-$dbs_start)/60" | bc` |& tee -a $timestamps



