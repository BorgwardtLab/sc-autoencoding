

source ~/anaconda3/etc/profile.d/conda.sh # to source the conda command. Check directory if it doesn't work.
conda activate tf
printf "[optimize_rand_forest.sh ] "
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

logfile="logs/5_optimize_rand_forest.log"
timestamps=$logfile


dbs_start=`date +%s`
go=`date`


echo "Evaluate rand_forest" |& tee -a $timestamps
echo Starting: $go |& tee -a $timestamps




# make sure titles and directories have the same length
if [ ${#directories[@]} = ${#titles[@]} ]; then 
range=$(eval echo "{0..$[${#directories[@]}-1]}")
else
exit
fi




########################### Random_forest
###############################################################################################################################

for i in $range; do

	if [ ${titles[$i]} = "tSNE" ] || [ ${titles[$i]} = "DCA" ]
	then	
		echo ${titles[$i]} was skipped
		continue      # Skip rest of this particular loop iteration.
	fi


		for limit in {1..200..5}; do
		(
		python ../4_Evaluation/sca_randforest.py --title "${titles[$i]}_${limit}_trees" --n_trees $limit --input_dir ${directories[$i]} --output_dir "${output_dir}random_forest_ntrees/${titles[$i]}/" |& tee -a $logfile
		) &
		done

done

for i in $range; do

python ../4_Evaluation/visualize.py  --title "${titles[$i]}" --general_input "${output_dir}random_forest/n_trees/${titles[$i]}/" --output_dir "${output_dir}random_forest/n_trees/" |& tee -a $logfile

done

wait


dbs_end=`date +%s`


echo Finished: `date` |& tee -a $timestamps
printf "\nrand_forest optimization took %d minutes\n\n\n" `echo "($dbs_end-$dbs_start)/60" | bc` |& tee -a $timestamps

