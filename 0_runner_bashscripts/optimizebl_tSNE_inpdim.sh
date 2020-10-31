


source ~/anaconda3/etc/profile.d/conda.sh # to source the conda command. Check directory if it doesn't work.
conda activate tf
printf "[optimize_tsne_inpdim.sh ] "
conda env list	# it should be visible in the log-textfile. I'm not saving it to anything else. 

mkdir logs




PCA_output="../inputs/baseline_data/scaPCA_output/"
output="../outputs/optimization/"



numbers=(026 028 033 037 040 050 055 070 090)
#numbers=(016 017 018 019 021 022 023)
#numbers=(003 005 010 020 030 045 060 080 100)
#numbers=(2 3 4 5 6 7 8 9 10 12 14 16 18 20 24 28 32 36 40 45 50 55 60 65 70 75 80 85 90 95 100)


logfile="logs/5_optimize_tSNE_inDs.log"

foldername="tsne_nimput/"
folderdata="tsne_data/"
folderclust="tsne_kmclresult/"
foldertree="tsne_result/"


printf "\n\n" #for the logtxt
printf "############################################################################\n################### " &>> $logfile
echo -n START: `date` |& tee -a $logfile
printf " ###################\n############################################################################\n\n" &>> $logfile
start=`date +%s`



ntrees=100

(
for limit in ${numbers[@]}; do

	(
	python ../2_Baseline_Scripts/sca_tSNE.py --mode nosplit --num_components 2 --input_dims $limit --verbosity 0 --input_dir $PCA_output --output_dir "${output}${foldername}${folderdata}${limit}/" --outputplot_dir "${output}${foldername}${folderdata}${limit}/" |& tee -a $logfile 

	printf "\n\nDimension reduction done, now proceding with evaluation\n\n" |& tee -a $logfile

	(
	python ../4_Evaluation/sca_kmcluster.py --title "${limit[$i]}inDs" --k 10 --limit_dims 0 --verbosity 0 --input_dir "${output}${foldername}${folderdata}${limit}/" --output_dir ${output}${foldername}${folderclust} |& tee -a $logfile
	) # & (
	#python ../4_Evaluation/sca_randforest.py --title "${limit[$i]}inDs" --n_trees $ntrees --input_dir "${output}${foldername}${folderdata}${limit}/" --output_dir ${output}${foldername}${foldertree} |& tee -a $logfile
	#)
	wait
	) &


done
wait
)

echo "I got here"

# (
# python ../4_Evaluation/visualize.py  --title "tSNE"  --output_dir ${output}${foldername} --random_forest_results ${output}${foldername}${foldertree} |& tee -a $logfile
# ) & 
(
python ../4_Evaluation/visualize.py  --title "tSNE"  --output_dir ${output}${foldername} --kmcluster_results ${output}${foldername}${folderclust} |& tee -a $logfile
)

wait

end=`date +%s`
printf "\ntSNE optimization took %d minutes\n" `echo "($end-$start)/60" | bc` |& tee -a $logfile
printf "\n################### " |& tee -a $logfile
echo -n DONE: `date` |& tee -a $logfile
printf " ####################\n############################################################################\n\n\n\n\n\n" &|& tee -a $logfile

