


source ~/anaconda3/etc/profile.d/conda.sh # to source the conda command. Check directory if it doesn't work.
conda activate tf
printf "[optimize_nLSA.sh ] "
conda env list	# it should be visible in the log-textfile. I'm not saving it to anything else. 

mkdir logs




PCA_output="../inputs/baseline_data/scaPCA_output/"
output="../outputs/optimization/"


numbers=[2 3 4 5 6 7 8 9 10 12 14 16 18 20 24 28 32 36 40 45 50 55 60 65 70 75 80 85 90 95 100]


logfile="logs/5_optimize_sec_tSNE.log"

foldername="tsne_nimput/"
folderdata="tsne_tsne_data/"
folderclust="tsne_kmclresult/"
foldertree="tsne_rfresult/"


(
for limit in $numbers; do

### tSNE
	(
	printf "\n\n" #for the logtxt
	printf "############################################################################\n################### " &>> $logfile
	echo -n START: `date` &>> $logfile
	printf " ###################\n############################################################################\n\n" &>> $logfile
	start=`date +%s`

	python ../2_Baseline_Scripts/sca_tSNE.py --mode nosplit --num_components 2 --input_dims $limit --verbosity 0 --input_dir $PCA_output --output_dir ${output}${foldername}${folderdata}${limit} --outputplot_dir ${output}${foldername}${folderdata}${limit} &>> $logfile #|& tee -a $logfile


	printf "\n\nDimension reduction done, now proceding with evaluation\n\n" |& tee -a $logfile

	(
	python ../4_Evaluation/sca_kmcluster.py --title "${limit[$i]}inDs" --k 10 --dimensions 0 --verbosity 0 --input_dir ${output}${foldername}${folderdata}${limit} --output_dir ${output}${foldername}${folderclust} |& tee -a $logfile
	) & (
	python ../4_Evaluation/sca_randforest.py --title "${limit[$i]}inDs" --n_trees $ntrees --input_dir ${output}${foldername}${folderdata}${limit} --output_dir ${output}${foldername}${foldertree} |& tee -a $logfile
	)
	wait
	) &

	
done
wait
)


(
python ../4_Evaluation/visualize.py  --title "tSNE optimization results"  --output_dir ${output}${foldername} --random_forest_results ${output}${foldername}${foldertree} |& tee -a $logfile
) & (
python ../4_Evaluation/visualize.py  --title "tSNE optimization results"  --output_dir ${output}${foldername} --kmcluster_results ${output}${foldername}${folderclust} |& tee -a $logfile
)

wait

end=`date +%s`
printf "\ntSNE optimization took %d minutes\n" `echo "($end-$start)/60" | bc` &>> $logfile
printf "\n################### " &>> $logfile
echo -n DONE: `date` &>> $logfile
printf " ####################\n############################################################################\n\n\n\n\n\n" &>> $logfile






