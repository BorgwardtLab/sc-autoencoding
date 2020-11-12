
source ~/anaconda3/etc/profile.d/conda.sh # to source the conda command. Check directory if it doesn't work.
conda activate tf
printf "[SCA experiment 11types]" 
conda env list	# it should be visible in the log-textfile. I'm not saving it to anything else.




#AEtypes=("normal" "poisson" "nb" "nb-conddisp" "nb-shared" "nb-fork" "zinb" "zinb-conddisp" "zinb-shared" "zinb-fork" "zinb-elempi")
AEtypes=("normal" "poisson" "nb")



start=`date +%s`
printf "############################################################################\n################### " &>> $logfile
echo -n START: `date` |& tee -a $logfile
printf " ###################\n############################################################################\n\n" &>> $logfile


preprocessed_ctdata="../inputs/data/preprocessed_data_autoencoder/"
outdir="../outputs/experiments/AEtypes/"




(
for AEtype in ${AEtypes[@]}; do


#remember to change filenames as well
echo $AEtype

	(
	python ../3_Autoencoder/sca_autoencoder.py --mode complete --AEtype $AEtype --input_dir "../inputs/data/preprocessed_data_autoencoder/" --output_dir "${outdir}bca_data/${$AEtype}/" --outputplot_dir "${outdir}bca_data/${$AEtype}/"

	(
	python ../4_Evaluation/sca_kmcluster.py --title ${AEtype} --k 8 --limit_dims 0 --verbosity 0 --input_dir "${outdir}bca_data/${$AEtype}/" --output_dir "${outdir}cluster_result/" |& tee -a $logfile
	) & (
	python ../4_Evaluation/sca_randforest.py --title ${AEtype} --n_trees 100 --input_dir "${outdir}bca_data/${$AEtype}/" --output_dir "${outdir}randomforest_result/" |& tee -a $logfile
	)

	wait
	)
done
wait
)




(
python ../4_Evaluation/visualize.py  --title "BCAopti"  --output_dir ${outdir} --random_forest_results "${outdir}randomforest_result/" --kmcluster_results "${outdir}cluster_result/" --dbscan_results "${outdir}dbscan_result/" |& tee -a $logfile
)



end=`date +%s`
printf "\nBCA experiment activation took %d minutes\n" `echo "($end-$start)/60" | bc` |& tee -a $logfile
printf "\n################### " &>> $logfile
echo -n DONE: `date` |& tee -a $logfile
printf " ####################\n############################################################################\n\n\n\n\n\n" &>> $logfile




