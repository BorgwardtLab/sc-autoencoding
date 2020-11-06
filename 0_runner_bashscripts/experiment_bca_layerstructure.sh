
source ~/anaconda3/etc/profile.d/conda.sh # to source the conda command. Check directory if it doesn't work.
conda activate tf
printf "[BCA experiment layers]" 
conda env list	# it should be visible in the log-textfile. I'm not saving it to anything else.



mkdir logs
logfile="logs/6_exBCA_layerstructures.log"

preprocessed_ctdata="../../inputs/data/preprocessed_data_autoencoder/"
outdir="../../outputs/experiments/layerstructure/"


autoencoders=(asymmetric default extraconstant inverted large larger short smaller)


start=`date +%s`
printf "############################################################################\n################### " &>> $logfile
echo -n START: `date` |& tee -a $logfile
printf " ###################\n############################################################################\n\n" &>> $logfile





for aename in ${autoencoders[@]}; do
	(
	echo START: `date` |& tee -a $logfile
	locstart=`date +%s`

	python ../3_Autoencoder/ex_bca_layerstructures/bca_autoencoder_${aename}.py --mode complete --loss poisson --activation relu --optimizer RMSprop --input_dir $preprocessed_ctdata --output_dir "${outdir}bca_data/${aename}/" --outputplot_dir "${outdir}bca_data/${aename}/"  |& tee -a $logfile

	locend=`date +%s` 
	echo END: `date` |& tee -a $logfile
	printf "\n${aename} took %d minutes\n" `echo "($locend-$locstart)/60" | bc` |& tee -a $logfile
	)
done




# ########## Analyse afterwards I guess
# (
# for aename in ${autoencoders[@]}; do
	# (
	# python ../4_Evaluation/sca_kmcluster.py --title ${aename} --k 8 --limit_dims 0 --verbosity 0 --input_dir "${outdir}bca_data/${aename}/" --output_dir "${outdir}cluster_result/" |& tee -a $logfile
	# ) & (
	# python ../4_Evaluation/sca_randforest.py --title ${aename} --n_trees 100 --input_dir "${outdir}bca_data/${aename}/" --output_dir "${outdir}randomforest_result/" |& tee -a $logfile
	# ) & (
	# python ../4_Evaluation/sca_dbscan.py  --title ${aename} --verbosity 0 --eps 17 --min_samples 3 --input_dir "${outdir}bca_data/${aename}/" --output_dir "${outdir}dbscan_result/" |& tee -a $logfile
	# ) &

# done
# wait
# )



python ../4_Evaluation/visualize.py  --title "BCAlayers"  --output_dir ${outdir} --random_forest_results "${outdir}randomforest_result/" --kmcluster_results "${outdir}cluster_result/" --dbscan_results "${outdir}dbscan_result/" |& tee -a $logfile



end=`date +%s`
printf "\nBCA experiment layerstructure took %d minutes\n" `echo "($end-$start)/60" | bc` |& tee -a $logfile
printf "\n################### " &>> $logfile
echo -n DONE: `date` |& tee -a $logfile
printf " ####################\n############################################################################\n\n\n\n\n\n" &>> $logfile




