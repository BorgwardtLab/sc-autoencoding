
source ~/anaconda3/etc/profile.d/conda.sh # to source the conda command. Check directory if it doesn't work.
conda activate tf
printf "[BCA experiment activation]" 
conda env list	# it should be visible in the log-textfile. I'm not saving it to anything else.



mkdir logs
logfile="logs/6_exBCA_activation.log"

preprocessed_ctdata="../inputs/data/preprocessed_data_autoencoder/"
outdir="../outputs/experiments/activation/"


activations=("relu", "sigmoid", "mixed1", "mixed2")
activations=("relu", "sigmoid")



start=`date +%s`
printf "############################################################################\n################### " &>> $logfile
echo -n START: `date` |& tee -a $logfile
printf " ###################\n############################################################################\n\n" &>> $logfile


(
for activation in ${activations[@]}; do
#remember to change filenames as well
echo $optimizer
	(
	python ../3_Autoencoder/bca_autoencoder.py --mode complete --loss poisson --activation $activation --optimizer Adam --input_dir $preprocessed_ctdata --output_dir "${outdir}bca_data/${activation}/" --outputplot_dir "${outdir}bca_data/${activation}/"  |& tee -a $logfile
	
	(
	python ../4_Evaluation/sca_kmcluster.py --title ${activation} --k 8 --limit_dims 0 --verbosity 0 --input_dir "${outdir}bca_data/${activation}/" --output_dir "${outdir}cluster_result/" |& tee -a $logfile
	) & (
	python ../4_Evaluation/sca_randforest.py --title ${activation} --n_trees 100 --input_dir "${outdir}bca_data/${activation}/" --output_dir "${outdir}randomforest_result/" |& tee -a $logfile
	) & (
	python ../4_Evaluation/sca_dbscan.py  --title ${activation} --verbosity 0 --eps 17 --min_samples 3 --input_dir "${outdir}bca_data/${activation}/" --output_dir "${outdir}dbscan_result/" |& tee -a $logfile
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




