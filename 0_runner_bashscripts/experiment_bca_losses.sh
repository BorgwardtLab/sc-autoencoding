
source ~/anaconda3/etc/profile.d/conda.sh # to source the conda command. Check directory if it doesn't work.
conda activate tf
printf "[BCA experiment losses] " 
conda env list	# it should be visible in the log-textfile. I'm not saving it to anything else.



mkdir logs
logfile="logs/6_exBCA_losses.log"

preprocessed_ctdata="../inputs/data/preprocessed_data_autoencoder/"
outdir="../outputs/experiments/losses/"



losses=("poisson_loss" "poisson" "mse" "mae" "mape" "msle" "squared_hinge" "hinge" "binary_crossentropy" "categorical_crossentropy" "kld" "cosine_proximity")
losses=("poisson" "mse")

losses=("poisson_loss")





start=`date +%s`
printf "############################################################################\n################### " &>> $logfile
echo -n START: `date` |& tee -a $logfile
printf " ###################\n############################################################################\n\n" &>> $logfile


(
for loss in ${losses[@]}; do
echo $loss

	(
	python ../3_Autoencoder/bca_autoencoder.py --mode complete --loss $loss --activation relu --optimizer Adam --input_dir $preprocessed_ctdata --output_dir "${outdir}bca_data/${loss}/" --outputplot_dir "${outdir}bca_data/${loss}/"  |& tee -a $logfile
	
	(
	python ../4_Evaluation/sca_kmcluster.py --title ${loss} --k 8 --limit_dims 0 --verbosity 0 --input_dir "${outdir}bca_data/${loss}/" --output_dir "${outdir}cluster_result/" |& tee -a $logfile
	) & (
	python ../4_Evaluation/sca_randforest.py --title ${loss} --n_trees $ntrees --input_dir "${outdir}bca_data/${loss}/" --output_dir "${outdir}randomforest_result/" |& tee -a $logfile
	)
	
	
	) &


done
wait
)


(
python ../4_Evaluation/visualize.py  --title "BCAloss"  --output_dir ${outdir} --random_forest_results "${outdir}randomforest_result/" --kmcluster_results "${outdir}cluster_result/" |& tee -a $logfile
)



end=`date +%s`
printf "\nPreprocessing took %d minutes\n" `echo "($end-$start)/60" | bc` |& tee -a $logfile
printf "\n################### " &>> $logfile
echo -n DONE: `date` |& tee -a $logfile
printf " ####################\n############################################################################\n\n\n\n\n\n" &>> $logfile
)



