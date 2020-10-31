
source ~/anaconda3/etc/profile.d/conda.sh # to source the conda command. Check directory if it doesn't work.
conda activate tf
printf "[do preprocessing] " 
conda env list	# it should be visible in the log-textfile. I'm not saving it to anything else.



mkdir logs
logfile="logs/6_exBCA_losses.log"

preprocessed_ctdata="../inputs/data/preprocessed_data_autoencoder/"




losses=("poisson_loss" "poisson" "mse" "mae" "mape" "msle" "squared_hinge" "hinge" "binary_crossentropy" "categorical_crossentropy" "kld" "cosine_proximity")
losses=("poisson" "mse")









start=`date +%s`
printf "############################################################################\n################### " &>> $logfile
echo -n START: `date` |& tee -a $logfile
printf " ###################\n############################################################################\n\n" &>> $logfile


(
for loss in ${losses[@]}; do
echo $loss


	
	(
	python ../3_Autoencoder/bca_autoencoder.py --mode nosplit --loss $loss --activation relu --optimizer Adam --input_dir $preprocessed_ctdata --output_dir "../inputs/autoencoder_data/BCA_output/" --outputplot_dir "../outputs/autoencoder_data/BCA/"
	) &



done
wait
)





end=`date +%s`
printf "\nPreprocessing took %d minutes\n" `echo "($end-$start)/60" | bc` |& tee -a $logfile
printf "\n################### " &>> $logfile
echo -n DONE: `date` |& tee -a $logfile
printf " ####################\n############################################################################\n\n\n\n\n\n" &>> $logfile
) &


