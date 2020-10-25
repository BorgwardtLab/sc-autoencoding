start=`date +%s`

mkdir logs
logfile=logs/3_autoencoder_SCA.log


printf "\n\n" #for the logtxt, not saved into logfile
printf "############################################################################\n################### " &>> $logfile
echo -n START: `date` &>> $logfile
printf " ###################\n############################################################################\n\n" &>> $logfile


######## SCA
source ~/anaconda3/etc/profile.d/conda.sh # to source the conda command. Check directory if it doesn't work.
conda activate tf
printf "[run_SCA.sh ] " |& tee -a $logfile 
conda env list	|& tee -a $logfile 

(
python ../3_Autoencoder/sca_autoencoder.py --loss poisson_loss --mode nosplit --input_dir "../inputs/data/preprocessed_data_autoencoder/" --output_dir "../inputs/autoencoder_data/SCA_output/" --outputplot_dir "../outputs/autoencoder_data/SCA/" |& tee -a $logfile
) & (
python ../3_Autoencoder/sca_autoencoder.py --loss poisson_loss --mode split --input_dir "../inputs/data/preprocessed_data_autoencoder/" --output_dir "../inputs/autoencoder_data/SCA_output/" --outputplot_dir "../outputs/autoencoder_data/SCA/" |& tee -a $logfile
)
wait

end=`date +%s`
printf "\nSCA took %d minutes\n" `echo "($end-$start)/60" | bc` &>> $logfile
printf "\n################### " &>> $logfile
echo -n DONE: `date` &>> $logfile
printf " ####################\n############################################################################\n\n\n\n\n\n" &>> $logfile




