start=`date +%s`

mkdir logs
logfile=logs/3_autoencoder_BCA.log


printf "\n\n" #for the logtxt, not saved into logfile
printf "############################################################################\n################### " &>> $logfile
echo -n START: `date` &>> $logfile
printf " ###################\n############################################################################\n\n" &>> $logfile


######## BCA
source ~/anaconda3/etc/profile.d/conda.sh # to source the conda command. Check directory if it doesn't work.
conda activate tf
printf "[run_BCA.sh ] " |& tee -a $logfile 
conda env list	|& tee -a $logfile 





# Slooooow
######################
python ../3_Autoencoder/sca_autoencoder.py --mode complete --loss poisson --activation relu --optimizer Adam --input_dir "../inputs/data/preprocessed_data_autoencoder/" --output_dir "../inputs/autoencoder_data/BCA_output/" --outputplot_dir "../outputs/autoencoder_data/BCA/" &>> $logfile



# Faster and better
###################### 




# Fastest, but not dynamic so pay attention that all splits are taken care of
(










wait
echo "BCA is done"

end=`date +%s`
printf "\nBCA took %d minutes\n" `echo "($end-$start)/60" | bc` &>> $logfile
printf "\n################### " &>> $logfile
echo -n DONE: `date` &>> $logfile
printf " ####################\n############################################################################\n\n\n\n\n\n" &>> $logfile




