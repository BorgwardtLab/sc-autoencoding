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





# Slooooow
######################
#python ../3_Autoencoder/sca_autoencoder.py --loss poisson_loss --mode complete --input_dir "../inputs/data/preprocessed_data_autoencoder/" --output_dir "../inputs/autoencoder_data/SCA_output/" --outputplot_dir "../outputs/autoencoder_data/SCA/" &>> $logfile



# Faster and better
###################### 
# (
# python ../3_Autoencoder/sca_autoencoder.py --loss poisson_loss --mode nosplit --input_dir "../inputs/data/preprocessed_data_autoencoder/" --output_dir "../inputs/autoencoder_data/SCA_output/" --outputplot_dir "../outputs/autoencoder_data/SCA/" &>> $logfile
# ) & (
# python ../3_Autoencoder/sca_autoencoder.py --loss poisson_loss --mode split --input_dir "../inputs/data/preprocessed_data_autoencoder/" --output_dir "../inputs/autoencoder_data/SCA_output/" --outputplot_dir "../outputs/autoencoder_data/SCA/" &>> $logfile
# )
##### FIX OUTPUT IF YOU USE THIS




# Fastest, but not dynamic so pay attention that all splits are taken care of
(
output1=$(python ../3_Autoencoder/sca_autoencoder.py --mode nosplit --loss poisson_loss --input_dir "../inputs/data/preprocessed_data_autoencoder/" --output_dir "../inputs/autoencoder_data/SCA_output/" --outputplot_dir "../outputs/autoencoder_data/SCA/")
) & (
output2=$(python ../3_Autoencoder/sca_autoencoder.py --mode split --splitnumber 1 --loss poisson_loss --input_dir "../inputs/data/preprocessed_data_autoencoder/" --output_dir "../inputs/autoencoder_data/SCA_output/" --outputplot_dir "../outputs/autoencoder_data/SCA/")
) & (
output3=$(python ../3_Autoencoder/sca_autoencoder.py --mode split --splitnumber 2 --loss poisson_loss --input_dir "../inputs/data/preprocessed_data_autoencoder/" --output_dir "../inputs/autoencoder_data/SCA_output/" --outputplot_dir "../outputs/autoencoder_data/SCA/")
) & (
output4=$(python ../3_Autoencoder/sca_autoencoder.py --mode split --splitnumber 3 --loss poisson_loss --input_dir "../inputs/data/preprocessed_data_autoencoder/" --output_dir "../inputs/autoencoder_data/SCA_output/" --outputplot_dir "../outputs/autoencoder_data/SCA/")
)

wait

printf "%s" "output1" &>> $logfile
printf "%s" "output2" &>> $logfile
printf "%s" "output3" &>> $logfile
printf "%s" "output4" &>> $logfile

echo "SCA is done" |& tee -a $logfile
echo "(accept this as replacement for the actual stdout)" #to avoid cluttering the logtxt from analyse_all.sh





end=`date +%s`
printf "\nSCA took %d minutes\n" `echo "($end-$start)/60" | bc` &>> $logfile
printf "\n################### " &>> $logfile
echo -n DONE: `date` &>> $logfile
printf " ####################\n############################################################################\n\n\n\n\n\n" &>> $logfile




