


source ~/anaconda3/etc/profile.d/conda.sh # to source the conda command. Check directory if it doesn't work.
conda activate tf
conda env list	# it should be visible in the log-textfile. I'm not saving it to anything else. 



mkdir logs

pcalog=logs/2_baselines_PCA.log
icalog=logs/2_baselines_ICA.log
lsalog=logs/2_baselines_LSA.log
tsnelog=logs/2_baselines_tSNE.log
umaplog=logs/2_baselines_UMAP.log



input="../inputs/data/preprocessed_data/"
output="../inputs/baseline_data/"
outputplot="../outputs/baselines/"




(
### PCA
logfile=$pcalog
foldername="scaPCA_output/"

printf "############################################################################\n################### " &>> $logfile
echo -n START: `date` &>> $logfile
printf " ###################\n############################################################################\n\n" &>> $logfile


python ../2_Baseline_Scripts/sca_PCA.py --mode complete --num_components 100 --input_dir $input --output_dir ${output}${foldername} --outputplot_dir ${outputplot}${foldername} |& tee -a $logfile


printf "\n################### " &>> $logfile
echo -n DONE: `date` &>> $logfile
printf " ####################\n############################################################################\n\n\n\n\n\n" &>> $logfile
) & 







### ICA
(
logfile=$icalog
foldername="scaICA_output/"

printf "############################################################################\n################### " &>> $logfile
echo -n START: `date` &>> $logfile
printf " ###################\n############################################################################\n\n" &>> $logfile


python ../2_Baseline_Scripts/sca_ICA.py --num_components 100 --mode complete --input_dir $input --output_dir ${output}${foldername} --outputplot_dir ${outputplot}${foldername} |& tee -a $logfile
 

printf "\n################### " &>> $logfile
echo -n DONE: `date` &>> $logfile
printf " ####################\n############################################################################\n\n\n\n\n\n" &>> $logfile
) & 







### LSA
(
logfile=$lsalog
foldername="scaLSA_output/"

printf "############################################################################\n################### " &>> $logfile
echo -n START: `date` &>> $logfile
printf " ###################\n############################################################################\n\n" &>> $logfile

python ../2_Baseline_Scripts/sca_LSA.py --mode complete --num_components 100 --input_dir $input --output_dir ${output}${foldername} --outputplot_dir ${outputplot}${foldername} |& tee -a $logfile


printf "\n################### " &>> $logfile
echo -n DONE: `date` &>> $logfile
printf " ####################\n############################################################################\n\n\n\n\n\n" &>> $logfile
) & 








### tSNE
(
logfile=$tsnelog
foldername="scaTSNE_output/"

printf "############################################################################\n################### " &>> $logfile
echo -n START: `date` &>> $logfile
printf " ###################\n############################################################################\n\n" &>> $logfile


python ../2_Baseline_Scripts/sca_tSNE.py --mode nosplit --num_components 2 --dimensions 30 --verbosity 0 --input_dir $input --output_dir ${output}${foldername} --outputplot_dir ${outputplot}${foldername} |& tee -a $logfile


printf "\n################### " &>> $logfile
echo -n DONE: `date` &>> $logfile
printf " ####################\n############################################################################\n\n\n\n\n\n" &>> $logfile
) & 







### UMAP
(
logfile=$umaplog
foldername="scaUMAP_output/"

printf "############################################################################\n################### " &>> $logfile
echo -n START: `date` &>> $logfile
printf " ###################\n############################################################################\n\n" &>> $logfile


python ../2_Baseline_Scripts/sca_UMAP.py --mode complete --num_components 2 --dimensions 30 --verbosity 0 --input_dir $input --output_dir ${output}${foldername} --outputplot_dir ${outputplot}${foldername} |& tee -a $logfile


printf "\n################### " &>> $logfile
echo -n DONE: `date` &>> $logfile
printf " ####################\n############################################################################\n\n\n\n\n\n" &>> $logfile
) 

