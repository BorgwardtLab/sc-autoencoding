
directories=(
"../inputs/baseline_data/scaPCA_output/"
"../inputs/baseline_data/scaICA_output/"
"../inputs/baseline_data/scaLSA_output/"
"../inputs/baseline_data/scaTSNE_output/"
"../inputs/baseline_data/scaUMAP_output/"
)


titles=(
"PCA"
"ICA"
"LSA"
"tSNE"
"UMAP"
)


mkdir logs


# make sure titles and directories have the same length
if [ ${#directories[@]} = ${#titles[@]} ]; then 
range=$(eval echo "{0..$[${#directories[@]}-1]}")
else
exit
fi







(
tech=random_forest
output_dir=../outputs/$tech/
ntrees=100

for i in $range; do
(
input_dir=${directories[$i]}
logfile=logs/4_${tech}_${titles[$i]}.log

printf "############################################################################\n################### " &>> $logfile
echo -n START: `date` &>> $logfile
printf " ###################\n############################################################################\n\n" &>> $logfile

python ../4_Evaluation/sca_randforest.py --title ${titles[$i]} --n_trees $ntrees --input_dir $input_dir --output_dir $output_dir |& tee -a $logfile

printf "\n################### " &>> $logfile
echo -n DONE: `date` &>> $logfile
printf " ####################\n############################################################################\n\n\n\n\n\n" &>> $logfile
) &
done

)



(
tech=kmcluster
output_dir=../outputs/$tech/

for i in $range; do
(
input_dir=${directories[$i]}
logfile=logs/4_${tech}_${titles[$i]}.log

printf "############################################################################\n################### " &>> $logfile
echo -n START: `date` &>> $logfile
printf " ###################\n############################################################################\n\n" &>> $logfile

python ../4_Evaluation/sca_kmcluster.py --title ${titles[$i]} --k 10 --dimensions 0 --verbosity 0 --input_dir $input_dir --output_dir $output_dir |& tee -a $logfile

printf "\n################### " &>> $logfile
echo -n DONE: `date` &>> $logfile
printf " ####################\n############################################################################\n\n\n\n\n\n" &>> $logfile
) &
done

)




wait
echo "All Done"






