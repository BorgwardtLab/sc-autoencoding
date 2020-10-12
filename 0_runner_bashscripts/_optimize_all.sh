
mkdir logs
filename=logs/timestamps_optimizations.log


date |& tee -a $filename
echo "starting all optimizations" |& tee -a $filename




bash optimize_DBScan.sh
date |& tee -a $filename
echo "finished optimize_DBScan.sh" |& tee -a $filename

bash optimize_num_k_kmcluster.sh
date |& tee -a $filename
echo "finished optimize_num_k_kmcluster.sh" |& tee -a $filename

bash optimize_num_PCA.sh
date |& tee -a $filename
echo "finished optimize_num_PCA.sh" |& tee -a $filename

bash optimize_random_forrest.sh
date |& tee -a $filename
echo "finished optimize_random_forrest.sh" |& tee -a $filename






echo "finished running all runners" |& tee -a $filename




echo "" >> $filename
echo "" >> $filename
echo "" >> $filename
echo "" >> $filename
echo "" >> $filename
echo "" >> $filename
echo "" >> $filename
echo "" >> $filename
