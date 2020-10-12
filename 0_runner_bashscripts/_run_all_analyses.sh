
filename = timestamps_analyses.log




date |& tee -a $filename
echo "starting all analyses" |& tee -a $filename



bash run_baselines.sh
date  |& tee -a $filename
echo "finished run_baselines.sh" |& tee -a $filename

bash run_autoencoder_DCA.sh
date |& tee -a $filename
echo "finished run_autoencoder_DCA.sh" |& tee -a $filename

bash run_autoencoder.sh
date |& tee -a $filename
echo "finished run_autoencoder.sh" |& tee -a $filename




echo "finished running all runners" |& tee -a $filename




echo "" >> $filename
echo "" >> $filename
echo "" >> $filename
echo "" >> $filename
echo "" >> $filename
echo "" >> $filename
echo "" >> $filename
echo "" >> $filename



