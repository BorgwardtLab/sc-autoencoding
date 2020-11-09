
mkdir logs
logfile=logs/5_optimize_DBScan_gridsearch_bash.log
timestamps=$logfile



source ~/anaconda3/etc/profile.d/conda.sh # to source the conda command. Check directory if it doesn't work.
conda activate tf
printf "[dbscan_gridsearch_bash.sh ] "
conda env list	# it should be visible in the log-textfile. I'm not saving it to anything else. 



pcadir="../inputs/baseline_data/scaPCA_output/"
output_dir="../outputs/optimization/technique_evaluation/dbscan_gridsearch_bash/"



# minpts=(002 003 004 005 008 010 020 060 200) already done

minpts=(030 040 050 080 090 100 150 200 250 300) # batch 2
eps=(005 010 015 020 025 030 035 040 050 100 300)

#minpts=(002 003)
#eps=(018 019)







printf "\n\n" #for the logtxt
printf "############################################################################\n################### " &>> $logfile
echo -n START: `date` |& tee -a $logfile
printf " ###################\n############################################################################\n\n" &>> $logfile
start=`date +%s`
echo ""

(
for mp in ${minpts[@]}; do
	(
	for ep in ${eps[@]}; do
		(
		python ../4_Evaluation/sca_dbscan.py  --title "mp${mp}_ep${ep}" --verbosity 0 --eps $ep --min_samples $mp --input_dir $pcadir --output_dir $output_dir &>> $logfile
		echo I just did ep $ep and mp $mp: `date` |& tee -a $logfile
		) #&
	done
	wait
	) #&
done

wait 
)

echo "I'm done with all the gridpoints" |& tee -a $logfile
echo `date` |& tee -a $logfile




python ../5_Optimization/dbscan_gridsearch_visualizer.py --input_dir $output_dir --output_dir $output_dir &>> $logfile








end=`date +%s`
printf "\ndbscan_gridsearch took %d minutes\n" `echo "($end-$start)/60" | bc` |& tee -a $logfile
printf "\n################### " &>> $logfile
echo -n DONE: `date` |& tee -a $logfile
printf " ####################\n############################################################################\n\n\n\n\n\n" &>> $logfile

