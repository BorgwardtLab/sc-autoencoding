
logfile=logs/0_timestamps.log
logtext=logs/0_logtext.log



printf `date`":> starting run_preprocessing\n"  &>> $logfile
bash run_preprocessing.sh &>> $0_logtext
printf `date`":> starting run_preprocessing\n\n"  &>> $logfile



printf `date`":> starting run_baselines\n"  &>> $logfile
bash run_baselines.sh &>> $0_logtext
printf `date`":> starting run_baselines\n\n"  &>> $logfile



printf `date`":> starting run_evaluate_baselines\n"  &>> $logfile
bash run_evaluation_baseliens.sh &>> $0_logtext
printf `date`":> starting run_evaluate_baselines\n\n"  &>> $logfile





