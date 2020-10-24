rm -rf logs
mkdir logs

logfile="logs/0_timestamps.log"
logtext="logs/0_logtext.log"



techname=run_preprocessing
start=`date +%s`
echo $techname |& tee -a $logfile
printf "%s %s %s %s %s %s:> starting $techname\n" `date` &>> $logfile
bash run_preprocessing.sh &>> $logtext
printf "%s %s %s %s %s %s:> finished $techname\n" `date` &>> $logfile
end=`date +%s`
printf "$techname took %d minutes\n\n" `echo "($end-$start)/60" | bc` &>> $logfile




techname=run_baselines
start=`date +%s`
echo $techname |& tee -a $logfile
printf "%s %s %s %s %s %s:> starting $techname\n" `date` &>> $logfile
bash run_baselines.sh &>> $logtext
printf "%s %s %s %s %s %s:> finished $techname\n" `date` &>> $logfile
end=`date +%s`
printf "$techname took %d minutes\n\n" `echo "($end-$start)/60" | bc` &>> $logfile




techname=run_evaluation_baselines
start=`date +%s`
echo $techname |& tee -a $logfile
printf "%s %s %s %s %s %s:> starting $techname\n" `date` &>> $logfile
bash run_evaluation_baseliens.sh &>> $logtext
printf "%s %s %s %s %s %s:> finished $techname\n" `date` &>> $logfile
end=`date +%s`
printf "$techname took %d minutes\n\n" `echo "($end-$start)/60" | bc` &>> $logfile




echo "i'm done"