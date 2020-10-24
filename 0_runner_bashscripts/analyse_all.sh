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
printf "$techname took %d minutes\n\n\n" `echo "($end-$start)/60" | bc` &>> $logfile




# techname=run_baselines
# start=`date +%s`
# echo $techname |& tee -a $logfile
# printf "%s %s %s %s %s %s:> starting $techname\n" `date` &>> $logfile
# bash run_baselines.sh &>> $logtext
# printf "%s %s %s %s %s %s:> finished $techname\n" `date` &>> $logfile
# end=`date +%s`
# printf "$techname took %d minutes\n\n" `echo "($end-$start)/60" | bc` &>> $logfile
(
techname=run_baselines
start=`date +%s`
#echo $techname |& tee -a $logfile
border1="%s %s %s %s %s %s:> starting $techname\n" `date`
bash run_baselines.sh &>> $logtext
border2="%s %s %s %s %s %s:> finished $techname\n" `date`
end=`date +%s`
border3="$techname took %d minutes\n\n\n" `echo "($end-$start)/60" | bc`
string=${techname}\n${border1}${border2}${border3}
printf $string
) & 



(
techname=run_DCA
start=`date +%s`
#echo $techname |& tee -a $logfile
border1="%s %s %s %s %s %s:> starting $techname\n" `date`
bash run_baselines.sh &>> $logtext
border2="%s %s %s %s %s %s:> finished $techname\n" `date`
end=`date +%s`
border3="$techname took %d minutes\n\n\n" `echo "($end-$start)/60" | bc`
string=${techname}\n${border1}${border2}${border3}
printf $string
)



techname=run_evaluation_baselines
start=`date +%s`
echo $techname |& tee -a $logfile
printf "%s %s %s %s %s %s:> starting $techname\n" `date` &>> $logfile
bash run_evaluation_baseliens.sh &>> $logtext
printf "%s %s %s %s %s %s:> finished $techname\n" `date` &>> $logfile
end=`date +%s`
printf "$techname took %d minutes\n\n\n" `echo "($end-$start)/60" | bc` &>> $logfile




echo "i'm done"
