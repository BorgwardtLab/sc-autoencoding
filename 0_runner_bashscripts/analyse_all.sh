rm -rf logs
mkdir logs

timestamps="logs/0_timestamps.log"
logtext="logs/0_logtext.log"



techname=run_preprocessing.py
start=`date +%s`
echo $techname |& tee -a $timestamps
printf "%s %s %s %s %s %s:> starting $techname\n" `date` &>> $timestamps
bash run_preprocessing.sh &>> $logtext
printf "%s %s %s %s %s %s:> finished $techname\n" `date` &>> $timestamps
end=`date +%s`
printf "$techname took %d minutes\n\n\n" `echo "($end-$start)/60" | bc` &>> $timestamps




	# techname=run_baselines.py
	# start=`date +%s`
	# echo $techname |& tee -a $timestamps
	# printf "%s %s %s %s %s %s:> starting $techname\n" `date` &>> $timestamps
	# bash run_baselines.sh &>> $logtext
	# printf "%s %s %s %s %s %s:> finished $techname\n" `date` &>> $timestamps
	# end=`date +%s`
	# printf "$techname took %d minutes\n\n" `echo "($end-$start)/60" | bc` &>> $timestamps
	(
	techname=run_baselines.py
	
	startdate=`date`
	start=`date +%s`
	bash run_baselines.sh &>> $logtext
	end=`date +%s`
	
	# write all at once, so it doesn't get into a conflict with the parallel running processes also accessing timestamps
	echo $techname |& tee -a $timestamps
	printf "%s %s %s %s %s %s:> starting $techname\n" $startdate &>> $timestamps
	printf "%s %s %s %s %s %s:> finished $techname\n" `date` &>> $timestamps
	printf "$techname took %d minutes\n\n\n" `echo "($end-$start)/60" | bc` &>> $timestamps
	) & 


	(
	techname=run_DCA.py

	startdate=`date`
	start=`date +%s`
	bash run_baselines.sh &>> $logtext
	end=`date +%s`

	# write all at once, so it doesn't get into a conflict with the parallel running processes also accessing timestamps
	echo $techname |& tee -a $timestamps
	printf "%s %s %s %s %s %s:> starting $techname\n" $startdate &>> $timestamps
	printf "%s %s %s %s %s %s:> finished $techname\n" `date` &>> $timestamps
	printf "$techname took %d minutes\n\n\n" `echo "($end-$start)/60" | bc` &>> $timestamps
	)



techname=run_evaluation_baselines.py
start=`date +%s`
echo $techname |& tee -a $timestamps
printf "%s %s %s %s %s %s:> starting $techname\n" `date` &>> $timestamps
bash run_evaluation_baseliens.sh &>> $logtext
printf "%s %s %s %s %s %s:> finished $techname\n" `date` &>> $timestamps
end=`date +%s`
printf "$techname took %d minutes\n\n\n" `echo "($end-$start)/60" | bc` &>> $timestamps




echo "i'm done"
