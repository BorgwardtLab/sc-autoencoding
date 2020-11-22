
logfile="./logs/6_replicate_Zheng.log"


#python ../1_Processing/sca_datamerger.py --mode both --input_dir "../inputs/replication_zheng/original/" --output_dir "../inputs/replication_zheng/combined/"


python ../9_toyscripts/rp_barcodes_add_meaningless_column.py --file "../outputs/experiments/replication_zheng/original/filtered_matrices_mex/hg19/barcodes_original.tsv" --outfile "../outputs/experiments/replication_zheng/original/filtered_matrices_mex/hg19/barcodes.tsv" |& tee -a $logfile

python ../1_Processing/sca_countdata_preprocessor.py --n_splits 0 --mingenes 200 --mincells 1 --maxfeatures 1500 --maxmito 5 --features 1000 --test_fraction 0.25 --input_dir "../outputs/experiments/replication_zheng/original/filtered_matrices_mex/hg19/" --output_dir "../outputs/experiments/replication_zheng/preprocessed/" --verbosity 0 |& tee -a $logfile

python ../3_Autoencoder/sca_autoencoder_2neck.py --mode nosplit --loss poisson_loss --AEtype 'zinb-shared' --input_dir "../outputs/experiments/replication_zheng/preprocessed/" --output_dir "../outputs/experiments/replication_zheng/AE_output/" --outputplot_dir "../outputs/experiments/replication_zheng/AE_output/" |& tee -a $logfile


#'zinb-conddisp': ZINBAutoencoder,
#'zinb': ZINBConstantDispAutoencoder, 


    # AE_types = {'': Autoencoder, '': PoissonAutoencoder,
    #         '': NBConstantDispAutoencoder, 'nb-conddisp': NBAutoencoder,
    #         'nb-shared': NBSharedAutoencoder, 'nb-fork': NBForkAutoencoder,
    #         'zinb-shared': ZINBSharedAutoencoder, 'zinb-fork': ZINBForkAutoencoder,
    #         'zinb-elempi': ZINBAutoencoderElemPi}    





directories=(
"../outputs/experiments/replication_zheng/AE_output/"
)

titles=(
"Zheng_2nodes"
)

#how many repetition for clustering
reps=1

general_output=../outputs/experiments/replication_zheng/results/


mkdir logs
errfile="../ERROR_ERROR_ERROR_ERROR_ERROR_ERROR_ERROR.error"

# make sure titles and directories have the same length
if [ ${#directories[@]} = ${#titles[@]} ]; then 
range=$(eval echo "{0..$[${#directories[@]}-1]}")
else
exit
fi



n_reps=1


# 		PCA	ICA	LSA	tSE	UMP	ori	DCA	BCA	SCA	denoi)
k_array=(9 	6 	8 	8 	7 	10 	8 	8 	8 	8)
k_array=(10 10 	10 	10 	10 	10 	10 	10 	10 	10)

(
tech=kmcluster
output_dir=${general_output}${tech}/

for i in $range; do
	(
	input_dir=${directories[$i]}
	logfile=logs/6_${tech}_${titles[$i]}.log

	printf "############################################################################\n################### " &>> $logfile
	echo -n START: `date` &>> $logfile
	start=`date +%s`
	printf " ###################\n############################################################################\n\n" &>> $logfile

	python ../4_Evaluation/sca_kmcluster.py --title ${titles[$i]} --k ${k_array[$i]} --num_reps $reps --n_init 10 --limit_dims 0 --verbosity 0 --input_dir $input_dir --output_dir $output_dir |& tee -a $logfile

	end=`date +%s`
	printf "\n$tech took %d minutes\n" `echo "($end-$start)/60" | bc` &>> $logfile
	printf "\n################### " &>> $logfile
	echo -n DONE: `date` &>> $logfile
	printf " ####################\n############################################################################\n\n\n\n\n\n" &>> $logfile
	) &
done
wait # we ABSOLUTELY need a wait within the brackets, and a "&" outside of it in order to ensure the last echo to wait for all commands before ending the script
) 



hierarchi_k=(10)
(
tech=hierarchical
output_dir=${general_output}${tech}/

for i in $range; do
	(
	input_dir=${directories[$i]}
	logfile=logs/4_${tech}_${titles[$i]}.log

	printf "############################################################################\n################### " &>> $logfile
	echo -n START: `date` &>> $logfile
	start=`date +%s`
	printf " ###################\n############################################################################\n\n" &>> $logfile

	python ../6_Evaluation/sca_hierarchcluster.py --k ${hierarchi_k[$i]} --threshold 0.0 --title ${titles[$i]} --num_reps $reps --limit_dims 0 --input_dir $input_dir --output_dir $output_dir |& tee -a $logfile

	end=`date +%s`
	printf "\n$tech took %d minutes\n" `echo "($end-$start)/60" | bc` &>> $logfile
	printf "\n################### " &>> $logfile
	echo -n DONE: `date` &>> $logfile
	printf " ####################\n############################################################################\n\n\n\n\n\n" &>> $logfile
	)
done
wait # we ABSOLUTELY need a wait within the brackets, and a "&" outside of it in order to ensure the last echo to wait for all commands before ending the script
) 







wait





### Visualize
logfile=logs/4_visualizer.log


printf "\n\n" #for the logtxt, not saved into logfile
printf "############################################################################\n################### " &>> $logfile
echo -n START: `date` &>> $logfile
printf " ###################\n############################################################################\n\n" &>> $logfile
start=`date +%s`


python ../4_Evaluation/visualize.py  --title main --hierarch_results "../outputs/results/hierarchical/" --output_dir "../outputs/results/visualized_results/" |& tee -a $logfile
python ../4_Evaluation/visualize.py  --title main --kmcluster_results "../outputs/results/kmcluster/" --output_dir "../outputs/results/visualized_results/" |& tee -a $logfile




end=`date +%s`
printf "\nVisualization took %d minutes\n" `echo "($end-$start)/60" | bc` &>> $logfile
printf "\n################### " &>> $logfile
echo -n DONE: `date` &>> $logfile
printf " ####################\n############################################################################\n\n\n\n\n\n" &>> $logfile



wait
echo "All Done - " `date`






