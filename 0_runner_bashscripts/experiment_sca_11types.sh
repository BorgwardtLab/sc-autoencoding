
source ~/anaconda3/etc/profile.d/conda.sh # to source the conda command. Check directory if it doesn't work.
conda activate tf
printf "[SCA experiment 11types]" 
conda env list	# it should be visible in the log-textfile. I'm not saving it to anything else.






# AEtypes=("normal" "poisson" "nb" "nb-conddisp" "nb-shared" "nb-fork" "zinb" "zinb-conddisp" "zinb-shared" "zinb-fork" "zinb-elempi")


# worked ones: nb normal poisson zinbconddisp zinbfork


AEtypes=("nb-conddisp" "nb-shared" "nb-fork" "zinb-shared" "zinb-elempi" "zinb")




preprocessed_ctdata="../inputs/data/preprocessed_data_autoencoder/"
outdir="../outputs/experiments/AEtypes/"




(
for AEtype in ${AEtypes[@]}; do
(
#remember to change filenames as well
echo $AEtype
logfile=logs/6_11AEtypes_$AEtype.log


start=`date +%s`
printf "############################################################################\n################### " &>> $logfile
echo -n START: `date` |& tee -a $logfile
printf " ###################\n############################################################################\n\n" &>> $logfile



	(
	python ../3_Autoencoder/sca_autoencoder.py --mode complete --AEtype $AEtype --input_dir "../inputs/data/preprocessed_data_autoencoder/" --output_dir "${outdir}bca_data/${AEtype}/" --outputplot_dir "${outdir}bca_data/${AEtype}/" |& tee -a $logfile

	(
	python ../4_Evaluation/sca_kmcluster.py --title ${AEtype} --k 10 --limit_dims 0 --verbosity 0 --input_dir "${outdir}bca_data/${AEtype}/" --output_dir "${outdir}cluster_result/" |& tee -a $logfile
	) & (
	python ../4_Evaluation/sca_randforest.py --title ${AEtype} --n_trees 100 --input_dir "${outdir}bca_data/${AEtype}/" --output_dir "${outdir}randomforest_result/" |& tee -a $logfile
	) & (
	python ../4_Evaluation/sca_hierarchcluster.py --k 10 --threshold 0.0 --title ${AEtype} --num_reps 50 --limit_dims 0 --input_dir "${outdir}bca_data/${AEtype}/" --output_dir "${outdir}hierarchical/" |& tee -a $logfile
	) & (
	python ../4_Evaluation/sca_svm.py --title ${AEtype} --limit_dims 0 --input_dir "${outdir}bca_data/${AEtype}/" --output_dir "${outdir}svm/" |& tee -a $logfile
	)

	wait
	) 
	
	
end=`date +%s`
printf "\nSCA experiment $AEtype took %d minutes\n" `echo "($end-$start)/60" | bc` |& tee -a $logfile
printf "\n################### " &>> $logfile
echo -n DONE: `date` |& tee -a $logfile
printf " ####################\n############################################################################\n\n\n\n\n\n" &>> $logfile
	
) &
done
wait
)


echo `date` "I'm now done with all evaluations, and can start the visualization"



python ../4_Evaluation/visualize.py  --title "AEtypes"  --output_dir ${outdir} --kmcluster_results "${outdir}cluster_result/" --random_forest_results "${outdir}randomforest_result/" --svm_results "${outdir}svm/" --hierarch_results "${outdir}hierarchical/"





# python ../4_Evaluation/visualize.py  --title "AEtypes"  --output_dir ${outdir} --kmcluster_results "${outdir}cluster_result/"
# python ../4_Evaluation/visualize.py  --title "AEtypes"  --output_dir ${outdir} --random_forest_results "${outdir}randomforest_result/"
# python ../4_Evaluation/visualize.py  --title "AEtypes"  --output_dir ${outdir} --svm_results "${outdir}svm/"
# python ../4_Evaluation/visualize.py  --title "AEtypes"  --output_dir ${outdir} --hierarch_results "${outdir}hierarchical/"


echo "11ae type experiment is now officially d-o-n-e. 
