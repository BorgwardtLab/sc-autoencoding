

mkdir logs


logfile="logs/1_preprocessing.log"

printf "############################################################################\n################### " &>> $logfile
echo -n START: `date` &>> $logfile
printf " ###################\n############################################################################\n\n" &>> $logfile


python ../1_Processing/sca_datamerger.py --mode both --input_dir "../inputs/data/raw_input" --output_dir "../inputs/data/raw_input_combined" |& tee -a $logfile
python ../1_Processing/sca_preprocessor.py --n_splits 3 --mingenes 200 --mincells 1 --maxfeatures 1500 --maxmito 5 --features 2000 --test_fraction 0.25 --input_dir "../inputs/data/raw_input_combined/filtered_matrices_mex/hg19/" --output_dir "../inputs/data/preprocessed_data/" --outputplot_dir "../outputs/preprocessing/preprocessed_data/" --verbosity 0 |& tee -a $logfile
python ../1_Processing/sca_countdata_preprocessor.py --n_splits 3 --mingenes 200 --mincells 1 --maxfeatures 1500 --maxmito 5 --features 2000 --test_fraction 0.25 --input_dir "../inputs/data/raw_input_combined/filtered_matrices_mex/hg19/" --output_dir "../inputs/data/preprocessed_data_autoencoder/" --verbosity 0 |& tee -a $filename


printf "\n################### " &>> $logfile
echo -n DONE: `date` &>> $logfile
printf " ####################\n############################################################################\n\n\n\n\n\n" &>> $logfile



