# conda activate tf
rm run_all.log

python 1_Main_Scripts/sca_datamerger.py --mode both --input_dir "../inputs/raw_input" --output_dir "../inputs/raw_input_combined" |& tee -a run_all.log


python 1_Main_Scripts/sca_preprocessor.py --mingenes 200 --mincells 1 --maxfeatures 1500 --maxmito 5 --features 1000 --input_dir "../inputs/raw_input_combined/filtered_matrices_mex/hg19/" --output_dir "../inputs/preprocessed_data/" --outputplot_dir "../outputs/preprocessed_data/" --verbosity 0 |& tee -a run_all.log

#Rscript 1_Main_Scripts/sca_preprocessing.R ../inputs/raw_input_combined/filtered_matrices_mex/hg19/ ../inputs/preprocessed_data/ ../outputs/preprocessed_data/ 200 1750 5 2000 |& tee -a run_all.log

python 2_Baseline_Scripts/sca_PCA.py --num_components 50 --input_dir "../inputs/preprocessed_data/" --output_dir "../inputs/baseline_data/scaPCA_output/" --outputplot_dir "../outputs/baseline_data/scaPCA_output/" |& tee -a run_all.log
python 2_Baseline_Scripts/sca_ICA.py --num_components 2 --input_dir "../inputs/baseline_data/scaPCA_output/" --output_dir "../inputs/baseline_data/scaICA_output/" --outputplot_dir "../outputs/baseline_data/scaICA_output/" |& tee -a run_all.log
python 2_Baseline_Scripts/sca_LSA.py --num_components 50 --input_dir "../inputs/preprocessed_data/" --output_dir "../inputs/baseline_data/scaLSA_output/" --outputplot_dir "../outputs/baseline_data/scaLSA_output/" |& tee -a run_all.log
python 2_Baseline_Scripts/sca_Isomap.py --num_components 2 --input_dir "../inputs/baseline_data/scaPCA_output/" --output_dir "../inputs/baseline_data/scaIsomap_output/" --outputplot_dir "../outputs/baseline_data/scaIsomap_output/" |& tee -a run_all.log
python 2_Baseline_Scripts/sca_tSNE.py --num_components 2 --verbosity 0 --input_dir "../inputs/baseline_data/scaPCA_output/" --output_dir "../inputs/baseline_data/scaTSNE_output/" --outputplot_dir "../outputs/baseline_data/scaTSNE_output/" |& tee -a run_all.log
python 2_Baseline_Scripts/sca_UMAP.py --num_components 2 --verbosity 0 --input_dir "../inputs/baseline_data/scaPCA_output/" --output_dir "../inputs/baseline_data/scaUMAP_output/" --outputplot_dir "../outputs/baseline_data/scaUMAP_output/" |& tee -a run_all.log

### Evaluate the baselines with Kmeans clustering
python 1_Main_Scripts/sca_kmcluster.py --reset --title "PCA" --k 5 --dimensions 0 --verbosity 0 --input_dir "../inputs/baseline_data/scaPCA_output/" --output_dir "../outputs/kmcluster/" --outputplot_dir "../outputs/kmcluster/PCA/" |& tee -a run_all.log
python 1_Main_Scripts/sca_kmcluster.py --title "ICA" --k 5 --dimensions 0 --verbosity 0 --input_dir "../inputs/baseline_data/scaICA_output/" --output_dir "../outputs/kmcluster/" --outputplot_dir "../outputs/kmcluster/ICA/" |& tee -a run_all.log
python 1_Main_Scripts/sca_kmcluster.py --title "LSA" --k 5 --dimensions 0 --verbosity 0 --input_dir "../inputs/baseline_data/scaLSA_output/" --output_dir "../outputs/kmcluster/" --outputplot_dir "../outputs/kmcluster/LSA/" |& tee -a run_all.log
python 1_Main_Scripts/sca_kmcluster.py --title "Isomap" --k 5 --dimensions 0 --verbosity 0 --input_dir "../inputs/baseline_data/scaIsomap_output/" --output_dir "../outputs/kmcluster/" --outputplot_dir "../outputs/kmcluster/Isomap/" |& tee -a run_all.log
python 1_Main_Scripts/sca_kmcluster.py --title "t-SNE" --k 5 --dimensions 0 --verbosity 0 --input_dir "../inputs/baseline_data/scaTSNE_output/" --output_dir "../outputs/kmcluster/" --outputplot_dir "../outputs/kmcluster/tSNE/" |& tee -a run_all.log
python 1_Main_Scripts/sca_kmcluster.py --title "UMAP" --k 5 --dimensions 0 --verbosity 0 --input_dir "../inputs/baseline_data/scaUMAP_output/" --output_dir "../outputs/kmcluster/" --outputplot_dir "../outputs/kmcluster/UMAP/" |& tee -a run_all.log

### Evaluate the baselines with classification
python 1_Main_Scripts/sca_classification.py --reset --title "PCA" --kfold 5 --classifier "logreg" --input_dir "../inputs/baseline_data/scaPCA_output/" --output_dir "../outputs/ova_classification/" |& tee -a run_all.log
python 1_Main_Scripts/sca_classification.py --title "ICA" --kfold 5 --classifier "logreg" --input_dir "../inputs/baseline_data/scaICA_output/" --output_dir "../outputs/ova_classification/" |& tee -a run_all.log
python 1_Main_Scripts/sca_classification.py --title "LSA" --kfold 5 --classifier "logreg" --input_dir "../inputs/baseline_data/scaLSA_output/" --output_dir "../outputs/ova_classification/" |& tee -a run_all.log
python 1_Main_Scripts/sca_classification.py --title "Isomap" --kfold 5 --classifier "logreg" --input_dir "../inputs/baseline_data/scaIsomap_output/" --output_dir "../outputs/ova_classification/" |& tee -a run_all.log
python 1_Main_Scripts/sca_classification.py --title "t-SNE" --kfold 5 --classifier "logreg" --input_dir "../inputs/baseline_data/scaTSNE_output/" --output_dir "../outputs/ova_classification/" |& tee -a run_all.log
python 1_Main_Scripts/sca_classification.py --title "UMAP" --kfold 5 --classifier "logreg" --input_dir "../inputs/baseline_data/scaUMAP_output/" --output_dir "../outputs/ova_classification/" |& tee -a run_all.log


