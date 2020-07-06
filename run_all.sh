# conda activate tf

python 1_Main_Scripts/sca_datamerger.py |& tee -a run_all.log
python 2_Baseline_Scripts/sca_PCA.py |& tee -a run_all.log
python 2_Baseline_Scripts/sca_LSA.py |& tee -a run_all.log
python 2_Baseline_Scripts/sca_Isomap.py |& tee -a run_all.log
python 2_Baseline_Scripts/sca_tSNE.py |& tee -a run_all.log
python 2_Baseline_Scripts/sca_UMAP.py |& tee -a run_all.log