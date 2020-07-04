conda activate tf
python 1_Main_Scripts/sca_datamerger.py 2>&1 run_all.log
python 2_Baseline_Scripts/sca_PCA.py 2>&1 run_all.log
python 2_Baseline_Scripts/sca_LSA.py 2>&1 run_all.log
python 2_Baseline_Scripts/sca_Isomap.py 2>&1 run_all.log
python 2_Baseline_Scripts/sca_tSNE.py 2>&1 run_all.log
python 2_Baseline_Scripts/sca_UMAP.py 2>&1 run_all.log