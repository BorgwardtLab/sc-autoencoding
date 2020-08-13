
date
echo "starting"

bash run_baselines.sh
date
echo "finished run_baselines.sh"


bash run_autoencoder.sh
date
echo "finished run_autoencoder.sh"




bash evaluate_pca-kmcluster.sh
date
echo "finished evaluate_pca-kmcluster.sh"




echo "finished running all runners"



