
date
echo "starting"

bash optimize_DBScan.sh
date
echo "finished optimize_DBScan.sh"



bash optimize_num_k_kmcluster.sh
date
echo "finished optimize_num_k_kmcluster.sh"



bash optimize_num_PCA.sh
date
echo "finished optimize_num_PCA.sh"



bash optimize_random_forrest.sh
date
echo "finished optimize_random_forrest.sh"





echo "finished running all runners"



