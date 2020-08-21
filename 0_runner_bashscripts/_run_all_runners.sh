
date
echo "starting"

bash run_baselines.sh
date
echo "finished run_baselines.sh"



bash run_autoencoder_DCA.sh
date
echo "finished run_autoencoder_DCA.sh"




bash run_autoencoder.sh
date
echo "finished run_autoencoder.sh"







echo "finished running all runners"



