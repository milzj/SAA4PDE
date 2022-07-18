
date=$(date '+%d-%b-%Y-%H-%M-%S')
source problem_data.sh
experiment="Regularization_Parameter_Synthetic"
experiment="Monte_Carlo_Rate_Synthetic"
mpiexec -n 4 python simulate_experiment.py $date $experiment $betaFilename $Nref











