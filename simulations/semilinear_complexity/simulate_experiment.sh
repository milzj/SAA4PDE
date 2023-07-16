#!/bin/bash


#SBATCH ...

export ...


date=$(date '+%d-%b-%Y-%H-%M-%S')
source problem_data.sh
experiment="Regularization_Parameter"
experiment="Monte_Carlo_Rate"
experiment="Dimension_Dependence"
experiment="Dimension_Dependence_small_alpha"
mpiexec -n 48 python simulate_experiment.py $date $experiment $betaFilename $Nref










