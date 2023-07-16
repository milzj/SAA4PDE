#!/bin/bash

#SBATCH -J RegParam32                    # Job name
#SBATCH --account=gts-jmilz3                     # charge account
#SBATCH -N 48 --ntasks-per-node=1                  # number of nodes and cores per node
#SBATCH --mem-per-cpu=16gb                        # Memory per core
#SBATCH -qinferno                               # QOS Name
#SBATCH -o output/Report-%x-%j.out                         # Combined output and error messages file
#SBATCH --mail-type=ALL              # Mail preferences
#SBATCH --mail-user=jmilz3@gatech.edu           # E-mail address for notifications
#SBATCH --time=24:00:00

cd $HOME/SAA4PDEComplexity/simulations/semilinear_complexity

module load anaconda3/2022.05.0.1
conda activate SAA4PDE
module load gcc

export PYTHONPATH=$HOME/SAA4PDEComplexity:${PYTHONPATH}

date=$(date '+%d-%b-%Y-%H-%M-%S')
source problem_data.sh
experiment="Monte_Carlo_Rate"
experiment="Dimension_Dependence"
experiment="Dimension_Dependence_small_alpha"
experiment="Regularization_Parameter"
experiment="Tikhonov_n=64_N=256"
mpiexec -n 48 python simulate_experiment.py $date $experiment $betaFilename $Nref















