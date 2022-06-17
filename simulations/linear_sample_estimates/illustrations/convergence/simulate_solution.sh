#!/bin/bash

#SBATCH ...

export ...

source data.sh

date=$(date '+%d-%b-%Y-%H-%M-%S')
for n in $n_vec
do
	mpiexec -n 48 python simulate_solution.py $n 1 $date
done










