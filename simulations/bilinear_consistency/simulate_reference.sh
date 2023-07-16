#!/bin/bash

#SBATCH ...

export ...

date=$(date '+%d-%b-%Y-%H-%M-%S')
n="128"
N="1000"

mpiexec -n 48 python simulate_reference.py $n $N $date
python certify_reference.py "Reference_Simulation_n=${n}_N=${N}_date=${date}/${date}_reference_solution_mpi_rank=0_N=${N}_n=$n"










