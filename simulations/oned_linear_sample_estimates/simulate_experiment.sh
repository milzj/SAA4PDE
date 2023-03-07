#!/bin/bash

#SBATCH ...

#export ...

E="n_N_variable_small"
#mpirun -np 16 python simulate_experiment.py $E Simulation_n=144_N=20736_date=05-Mar-2023-23-11-07/05-Mar-2023-23-11-07_reference_solution_mpi_rank=0_N=20736_n=144
python simulate_experiment.py $E Simulation_n=144_N=20736_date=05-Mar-2023-23-11-07/05-Mar-2023-23-11-07_reference_solution_mpi_rank=0_N=20736_n=144










