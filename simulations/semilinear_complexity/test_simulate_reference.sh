date=$(date '+%d-%b-%Y-%H-%M-%S')

source problem_data.sh
N="4"
mpiexec -n 4 python simulate_reference.py $n $N $date $betaFilename
python certify_reference.py "Reference_Simulation_n=${n}_N=${N}_date=${date}/${date}_reference_solution_mpi_rank=0_N=${N}_n=$n" $betaFilename $N










