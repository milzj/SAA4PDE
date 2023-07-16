date=$(date '+%d-%b-%Y-%H-%M-%S')
n=9
N=`expr $n \* $n`
mpiexec -n 4 python simulate_reference.py $n $N $date
python certify_reference.py Simulation_n=${n}_N=${N}_date=${date}/${date}_reference_solution_mpi_rank=0_N=${N}_n=${n}










