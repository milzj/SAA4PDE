date=$(date '+%d-%b-%Y-%H-%M-%S')
source data.sh
for n in $n_vec
do
	mpiexec -n 4 python simulate_solution.py $n 1 $date
done









