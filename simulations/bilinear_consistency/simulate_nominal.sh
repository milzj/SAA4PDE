date=$(date '+%d-%b-%Y-%H-%M-%S')
n="128"
python simulate_nominal.py $n $date
python certify_nominal.py "Nominal_Simulation_n=${n}_date=${date}/${date}_nominal_solution_n=$n"
