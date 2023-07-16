date=$(date '+%d-%b-%Y-%H-%M-%S')
source problem_data.sh
python simulate_nominal.py $n $date $betaFilename
python certify_nominal.py "Nominal_Simulation_n=${n}_date=${date}/${date}_nominal_solution_n=$n" $betaFilename
