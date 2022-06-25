#!/bin/bash

#SBATCH ...

export ...

date=$(date '+%d-%b-%Y-%H-%M-%S')
n="128"
N="10,100"
python simulate_dependent.py $n $N $date







