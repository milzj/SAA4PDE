#!/bin/bash

outdir="output/Simulation"
filename="solution_n={}"

source data.sh

python plot_convergence.py "$n_vec" $outdir $filename

cd output/Simulation
convert -dispose 2 -delay 80 -loop 0 solution_n\=*.png convergence.gif
mv convergence.gif ../../convergence.gif
cd ../..
