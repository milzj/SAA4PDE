#!/bin/bash

source data.sh

python plot_dependent.py $n $N $Nsamples $outdir $filename

cd $outdir
convert -dispose 2 -delay 80 -loop 0 realization\=*.png realizations_N=${N}_n=${n}.gif
mv realizations_N=${N}_n=${n}.gif ../../realizations_N=${N}_n=${n}.gif
cd ../..
