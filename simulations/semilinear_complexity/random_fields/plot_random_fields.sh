#!/bin/bash

rm -rf output

Nsamples="10"

python plot_random_fields.py $Nsamples


cd output/
convert -dispose 2 -delay 80 -loop 0 b_sample\=*.png random_rhs.gif
convert -dispose 2 -delay 80 -loop 0 g_sample\=*.png random_g.gif
convert -dispose 2 -delay 80 -loop 0 kappa_sample\=*.png random_diffusion_coefficient.gif

