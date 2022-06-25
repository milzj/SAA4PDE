#!/bin/bash

rm -rf output

Nsamples="5"

python plot_random_fields.py $Nsamples


cd output/random_nonnegative_coefficient
convert -dispose 2 -delay 80 -loop 0 g_surface_sample\=*.png random_nonnegative_coefficient.gif
cd ../random_diffusion_coefficient
convert -dispose 2 -delay 80 -loop 0 kappa_surface_sample\=*.png random_diffusion_coefficient.gif
cd ../..


cp output/random_nonnegative_coefficient/random_nonnegative_coefficient.gif output/random_nonnegative_coefficient.gif
cp output/random_diffusion_coefficient/random_diffusion_coefficient.gif output/random_diffusion_coefficient.gif
