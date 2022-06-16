rm -rf output

n="128"
N="5"
python plot_random_field.py $n $N

cd output
convert -dispose 2 -delay 80 -loop 0 random_field_surface_sample\=*.png random_field.gif
mv random_field.gif ../random_field.gif
cd ..
