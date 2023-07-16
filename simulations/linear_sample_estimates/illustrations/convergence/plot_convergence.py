
import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentparentdir = os.path.dirname(parentdir)
sys.path.append(parentparentdir)

from plot_control import plot_control

n_vec = str(sys.argv[1])
outdir = str(sys.argv[2])
_filename = str(sys.argv[3])

n_vec = n_vec.split(" ")
n_vec = [int(n) for n in n_vec]

for n in n_vec:
	plot_control(outdir + "/" + _filename.format(n), zlim=[-7, 7],
			title="SAA solution with mesh width $1/{}$ and sample size ${}$".format(n,n**2))
