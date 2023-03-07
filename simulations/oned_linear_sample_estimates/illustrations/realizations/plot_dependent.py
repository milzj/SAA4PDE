
import os, sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentparentdir = os.path.dirname(parentdir)
sys.path.append(parentparentdir)

from plot_control import plot_control

n = int(sys.argv[1])
N = int(sys.argv[2])
Nsamples = int(sys.argv[3])
outdir = str(sys.argv[4])
_filename = str(sys.argv[5])

for i in range(Nsamples):
	plot_control(outdir + "/" + _filename.format(i), zlim=[-7, 7], 
			title="SAA solution with mesh width $1/{}$ and sample size ${}$ \n (realization $i={}$)".format(n,N, i))
