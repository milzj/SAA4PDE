import numpy as np
from dolfin import *

import matplotlib.pyplot as plt
from matplotlib import cm
from stats.surface_function import surface_function
from base.savefig import savefig
from stats.figure_style import *

import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from bilinear_sampler import BilinearSampler

from random_diffusion_coefficient import RandomDiffusionCoefficient
from random_nonnegative_coefficient import RandomNonnegativeCoefficient

def plot_random_field(outdir, rf, samples, control, n, random_field_name):


	i = 0
	for sample in samples:

		g = rf.sample(sample)

		c = plot(g)
		plt.colorbar(c)

		filename = outdir + random_field_name + "_sample=" + str(i)
		savefig(filename)

		surface_function(g, n, cmap=cm.coolwarm)
		filename = outdir + random_field_name + "_surface_sample=" + str(i)
		title = r"Sample of ${}$ ($i={}$)".format(random_field_name, i)

		if random_field_name == "kappa":
			title = r"Sample of $\kappa$ ($i={}$)".format(i)

		plt.title(title)
		savefig(filename)

		i += 1

if __name__ == "__main__":

	import os, sys


	n = 128
	mesh = UnitSquareMesh(n,n)
	U = FunctionSpace(mesh, "DG", 0)
	control = Function(U)

	N = int(sys.argv[1])


	sampler = BilinearSampler()
	sampler._seed = 10

	samples_kappa = []
	samples_g = []

	for i in range(N):
		sample_kappa, sample_g = np.split(sampler.sample(0), 2)
		samples_kappa.append(sample_kappa)
		samples_g.append(sample_g)

	outdir = "output/random_diffusion_coefficient/"
	if not os.path.exists(outdir):
		os.makedirs(outdir)

	rf = RandomDiffusionCoefficient()
	rf.function_space = control.function_space()
	rf.plot_eigenvalues("output/")


	plot_random_field(outdir, rf, samples_kappa, control, n, "kappa")

	samples = []

	for i in range(N):

		sample = np.random.uniform(-np.sqrt(3), np.sqrt(3), 400)
		samples.append(sample)



	outdir = "output/random_nonnegative_coefficient/"
	if not os.path.exists(outdir):
		os.makedirs(outdir)

	rf = RandomNonnegativeCoefficient()
	rf.function_space = control.function_space()

	plot_random_field(outdir, rf, samples_g, control, n, "g")




