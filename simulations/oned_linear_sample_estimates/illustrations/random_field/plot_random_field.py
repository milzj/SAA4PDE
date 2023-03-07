if __name__ == "__main__":

	import numpy as np
	import sys, os
	from dolfin import *

	currentdir = os.path.dirname(os.path.realpath(__file__))
	parentdir = os.path.dirname(currentdir)
	parentparentdir = os.path.dirname(parentdir)
	sys.path.append(parentparentdir)

	from  random_field import RandomField

	import matplotlib.pyplot as plt
	from stats import figure_style
	from stats.surface_function import surface_function
	from base.savefig import savefig

	from discrete_sampler import DiscreteSampler

	outdir = "output/"
	if not os.path.exists(outdir):
		os.makedirs(outdir)

	n = int(sys.argv[1])
	N = int(sys.argv[2])
	var = int(sys.argv[3])

	rf = RandomField(mean=0.0, var=var)
	mesh = UnitIntervalMesh(n)
	U = FunctionSpace(mesh, "CG", 1)
	rf.function_space = U
	u = Function(U)

	discrete_sampler = DiscreteSampler()


	for i in range(N):

		sample = discrete_sampler.sample(sample_index = 0, N=1)
		v = rf.sample(sample=sample)
		u.interpolate(v)

		plot(u)
		plt.tight_layout()
		filename = "random_field_sample=" + str(i)
		figure_name = outdir + filename
		savefig(figure_name)

