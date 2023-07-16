import numpy as np
import fenics

import matplotlib.pyplot as plt
from base.savefig import savefig
from stats.figure_style import *

import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from discrete_sampler import DiscreteSampler
from random_semilinear_problem import RandomSemilinearProblem

def plot_random_field(outdir, rsp, samples):


	i = 0

	kappa = fenics.Function(rsp.U)
	g = fenics.Function(rsp.U)
	f = fenics.Function(rsp.U)

	random_field_names = ["kappa", "g", "b"]

	for sample in samples:

		sample1, sample2, sample3, sample4 = sample

		kappa_expr, kappa_parameters = rsp.kappa_expr, rsp.kappa_parameters
		f_expr, f_parameters = rsp.f_expr, rsp.f_parameters
		g_expr, g_parameters = rsp.g_expr, rsp.g_parameters

		for k in range(rsp.m):
			kappa_parameters["p{}".format(k)] = sample1[k]
			kappa_parameters["q{}".format(k)] = sample2[k]
			g_parameters["r{}".format(k)] = sample3[k]
			f_parameters["t{}".format(k)] = sample4[k]

		kappa_expr.user_parameters.update(kappa_parameters)
		g_expr.user_parameters.update(g_parameters)
		f_expr.user_parameters.update(f_parameters)

		kappa.interpolate(kappa_expr)

		g.interpolate(g_expr)

		f.interpolate(f_expr)

		for (random_field_name, v) in zip(random_field_names, [kappa, g, f]):

			c = fenics.plot(v)
			plt.colorbar(c)

			filename = outdir + random_field_name + "_sample=" + str(i)
			title = r"Sample of ${}$ ($i={}$)".format(random_field_name, i)

			if random_field_name == "kappa":
				title = r"Sample of $\kappa$ ($i={}$)".format(i)

			plt.title(title)
			savefig(filename)

		i += 1

if __name__ == "__main__":

	n = 64

	rsp = RandomSemilinearProblem(n)
	N = int(sys.argv[1])

	sampler = DiscreteSampler()
	sampler._seed = 10

	samples = []

	for i in range(N):
		samples.append(sampler.sample(0))

	outdir = "output/"
	if not os.path.exists(outdir):
		os.makedirs(outdir)

	plot_random_field(outdir, rsp, samples)




