from dolfin import *
from random_field import RandomField
import matplotlib.pyplot as plt
import sys
from stats import figure_style
from stats.surface_function import surface_function
import numpy as np

import os
outdir = "random_field/"
if not os.path.exists(outdir):
	os.makedirs(outdir)


N = 5 # number of samples

rf = RandomField()
n = 70
mesh = UnitSquareMesh(n, n)
U = FunctionSpace(mesh, "CG", 1)
rf.function_space = U

rf.plot_eigenvalues(outdir)


for j in range(N):

	u = rf.sample()

	filename = "random_field_seed_" + str(rf.version)

	plot(u)
	plt.savefig(outdir + filename + ".pdf", bbox_inches="tight")
	plt.close()

	surface_function(u, n)
	plt.savefig(outdir + filename + "_surface" + ".pdf", bbox_inches="tight")
	plt.close()
