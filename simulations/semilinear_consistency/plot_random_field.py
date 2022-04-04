from dolfin import *
from random_field import RandomField
import matplotlib.pyplot as plt
import sys
import numpy as np

import os
outdir = "random_field/"
if not os.path.exists(outdir):
	os.makedirs(outdir)


N = 5 # number of samples

rf = RandomField()
n = 32
mesh = UnitSquareMesh(n, n)
U = FunctionSpace(mesh, "DG", 0)
rf.function_space = U


for j in range(N):

	u = rf.sample()

	filename = "random_field_seed_" + str(rf.version)

	plot(u)
	plt.savefig(outdir + filename + ".pdf", bbox_inches="tight")
	plt.close()
