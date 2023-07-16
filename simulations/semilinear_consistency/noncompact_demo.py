from dolfin import *

import matplotlib.pyplot as plt

import sys
import os
import numpy as np

if not os.path.exists("noncompact_demo"):
		os.makedirs("noncompact_demo")

def noncompact_demo(k):
	"""
	Creates plots of the functions 2.0*sin(pi*k*x[0])*sin(pi*k*x[1])
	for several k. The functions can be used to show that the set of
	square integrable functions on (0,1)^2 with values in [-2,2] are
	noncompact, as they lack a norm-convergent subsequence. Roughly
	speaking, the set is noncompact as it contains oscillatory functions.
	"""

	input_dir = "noncompact_demo/noncompact_demo_k={}".format(int(k))

	n = 2**8

	mesh = UnitSquareMesh(n, n)
	U = FunctionSpace(mesh, "CG", 1)
	u = Function(U)

	u_expr = Expression("2.0*sin(pi*k*x[0])*sin(pi*k*x[1])", k = k, degree = 5)
	u.interpolate(u_expr)

	plot(u)
	plt.savefig(input_dir + ".pdf", bbox_inches="tight")
	plt.close()

for k in np.arange(13, 18):
	noncompact_demo(float(k))
