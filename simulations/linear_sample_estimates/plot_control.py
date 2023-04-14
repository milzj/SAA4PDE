from dolfin import *
import matplotlib.pyplot as plt

import sys
from stats import figure_style
from base import savefig

from stats.surface_function import surface_function

import numpy as np


def plot_control(filename, zlim=None, lb=None, ub=None, title=None):
	"""Generate surface plot of piecewise constant fenics function.

	The fenics function is supposed to be defined on a regular
	UnitSquareMesh with the same number of cells in each direction.
	"""

	u_vec = np.loadtxt(filename + ".txt")

	n = int(np.sqrt(len(u_vec)/2))
	mesh = UnitSquareMesh(n, n)
	U = FunctionSpace(mesh, "DG", 0)
	u = Function(U)
	u.vector()[:] = u_vec

	c = plot(u)
	plt.colorbar(c)
	savefig(filename)

	fig, ax = surface_function(u, n)
	if zlim != None:
		ax.set_zlim(zlim)
	if title != None:
		plt.title(title)

	savefig(filename + "_surface")

if __name__ == "__main__":

	input_dir = sys.argv[1]
	plot_control(input_dir, (-6., 6.))
