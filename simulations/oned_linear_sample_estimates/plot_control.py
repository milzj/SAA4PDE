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

    if isinstance(filename, str):
        _filename = [filename]
    else:
        _filename = filename

    for fn in _filename:

        u_vec = np.loadtxt(fn + ".txt")

        n = int(len(u_vec))
        mesh = UnitIntervalMesh(n)
        U = FunctionSpace(mesh, "DG", 0)
        u = Function(U)
        u.vector()[:] = u_vec
        w = Function(FunctionSpace(mesh, "CG", 1))
        w.interpolate(u)

        c = plot(w)


    plt.ylim([-6*(1+0.1), 6*(1+0.1)])
    plt.gca().legend(("reference sol.", "''expected diffusion'' sol.", "nominal sol."))
    plt.gca().set_aspect(1/12)

    if isinstance(filename, str):
        savefig(filename)
    else:
        savefig(filename[0] + "_nom_avg")

if __name__ == "__main__":

    input_dir = sys.argv[1:]
    plot_control(input_dir)
