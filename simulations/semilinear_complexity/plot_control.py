from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import sys

from stats import figure_style
from stats.surface_function import surface_function



input_dir = "output/" + sys.argv[1]
n = input_dir.split("_n=")[-1].split(".")[0].split("_")[0]
n = int(n)

input_filename = input_dir
u_vec = np.loadtxt(input_filename + ".txt")

mesh = UnitSquareMesh(n, n)
U = FunctionSpace(mesh, "DG", 0)
u = Function(U)
u.vector()[:] = u_vec


fig, ax, = surface_function(u, n, cmap=cm.coolwarm)
zlim_l = -10.
zlim_u = 10
ax.set_zlim([zlim_l, zlim_u])
ax.set_zticks(np.arange(zlim_l, zlim_u+4, 4.0))
plt.savefig(input_dir + "_surface" + ".pdf", bbox_inches="tight")
