from dolfin import *

import matplotlib.pyplot as plt
import sys
import numpy as np



input_dir = "output/" + sys.argv[1]
input_date = input_dir.split("=")
input_date = input_date[-1]


input_filename = input_dir
u_vec = np.loadtxt(input_filename + ".txt")

n = int(input_date)


mesh = UnitSquareMesh(n, n)
U = FunctionSpace(mesh, "DG", 0)
u = Function(U)
u.vector()[:] = u_vec

plot(u)
plt.savefig(input_dir + ".pdf", bbox_inches="tight")
plt.close()

