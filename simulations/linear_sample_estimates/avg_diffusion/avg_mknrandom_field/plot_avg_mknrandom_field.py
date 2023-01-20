"""Plot the expectation of the diffusion coefficient."""


from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

from avg_mknrandom_field import avg_mknrandom_field

import os
outdir = "output/"
if not os.path.exists(outdir):
	os.makedirs(outdir)

n = 144
mesh = UnitSquareMesh(n,n)
V = FunctionSpace(mesh, "CG", 1)
v = avg_mknrandom_field(V)

c = plot(v)
plt.colorbar(c)
plt.tight_layout()
plt.savefig(outdir + "avg_mknrandom_field.pdf")
plt.savefig(outdir + "avg_mknrandom_field.png")
