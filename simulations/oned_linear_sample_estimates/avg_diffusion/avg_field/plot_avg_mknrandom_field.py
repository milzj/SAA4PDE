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
mesh = UnitIntervalMesh(n)
V = FunctionSpace(mesh, "DG", 0)
v = avg_mknrandom_field(V)

plot(v)
plt.tight_layout()
plt.savefig(outdir + "avg_mknrandom_field.pdf")
plt.savefig(outdir + "avg_mknrandom_field.png")
