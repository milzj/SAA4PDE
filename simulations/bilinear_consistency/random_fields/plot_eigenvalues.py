
import matplotlib.pyplot as plt
from base.savefig import savefig
import itertools
import numpy as np
import os

m = 200
l = 0.15

outdir = "output/"
if not os.path.exists(outdir):
	os.makedirs(outdir)


indices = list(itertools.product(range(1,m), repeat=2))
indices = sorted(indices, key=lambda x:x[0]**2+x[1]**2)
indices = indices[0:m]

eigenvalues = np.ones(len(indices))

for i, pair in enumerate(indices):
	j, k = pair[0], pair[1]

	eigenvalue = .25*np.exp(-np.pi*(j**2+k**2)*l**2)
	eigenvalues[i] = eigenvalue


plt.scatter(range(m), eigenvalues, s=0.5)
plt.xlabel("Index of eigenvalue")
plt.ylabel("Eigenvalue")
plt.yscale("log")

savefig(outdir + "log_eigenvalues")
