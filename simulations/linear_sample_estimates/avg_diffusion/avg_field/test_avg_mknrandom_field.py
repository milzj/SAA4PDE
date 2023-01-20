"""The code can be used to empirically verify
the implementation of the "expected random
diffusion coefficient. The verification uses
Monte Carlo simulation. The code generates
a Monte Carlo convergence plots.
"""
import pytest


from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentparentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
sys.path.append(parentparentdir)

from mknrandom_field import MKNRandomField
from avg_mknrandom_field import avg_mknrandom_field

import os
outdir = "output/"
if not os.path.exists(outdir):
	os.makedirs(outdir)



n = 2*32
mesh = UnitSquareMesh(n,n)
V = FunctionSpace(mesh, "DG", 0)

avg_field_vec = avg_mknrandom_field(V).vector()[:]

rf = MKNRandomField(mean=0.0, var=1.0)
rf.function_space = V
u = Function(V)

def sampling_error(N):

    vs = []
    for i in range(N):

        sample = 2.0*np.random.rand(4)-1.0
        v = rf.sample(sample=sample)
        vs.append(v.vector().get_local())

    u.vector()[:] = np.mean(vs, axis=0)-avg_field_vec

    return norm(u, norm_type="L2")


n_conv = 50

errors = {}
medians = []
Ns = [2**n for n in range(4,12)]

for N in Ns:

    samples = []

    for n in range(n_conv):

        samples.append(sampling_error(N))

    errors[N] = samples
    medians.append(np.mean(samples))

fig, ax = plt.subplots()

for N in Ns:
    ax.scatter(N*np.ones(len(errors[N])), errors[N], marker="s")

x_vec = Ns
y_vec = medians
X = np.ones((len(x_vec), 2)); X[:, 1] = np.log(x_vec) # design matrix
x, residudals, rank, s = np.linalg.lstsq(X, np.log(y_vec), rcond=None)

rate = x[1]
constant = np.exp(x[0])

ax.plot(Ns, constant*Ns**rate, color="black", label=r"{} N^{}".format(constant,rate))

ax.set_xlabel("samples")
ax.set_ylabel("RMSE")
ax.set_xscale("log",base=2)
ax.set_yscale("log",base=2)

plt.legend()
fig.tight_layout()
fig.savefig(outdir + "mc_errors.png")

assert 0.45 < -rate < 0.55
