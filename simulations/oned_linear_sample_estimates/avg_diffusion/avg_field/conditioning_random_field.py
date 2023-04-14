"""
We emprically estimate the standard deviation

    sqrt( mean[ norm[kappa-mean(kappa)]**2 ])

where kappa is a random diffusion coefficient
and norm is the L^infinity norm.

Moreover, we estimate the mean of the condition
number of kappa

    max_x kappa(xi)(x) over min_x kappa(xi)(x).

"""

from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentparentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
sys.path.append(parentparentdir)

from stats import figure_style
from random_field import RandomField
from avg_mknrandom_field import avg_mknrandom_field
from discrete_sampler import DiscreteSampler

import os
outdir = "output/"
if not os.path.exists(outdir):
	os.makedirs(outdir)

now = datetime.now().strftime("%d-%m-%Y-%H:%M:%S")

n = 1024 # number of triangles
grid_points = int(np.sqrt(1024))
n_conv = 48 # number of replications
N = 100 # number of samples (per replication)
vars = np.linspace(1, 8, 15) # standard deviations

mesh = UnitIntervalMesh(n)
V = FunctionSpace(mesh, "DG", 0)
discrete_sampler = DiscreteSampler()


def empirical_estimates(rf, N):

    vs = []
    cs = []
    ks = []
    us = []


    for i in range(N):

        sample = discrete_sampler.sample(sample_index = i, N=1)

        v = rf.sample(sample=sample)
        v_vec = v.vector().get_local()
        vs.append(v_vec)
        cs.append(np.max(v_vec)/np.min(v_vec))
        ks.append(np.min(v_vec))
        us.append(np.max(v_vec))

    emp_mean = np.mean(vs, axis=0)

    ws = []
    for i in range(N):
        w = np.linalg.norm(vs[i] - emp_mean, ord=np.inf)
        ws.append(w**2)

    return np.sqrt(np.mean(ws)), np.mean(cs, axis=0), np.min(ks, axis=0), np.max(us, axis=0)


errors_var = {}
medians_var = []
errors_cond = {}
medians_cond = []
errors_min = {}
medians_min = []
errors_max = {}
medians_max = []

# simulations

for var in vars:

    samples_var = []
    samples_cond = []
    samples_min = []
    samples_max = []

    rf = RandomField(mean=0.0, var=var)
    rf.function_space = V

    for n in range(n_conv):

        v, w, x, y = empirical_estimates(rf, N)
        samples_var.append(v)
        samples_cond.append(w)
        samples_min.append(x)
        samples_max.append(y)

    errors_var[var] = samples_var
    errors_cond[var] = samples_cond
    errors_min[var] = samples_min
    errors_max[var] = samples_max
    medians_var.append(np.mean(samples_var))
    medians_cond.append(np.mean(samples_cond))
    medians_min.append(np.mean(samples_min))
    medians_max.append(np.mean(samples_max))



# plotting
fig, ax = plt.subplots()

for var in vars:
    # ax.scatter(var*np.ones(len(errors_var[var])), errors_var[var], marker="s", s=2, color="tab:blue")
    ax.scatter(var*np.ones(len(errors_cond[var])), errors_cond[var], marker="o", s=2, color="tab:orange")
    ax.scatter(var*np.ones(len(errors_max[var])), errors_max[var], marker="v", s=2, color="tab:blue")
    ax.scatter(var*np.ones(len(errors_min[var])), errors_min[var], marker="d", s=2, color="tab:brown")

for var in vars:
    # ax.scatter(vars, medians_var, marker="s", color="tab:blue", label=r"standard deviations")
    ax.scatter(vars, medians_cond, marker="o",color="tab:orange", label=r"$\frac{\max_{1 \leq i \leq n_{h^*}}\kappa_{\sigma}(\xi)(x_i)}{\min_{1 \leq i \leq n_{h^*}}\kappa_{\sigma}(\xi)(x_i)}$")
    ax.scatter(vars, medians_max, marker="v", color="tab:blue", label=r"$\max_{1 \leq i \leq n_{h^*}}\kappa_{\sigma}(\xi)(x_i)$")
    ax.scatter(vars, medians_min, marker="d", color="tab:brown", label=r"$\min_{1 \leq i \leq n_{h^*}}\kappa_{\sigma}(\xi)(x_i)$")


ax.set_xlabel(r"$\sigma$")
#ax.set_xscale("log",base=10)
ax.set_yscale("log",base=10)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys(), loc="upper left", borderpad=0.1,handlelength=1.0,handletextpad=0.0,borderaxespad=0.1, columnspacing=0.1)
plt.legend(by_label.values(), by_label.keys(), loc="upper left")

fig.tight_layout()
fig.savefig(outdir + "empirical_variance_{}.png".format(now))
fig.savefig(outdir + "empirical_variance_{}.pdf".format(now))

