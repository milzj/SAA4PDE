# -*- coding: utf-8 -*-
"""
This file is used to empirically verify the Monte Carlo convergence rate
for computing the mean. The Monte Carlo scheme is applied to a list of
distributions. 

The purpose of the file is to check whether a small number of replications 
(say 48) is sufficient to observe the Monte Carlo convergence rate 1/sqrt(N).
"""

import numpy as np
import matplotlib.pyplot as plt

def signif(beta):
	"Returns three significant figures of input float."
	return np.format_float_positional(beta, precision=3, unique=True, trim="k", fractional=False)

def lsqs_label(constant, rate, variable):
  "Least squares label"
  constant = signif(constant)
  rate = signif(rate)
  return r"${}\cdot {}^{}$".format(constant, variable, "{"+ str(rate)+"}")

# Sample size
N_vec  = [2**n for n in range(4, 14)]
reps = 48

a = 1.0
b = 1.0
distributions = [lambda x : 10*np.random.randn(x), 
                 lambda x : 2.0*np.random.rand(x)-1.0, 
                 lambda x : np.random.beta(a,b, x)-a/(a+b),
                 lambda x : np.random.lognormal(size=x)-np.exp(.5)]
distributions_label = {}
dlabel = ["normal distribution", "uniform [-1,1]", "beta", "lognormal"]
i = 0
for distribution in distributions:
  distributions_label[distribution] = dlabel[i]
  i+=1

# Compute errors
errors = {}
for distribution in distributions:
  errors[distribution] = {}
  for r in range(reps):
    errors[distribution][r] = {}
    for N in N_vec:
      Z = distribution(N)
      errors[distribution][r][N] = np.abs(Z.mean())
    
# Postprocessing
stats = {}
for distribution in distributions:
  stats[distribution] = {}
  for N in N_vec:
    _errors = [errors[distribution][r][N] for r in range(reps)]
    stats[distribution][N] = np.mean(_errors)

# Compute least squares fit
lsqs_rates = {}
for distribution in distributions:
  lsqs_rates[distribution] = {}
  X = np.ones((len(N_vec), 2));
  X[:, 1] = np.log(N_vec)
  e = [stats[distribution][N] for N in N_vec]
  x, residudals, rank, s = np.linalg.lstsq(X, np.log(e), rcond=None)
  lsqs_rates[distribution] = [x[1], np.exp(x[0])]

# Plot realizations and least squares fit
for distribution in distributions:
  fig, ax = plt.subplots()
  for r in range(reps):
    ax.scatter(N_vec, [errors[distribution][r][N] for N in N_vec], color="black", s=0.1)

  s, t = lsqs_rates[distribution]
  y = t*N_vec**s
  ax.plot(N_vec, y, color="black", linestyle="--", label=lsqs_label(t, s, "N"))

  ax.set_xscale("log")
  ax.set_yscale("log")
  ax.set_xlabel("N")
  ax.set_ylabel("error")
  ax.set_title(distributions_label[distribution])
  plt.legend()
  plt.savefig(distributions_label[distribution]+".pdf", bbox_inches="tight")
