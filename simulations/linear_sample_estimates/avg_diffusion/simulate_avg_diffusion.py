"""The code can be used to solve

min (1/2) norm(S(u)-yd)**2 + (alpha/2)**norm(u)**2 + beta*norm(u,L1)

where S(u) solves the PDE

- nabla avg(kappa) nabla y = u

and avg(kappa) is the expectation of a certain random diffusion
coefficient kappa.

"""



from dolfin import *
from dolfin_adjoint import *
import numpy as np

import sys, os
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

# random problem and options
from random_linear_problem import RandomLinearProblem
from solver_options import SolverOptions
from reference_sampler import ReferenceSampler

from avg_diffusion import AVGMKNRandomField

# stats and plotting
from stats import save_dict
from stats.surface_function import surface_function
from datetime import datetime
from stats import figure_style

import matplotlib.pyplot as plt

# algorithms
from random_problem import LocalReducedSAAFunctional, GlobalReducedSAAFunctional
from algorithms import SemismoothNewtonCG
from problem import NonsmoothFunctional
from random_problem import RieszMap
import moola


n = 144

import os
outdir = "output/"
outdir = outdir+"Avg_Simulation_n="+str(n)
if not os.path.exists(outdir):
	os.makedirs(outdir)


random_problem = RandomLinearProblem(n, None)
avg_kappa = AVGMKNRandomField()
avg_kappa.function_space = random_problem.V
# Update of diffusion coefficient
random_problem.kappa = avg_kappa

u = Function(random_problem.control_space)

sampler_mean = np.zeros(4)
J = random_problem(u, sampler_mean)
rf = ReducedFunctional(J, Control(u))

alpha = random_problem.alpha
beta = random_problem.beta
lb = random_problem.lb
ub = random_problem.ub

riesz_map = RieszMap(random_problem.control_space)
v_moola = moola.DolfinPrimalVector(u, riesz_map = riesz_map)
v_moola.zero()

nrf =  NonsmoothFunctional(rf, v_moola, alpha, beta, lb = lb, ub = ub)

nproblem = moola.Problem(nrf)

solver = SemismoothNewtonCG(nproblem, v_moola, options=SolverOptions().options)

sol = solver.solve()

u_opt = sol["control"].data

# save solution and exit message
filename = outdir + "/" +  "avg_solution_n={}".format(n)
np.savetxt(filename + ".txt", u_opt.vector().get_local())
## save relative path + filename of control
relative_path = filename.split("/")
relative_path = relative_path[1] + "/"+ relative_path[2]
np.savetxt(filename + "_filename.txt", np.array([relative_path]), fmt = "%s")

sol.pop("lbfgs")
sol.pop("control")
sol.pop("precond")
filename = "avg_solution_n={}_exit_data".format(n)
save_dict(outdir, filename, sol)

# plotting
fig, ax = surface_function(u_opt, n)
zlim_l = -6.
zlim_u = 6
ax.set_zlim([zlim_l, zlim_u])
plt.savefig(outdir + "/" + "surface_solution" +  "_avg_n={}".format(n) + ".pdf", bbox_inches="tight")
plt.close()

p = plot(u_opt)
plt.colorbar(p)
plt.savefig(outdir + "/" + "contour_solution" +  "_avg_n={}".format(n) + ".pdf", bbox_inches="tight")
plt.close()

file = File(outdir + "/" + "solution" +  "_avg_n={}".format(n) + ".pvd")
file << u_opt





