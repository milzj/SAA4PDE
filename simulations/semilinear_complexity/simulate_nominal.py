from dolfin import *
from dolfin_adjoint import *
import numpy as np

# random problem and options
from random_semilinear_problem import RandomSemilinearProblem
from solver_options import SolverOptions
from reference_sampler import ReferenceSampler

# stats and plotting
from stats import save_dict
from stats.surface_function import surface_function
from datetime import datetime

import matplotlib.pyplot as plt

# algorithms
from random_problem import LocalReducedSAAFunctional, GlobalReducedSAAFunctional
from algorithms import SemismoothNewtonCG
from problem import NonsmoothFunctional
from random_problem import RieszMap
import moola


n = int(sys.argv[1])
now = str(sys.argv[2])

import os
outdir = "output/"
outdir = outdir+"Nominal_Simulation_n="+str(n)+"_date={}".format(now)
if not os.path.exists(outdir):
	os.makedirs(outdir)

sampler = ReferenceSampler(Nref=1)
random_problem = RandomSemilinearProblem(n)

# update beta
beta_filename = str(sys.argv[3])
beta = np.loadtxt("output/" + beta_filename + ".txt")
random_problem.beta = beta

u = Function(random_problem.control_space)

J = random_problem(u, sample=sampler.mean)
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
filename = outdir + "/" + now + "_nominal_solution_n={}".format(n)
np.savetxt(filename + ".txt", u_opt.vector().get_local())
## save relative path + filename of control
relative_path = filename.split("/")
relative_path = relative_path[1] + "/"+ relative_path[2]
np.savetxt(filename + "_filename.txt", np.array([relative_path]), fmt = "%s")

sol.pop("lbfgs")
sol.pop("control")
sol.pop("precond")
filename = now + "_nominal_solution_n={}_exit_data".format(n)
save_dict(outdir, filename, sol)

# plotting
surface_function(u_opt, n)
plt.savefig(outdir + "/" + "surface_solution" +  "_nominal_n={}".format(n) + ".pdf", bbox_inches="tight")
plt.close()

p = plot(u_opt)
plt.colorbar(p)
plt.savefig(outdir + "/" + "contour_solution" +  "_nominal_n={}".format(n) + ".pdf", bbox_inches="tight")
plt.close()

file = File(outdir + "/" + "solution" +  "_nominal_n={}".format(n) + ".pvd")
file << u_opt





