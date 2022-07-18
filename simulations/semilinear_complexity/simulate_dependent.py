from dolfin import *
from dolfin_adjoint import *
import numpy as np

# random problem and options
from random_semilinear_problem import RandomSemilinearProblem
from solver_options import SolverOptions
from discrete_sampler import DiscreteSampler

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
N_vec = str(sys.argv[2])
N_vec = [int(x) for x in N_vec.split(",")]

if MPI.comm_world.Get_rank()  == 0:

	# dir for simulation output
	now = str(sys.argv[3])
	import os
	outdir = "output/"
	outdir = outdir+"Dependent_Simulation_n="+str(n)+"_date={}".format(now)
	if not os.path.exists(outdir):
		os.makedirs(outdir)

# update beta
beta_filename = str(sys.argv[4])
beta = np.loadtxt("output/" + beta_filename + ".txt")


Seeds = [0,0]
Seeds[0] = 5*10000
Seeds[1] = Seeds[0] + 5*10000

for seed in Seeds:

	for N in N_vec:

		sampler = DiscreteSampler()
		sampler._seed = seed
		random_problem = RandomSemilinearProblem(n)
		random_problem.beta = beta

		u = Function(random_problem.control_space)

		#rf = LocalReducedSAAFunctional(random_problem, u, sampler, N, mpi_comm = MPI.comm_self)
		rf = GlobalReducedSAAFunctional(random_problem, u, sampler, N)

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

		if MPI.comm_world.Get_rank()  == 0:

			filename = outdir + "/"+ now + "_solution_N={}_n={}_seed={}".format(N,n,seed)
			np.savetxt(filename + ".txt", u_opt.vector().get_local())

		if True and MPI.comm_world.Get_rank()  == 0:

			surface_function(u_opt, n)
			plt.savefig(outdir + "/" + "surface_solution" +  "_solution_N={}_n={}_seed={}".format(N,n,seed) + ".pdf", bbox_inches="tight")
			plt.close()

			p = plot(u_opt)
			plt.colorbar(p)
			plt.savefig(outdir + "/" + "contour" + "_solution_N={}_n={}_seed={}".format(N,n,seed) + ".pdf", bbox_inches="tight")
			plt.close()

			file = File(outdir + "/" + "piecewise" + "_solution_N={}_n={}_seed={}".format(N,n,seed) + ".pvd")
			file << u_opt

