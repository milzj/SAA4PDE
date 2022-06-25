from dolfin import *
from dolfin_adjoint import *
import numpy as np

# random problem and options
from random_bilinear_problem import RandomBilinearProblem
from solver_options import SolverOptions
from bilinear_sampler import BilinearSampler

# stats and plotting
from stats import save_dict
from datetime import datetime

# algorithms
from random_problem import LocalReducedSAAFunctional, GlobalReducedSAAFunctional
from algorithms import SemismoothNewtonCG
from problem import NonsmoothFunctional
from random_problem import RieszMap
import moola

n = int(sys.argv[1])
N_vec = str(sys.argv[2])
N_vec = [int(x) for x in N_vec.split(",")]

# dir for simulation output
now = str(sys.argv[3])
import os
outdir = "output/"
outdir = outdir+"Dependent_Simulation_n="+str(n)+"_date={}".format(now)
if not os.path.exists(outdir):
	os.makedirs(outdir)


Seeds = [10000]
for i in range(len(N_vec)-1):
	Seeds.append(Seeds[i] + 5*10000)

for seed in Seeds:

	for N in N_vec:

		sampler = BilinearSampler()
		sampler._seed = seed
		filename = outdir + "/"+ now + "_solution_N={}_n={}_seed={}".format(N,n,seed)
		random_problem = RandomBilinearProblem(n, sampler)

		u = Function(random_problem.control_space)

		rf = LocalReducedSAAFunctional(random_problem, u, sampler, N, mpi_comm = MPI.comm_self)

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

		np.savetxt(filename + ".txt", u_opt.vector().get_local())

		file = File(outdir + "/" + "piecewise" + "_solution_N={}_n={}_seed={}".format(N,n,seed) + ".pvd")
		file << u_opt

