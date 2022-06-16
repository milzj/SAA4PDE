from dolfin import *
from dolfin_adjoint import *
import moola
import numpy as np

from stats import save_dict

import sys, os

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentparentdir = os.path.dirname(parentdir)
sys.path.append(parentparentdir)

from random_linear_problem import RandomLinearProblem
from parallel_sampler import ParallelSampler
from random_problem import LocalReducedSAAFunctional, GlobalReducedSAAFunctional

from algorithms import SemismoothNewtonCG
from problem import NonsmoothFunctional
from random_problem import RieszMap


def simulate_solution(n, number_samples, seed):

	N = number_samples

	sampler = ParallelSampler(N,seed)
	random_problem = RandomLinearProblem(n, sampler)

	u = Function(random_problem.control_space)

	rf = GlobalReducedSAAFunctional(random_problem, u, sampler, number_samples)

	alpha = random_problem.alpha
	beta = random_problem.beta
	lb = random_problem.lb
	ub = random_problem.ub

	riesz_map = RieszMap(random_problem.control_space)
	v_moola = moola.DolfinPrimalVector(u, riesz_map = riesz_map)
	v_moola.zero()
	nrf =  NonsmoothFunctional(rf, v_moola, alpha, beta, lb = lb, ub = ub)

	nproblem = moola.Problem(nrf)

	solver = SemismoothNewtonCG(nproblem, v_moola, options={'gtol': 1e-8,
		                                                   'maxiter': 20,
		                                                   'display': 3,
		                                                   'ncg_hesstol': 0,
								   'line_search': 'fixed', "correction_step": False})


	sol = solver.solve()

	return sol

if __name__ == "__main__":


	mpi_rank = MPI.comm_world.Get_rank()

	if mpi_rank == 0:

		import os, sys

		n = int(sys.argv[1])
		seed = int(sys.argv[2])
		now = str(sys.argv[3])
		N = n**2

		outdir = "output/Simulation"

		if not os.path.exists(outdir):
			os.makedirs(outdir)


	else:
		now = None
		outdir = None
		n = None
		N = None
		seed = None

	now = MPI.comm_world.bcast(now, root=0)
	outdir = MPI.comm_world.bcast(outdir, root=0)

	n = MPI.comm_world.bcast(n, root=0)
	N = MPI.comm_world.bcast(N, root=0)
	seed = MPI.comm_world.bcast(seed, root=0)

	sol = simulate_solution(n, N, seed)
	u_opt = sol["control"].data

	if mpi_rank == 0:
		# save control
		filename = outdir + "/" + "solution_n={}".format(n)
		np.savetxt(filename + ".txt", u_opt.vector().get_local())

		# save relative path + filename of control
		relative_path = filename.split("/")
		relative_path = relative_path[1] + "/"+ relative_path[2]
		np.savetxt(filename + "_filename.txt", np.array([relative_path]), fmt = "%s")

		# save control as pvd
		file = File(MPI.comm_self, filename + ".pvd")
		file << u_opt

		# save exit data
		filename = now + "solution_n={}_exit_data".format(n)
		sol.pop("lbfgs")
		sol.pop("control")
		sol.pop("precond")
		save_dict(outdir, filename, sol)

