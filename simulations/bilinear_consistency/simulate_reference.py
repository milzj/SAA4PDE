from dolfin import *
from dolfin_adjoint import *
import moola
import numpy as np

from stats import save_dict

from random_bilinear_problem import RandomBilinearProblem
from solver_options import SolverOptions

from reference_sampler import ReferenceSampler
from random_problem import LocalReducedSAAFunctional, GlobalReducedSAAFunctional

from algorithms import SemismoothNewtonCG
from problem import NonsmoothFunctional
from random_problem import RieszMap


def simulate_reference(n, N):

	sampler = ReferenceSampler(N)
	random_problem = RandomBilinearProblem(n, sampler)

	u = Function(random_problem.control_space)

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

	return sol

if __name__ == "__main__":


	mpi_rank = MPI.comm_world.Get_rank()

	if mpi_rank == 0:

		import os, sys

		n = int(sys.argv[1])
		N = int(sys.argv[2])
		now = str(sys.argv[3])

		outdir = "output/"
		if not os.path.exists(outdir):
			os.makedirs(outdir)

		outdir = outdir+"Reference_Simulation_n="+str(n)+"_N="+str(N)+"_date={}".format(now)
		os.makedirs(outdir)


	else:
		now = None
		outdir = None
		n = None
		N = None


	now = MPI.comm_world.bcast(now, root=0)
	outdir = MPI.comm_world.bcast(outdir, root=0)

	n = MPI.comm_world.bcast(n, root=0)
	N = MPI.comm_world.bcast(N, root=0)

	sol = simulate_reference(n, N)
	u_opt = sol["control"].data

	if mpi_rank == 0:
		# save control
		filename = outdir + "/" + now + "_reference_solution_mpi_rank={}_N={}_n={}".format(mpi_rank, N,n)
		np.savetxt(filename + ".txt", u_opt.vector().get_local())

		# save relative path + filename of control
		relative_path = filename.split("/")
		relative_path = relative_path[1] + "/"+ relative_path[2]
		np.savetxt(filename + "_filename.txt", np.array([relative_path]), fmt = "%s")

		# save control as pvd
		file = File(MPI.comm_self, filename + ".pvd")
		file << u_opt

		# save exit data
		filename = now + "_reference_solution_mpi_rank={}_N={}_n={}_exit_data".format(mpi_rank, N,n)
		sol.pop("lbfgs")
		sol.pop("control")
		sol.pop("precond")
		save_dict(outdir, filename, sol)

