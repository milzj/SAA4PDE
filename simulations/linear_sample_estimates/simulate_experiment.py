from dolfin import *
from dolfin_adjoint import *
import moola
import numpy as np

from stats import save_dict, compute_fem_errors

from random_linear_problem import RandomLinearProblem
from discrete_sampler import DiscreteSampler
from random_problem import LocalReducedSAAFunctional
from solver_options import SolverOptions

from algorithms import SemismoothNewtonCG
from problem import NonsmoothFunctional
from random_problem import RieszMap

from experiments import Experiments

class SAAProblems(object):


	def __init__(self, input_filename, num_reps=48):

		self.degree_rise = 0
		self.num_reps = num_reps

		self.input_filename = input_filename

		self.mpi_size = MPI.comm_world.Get_size()
		self.mpi_rank = MPI.comm_world.Get_rank()
		self.LocalStats = {}


		self.load_reference_solution()
		self.n_vec_N_vec()
		self.seeds()
		self.divide_simulations()

	def load_reference_solution(self):

		fn_split = self.input_filename.split("=")
		n_ref = int(fn_split[-1])
		N_ref = int(fn_split[-2].split("_")[0])

		u_ref_vec = np.loadtxt("output/" + self.input_filename + ".txt")

		mesh_ref = UnitSquareMesh(MPI.comm_self, n_ref, n_ref)
		U_ref = FunctionSpace(mesh_ref, "DG", 0)
		u_ref = Function(U_ref)
		u_ref.vector()[:] = u_ref_vec

		self.n_ref = n_ref
		self.N_ref = N_ref
		self.u_ref = u_ref
		self.mesh_ref = mesh_ref

	def n_vec_N_vec(self):

		n_vec = [8, 12, 16, 24, 36, 48, 72]
		N_vec = [n**2 for n in n_vec]

		self.__n_vec = np.array(n_vec)
		self.__N_vec = np.array(N_vec)

	def seeds(self):

		Seeds = {}
		num_reps = self.num_reps

		seed = 1

		for r in range(1, 1+num_reps):

			Seeds[r] = {}

			for n in self.__n_vec:
				Seeds[r][n] = {}

				for N in self.__N_vec:
					Seeds[r][n][N] = seed
					seed += 4*N+1

		if np.__version__ == '1.12.1':
			period = 2**32-1
		else:
			period = 2**32-1

		assert seed <= period, "Period of random number generator (might) too small."

		self.Seeds = Seeds

	def divide_simulations(self):

		mpi_size = self.mpi_size
		mpi_rank = self.mpi_rank
		num_reps = self.num_reps

		reps = np.array(range(1,1+num_reps))

		Reps = np.array_split(reps, mpi_size)
		self.Reps = Reps


	def local_solve(self, sampler, n, number_samples):

		set_working_tape(Tape())

		random_problem = RandomLinearProblem(n, sampler)

		u = Function(random_problem.control_space)

		rf = LocalReducedSAAFunctional(random_problem, u, sampler, number_samples, mpi_comm = MPI.comm_self)

		assert rf.mpi_size == 1

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

		return sol["control"].data


	def simulate_mpi(self, experiment_name):

		LocalStats = {}

		degree_rise = self.degree_rise
		mpi_rank = self.mpi_rank

		u_ref = self.u_ref
		mesh_ref = u_ref.function_space().mesh()
		grid_points = int(self.N_ref**(1/4))
		sampler = DiscreteSampler(grid_points=grid_points)

		experiments = Experiments()
		experiment = experiments(experiment_name)

		for r in self.Reps[mpi_rank]:

			E = {}
			_n = -1
			for e in experiment[("n_vec", "N_vec")]:
				n, N = e

				if n != _n:
					E[n] = {}

				seed = self.Seeds[r][n][N]

				sampler.seed = seed
				assert sampler.seed == seed

				u_opt = self.local_solve(sampler, n, N)
				errors = compute_fem_errors(u_ref, u_opt, degree_rise = degree_rise, mesh=mesh_ref)

				E[n][N] = errors

				_n = n



			LocalStats[r] = E


		self.LocalStats = LocalStats


	def save_mpi(self, now, outdir):

		filename = now + "_mpi_rank=" + str(MPI.comm_world.Get_rank())
		save_dict(outdir, filename, self.LocalStats)



if __name__ == "__main__":

	import sys, os
	from datetime import datetime

	experiment_name = sys.argv[1]
	input_filename = sys.argv[2]

	saa_problems = SAAProblems(input_filename)

	if MPI.comm_world.Get_rank() == 0:

		outdir = input_filename.split("/")[0]
		now = datetime.now().strftime("%d-%B-%Y-%H-%M-%S")
		outdir = "output/" + outdir + "/" + now

		if not os.path.exists(outdir):
			os.makedirs(outdir)

	else:
		now = None
		outdir = None

	now = MPI.comm_world.bcast(now, root=0)
	outdir = MPI.comm_world.bcast(outdir, root=0)

	saa_problems.simulate_mpi(experiment_name)
	saa_problems.save_mpi(now, outdir)
