from dolfin import *
from dolfin_adjoint import *
import moola
import numpy as np

from stats import save_dict, compute_fem_errors

from random_semilinear_problem import RandomSemilinearProblem
from reference_sampler import ReferenceSampler
from discrete_sampler import DiscreteSampler
from random_problem import LocalReducedSAAFunctional

from solver_options import SolverOptions

from algorithms import SemismoothNewtonCG
from problem import NonsmoothFunctional
from random_problem import RieszMap
from prox import prox_box_l1

import warnings

class SAAProblems(object):


	def __init__(self, date=-1, experiment=None, beta=-1, Nref=-1, num_reps=48, experiment_name=None):

		self.date = date
		self.experiment = experiment
		self.num_reps = num_reps
		self.beta = beta
		self.Nref = Nref
		self.experiment_name = experiment_name

		self.mpi_size = MPI.comm_world.Get_size()
		self.mpi_rank = MPI.comm_world.Get_rank()
		self.LocalStats = {}

		self.seeds()
		self.divide_simulations()

		self.reference_sampler = ReferenceSampler(Nref)

	def seeds(self):

		Seeds = {}
		num_reps = self.num_reps

		seed = self.Nref

		for r in range(1, 1+num_reps):

			Seeds[r] = {}

			for e in self.experiment[("n_vec", "N_vec", "alpha_vec")]:
				n, N, alpha = e

				seed += 1*N+1
				Seeds[r][e] = seed

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


	def local_solve(self, sampler, n, number_samples, alpha, beta):

		set_working_tape(Tape())

		random_problem = RandomSemilinearProblem(n)

		u = Function(random_problem.control_space)
		# Update alpha and beta
		random_problem.alpha = alpha
		random_problem.beta = beta

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

		# compute gradient
		obj = nproblem.obj
		v_moola = sol["control"]
		obj(v_moola)
		grad = obj.derivative(v_moola).primal()

		return sol["control"].data, grad.data

	def criticality_measure(self, control_vec, gradient_vec, n, Nref, alpha, beta):
		"""Evaluate criticality measure without parallelization."""

		set_working_tape(Tape())

		random_problem = RandomSemilinearProblem(n)
		# Update alpha and beta
		random_problem.alpha = alpha
		random_problem.beta = beta

		sampler = self.reference_sampler

		u = Function(random_problem.control_space)
		u.vector()[:] = control_vec

		rf = LocalReducedSAAFunctional(random_problem, u, sampler, Nref, mpi_comm = MPI.comm_self)

		alpha = random_problem.alpha
		beta = random_problem.beta
		lb = random_problem.lb
		ub = random_problem.ub

		v_moola = moola.DolfinPrimalVector(u)

		nrf =  NonsmoothFunctional(rf, v_moola, alpha, beta, lb = lb, ub = ub)
		problem = moola.Problem(nrf)

		obj = problem.obj
		obj(v_moola)

		grad = obj.derivative(v_moola).primal()
		grad_vec = grad.data.vector().get_local()


		criticality_measures = []

		# reference crit measure

		g_vec = prox_box_l1(-(1.0/alpha)*grad_vec, lb, ub, beta/alpha)
		prox_grad = Function(random_problem.control_space)
		prox_grad.vector()[:] = g_vec
		criticality_measures.append(errornorm(u, prox_grad, degree_rise = 0))


		# reference crit measure with alpha = 0

		g_vec = prox_box_l1(control_vec-grad_vec, lb, ub, beta)
		prox_grad = Function(random_problem.control_space)
		prox_grad.vector()[:] = g_vec
		criticality_measures.append(errornorm(u, prox_grad, degree_rise = 0))

		# crit measure with alpha = 0
		g_vec = prox_box_l1(control_vec-gradient_vec, lb, ub, beta)
		prox_grad = Function(random_problem.control_space)
		prox_grad.vector()[:] = g_vec
		criticality_measures.append(errornorm(u, prox_grad, degree_rise = 0))


		return criticality_measures



	def simulate_mpi(self):

		LocalStats = {}
		mpi_rank = self.mpi_rank

		sampler = DiscreteSampler()

		beta = self.beta
		Nref = int(self.Nref)

		for r in self.Reps[mpi_rank]:
			E = {}

			for e in self.experiment[("n_vec", "N_vec", "alpha_vec")]:
				n, N, alpha = e
				print("n, N, alpha", n, N, alpha)

				seed = self.Seeds[r][e]
				sampler.seed = seed

				assert sampler.seed == seed
				if self.experiment_name.find("Synthetic") != -1:
					warnings.warn("Simulation output is synthetic." + 
						" This is a verbose mode used to generate test data for plotting purposes.")
					np.random.seed(seed)
					errors = np.random.randn(N)
					errors = abs(errors.mean())
				else:
					u_opt, grad_opt = self.local_solve(sampler, n, N, alpha, beta)
					errors = self.criticality_measure(u_opt.vector()[:], grad_opt.vector()[:], n, Nref, alpha, beta)
				E[e] = errors

			LocalStats[r] = E

		self.LocalStats = LocalStats



	def save_mpi(self, now, outdir):
		filename = now + "_mpi_rank=" + str(MPI.comm_world.Get_rank())
		save_dict(outdir, filename, self.LocalStats)



if __name__ == "__main__":

	import sys, os

	from experiments import Experiments
	from stats import save_dict

	if MPI.comm_world.Get_rank() == 0:

		# sys.argv
		date = sys.argv[1]
		experiment_name = sys.argv[2]
		beta_filename = str(sys.argv[3])
		Nref = int(sys.argv[4])

		# output dir
		outdir = "output/Experiments/" + experiment_name + "_" + date

		if not os.path.exists(outdir):
			os.makedirs(outdir)

		experiment = Experiments()(experiment_name)

		# save experiment
		filename = experiment_name
		save_dict(outdir, filename, {experiment_name: experiment})

		# update beta
		beta = np.loadtxt("output/" + beta_filename + ".txt")

		## save relative path + filename of control
		np.savetxt(outdir  + "/" + filename  + "_filename.txt", np.array([outdir]), fmt = "%s")


	else:
		date = None
		experiment_name = None
		outdir = None
		experiment = None
		beta = None
		Nref = None


	# bcast
	date = MPI.comm_world.bcast(date, root=0)
	experiment_name = MPI.comm_world.bcast(experiment_name, root=0)
	outdir = MPI.comm_world.bcast(outdir, root=0)
	experiment = MPI.comm_world.bcast(experiment, root=0)
	beta = MPI.comm_world.bcast(beta, root=0)
	Nref = MPI.comm_world.bcast(Nref, root=0)


	saa_problems = SAAProblems(date=date,experiment=experiment, experiment_name=experiment_name, beta=beta, Nref = Nref)

	saa_problems.simulate_mpi()
	saa_problems.save_mpi(date, outdir)
