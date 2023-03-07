from dolfin import *
from dolfin_adjoint import *
import numpy as np

import os

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentparentdir = os.path.dirname(parentdir)
sys.path.append(parentparentdir)


# random problem and options
from random_linear_problem import RandomLinearProblem
from discrete_sampler import DiscreteSampler
from solver_options import SolverOptions

# algorithms
from random_problem import LocalReducedSAAFunctional, GlobalReducedSAAFunctional
from algorithms import SemismoothNewtonCG
from problem import NonsmoothFunctional
from random_problem import RieszMap
import moola

# dir for simulation output
n = int(sys.argv[1])
N = int(sys.argv[2])
Nsamples = int(sys.argv[3])
outdir = str(sys.argv[4])
_filename = str(sys.argv[5])

if not os.path.exists(outdir):
	os.makedirs(outdir)


Seeds = []
seed = N

for i in range(Nsamples):
	Seeds.append(seed)
	seed += 4*N+1

i = 0
for seed in Seeds:

	sampler = DiscreteSampler()
	sampler._seed = seed

	filename = _filename.format(i)
	i += 1

	random_problem = RandomLinearProblem(n, sampler)

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

	np.savetxt(outdir + "/" + filename + ".txt", u_opt.vector().get_local())

	file = File(outdir + "/" + filename  + ".pvd")
	file << u_opt

