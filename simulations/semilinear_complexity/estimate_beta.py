from dolfin import *
from dolfin_adjoint import *
import numpy as np

from stats import save_dict

from random_semilinear_problem import RandomSemilinearProblem
from base import norm_Linf

from discrete_sampler import DiscreteSampler
from random_problem import LocalReducedSAAFunctional, GlobalReducedSAAFunctional

from random_problem import RieszMap
from problem import NonsmoothFunctional

import moola

def estimate_beta(n, N):
	"""Heuristic to obtain a scaling parameter for the L1 norm."""

	sampler = DiscreteSampler()
	sampler._seed = 1
	random_problem = RandomSemilinearProblem(n)

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

	problem = moola.Problem(nrf)

	obj = problem.obj
	obj(v_moola)

	grad = obj.derivative(v_moola).primal()

	grad_norm_inf = norm_Linf(grad.data)

	return grad_norm_inf

if __name__ == "__main__":


	import os, sys

	from base import signif

	n = int(sys.argv[1])
	N = int(sys.argv[2])
	now = str(sys.argv[3])
	ratio = np.float64(sys.argv[4])

	outdir = "output/"
	if not os.path.exists(outdir):
		os.makedirs(outdir)

	outdir = outdir+"Beta_Simulation_n="+str(n)+"_N="+str(N)+"_date={}".format(now)
	os.makedirs(outdir)

	grad_norm_inf = estimate_beta(n, N)

	filename = outdir + "/" + now + "_beta_N={}_n={}".format(N,n)
	beta = signif(ratio*grad_norm_inf, precision = 3)
	np.savetxt(filename + ".txt", np.array([beta]), fmt="%s")

	# save relative path + filename of control
	relative_path = filename.split("/")
	relative_path = relative_path[1] + "/"+ relative_path[2]
	np.savetxt(filename + "_filename.txt", np.array([relative_path]), fmt = "%s")

