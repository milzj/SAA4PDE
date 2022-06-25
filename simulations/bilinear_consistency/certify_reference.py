"""The file is used to check optimality of the reference solution.

This code only depends on the implementation of prox and the random field.
"""

from dolfin import *
from dolfin_adjoint import *
import moola

from stats import save_dict

import numpy as np
from random_fields.random_diffusion_coefficient import RandomDiffusionCoefficient
from random_fields.random_nonnegative_coefficient import RandomNonnegativeCoefficient
from prox import prox_box_l1

set_log_level(30)

def certify_reference(u, mesh, N, n, sampler, exit_data, input_filename):
	"""Computes a criticality measure to check whether u is a solution.

	We check whether the norm of u-prox(-1/alpha * nabla f(u)) is less
	than or equal to alpha*grad_norm, where grad_norm is the norm of the
	normal map at the final iterate.

	A file with ending "certify_reference" is printed to the directory
	of the solution u.

	The code should not be run in parallel.
	"""

	alpha = 1e-3
	ub = 100.
	lb = 0.
	beta = 0.0

	V = FunctionSpace(mesh, "CG", 1)
	W = FunctionSpace(mesh, "DG", 0)

	y = Function(V)
	v = TestFunction(V)

	yd_expr = Expression("sin(2.0*pi*x[0])*sin(2.0*pi*x[1])*exp(2.0*x[0])/6.0", \
			degree=0, domain=mesh, mpi_comm=mesh.mpi_comm())

	yd = Function(W)
	yd.interpolate(yd_expr)

	bcs = DirichletBC(V, 0.0, "on_boundary")

	kappa = RandomDiffusionCoefficient(mpi_comm=mesh.mpi_comm())
	kappa.function_space = W
	g = RandomNonnegativeCoefficient(mpi_comm=mesh.mpi_comm())
	g.function_space = W

	f = Constant(-1.0)

	J = 0.0
	for i in range(N):

		sample = sampler.sample(i)
		a, b = np.split(sample, 2)
		kappas = kappa.sample(a)
		gs = g.sample(b)

		F = (kappas*inner(grad(y), grad(v)) + gs*y*u*v - f*v) * dx
		solve(F == 0, y, bcs)

		j = assemble(0.5 * inner(y - yd, y - yd) * dx)
		J += 1.0/(1.0+i)*(j-J)

	rf = ReducedFunctional(J, Control(u))
	v_moola = moola.DolfinPrimalVector(u)

	problem = MoolaOptimizationProblem(rf)

	obj = problem.obj
	obj(v_moola)

	gradient = obj.derivative(v_moola).primal()
	grad_vec = gradient.data.vector().get_local()

	grad_vec = prox_box_l1(-(1.0/alpha)*grad_vec, lb, ub, beta/alpha)

	prox_grad = Function(W)
	prox_grad.vector()[:] = grad_vec

	error = errornorm(u, prox_grad, degree_rise = 0)

	grad_norm = exit_data["grad_norm"]
	certify_solution = {
		"is_solution": alpha*error <= grad_norm,
		"alpha*criticality_measure": alpha*error,
		"grad_norm": grad_norm,
		"alpha": alpha,
		"criticality_measure": error
		}

	save_dict("output/", input_filename + "_certify_reference", certify_solution)


if __name__ == "__main__":

	import sys
	from reference_sampler import ReferenceSampler
	from stats import load_dict

	filename = sys.argv[1]
	filename_split = filename.split("_")

	n = int(filename_split[-1].split("=")[-1])
	N = int(filename_split[-2].split("=")[-1])

	mesh = UnitSquareMesh(n, n)

	W = FunctionSpace(mesh, "DG", 0)
	u = Function(W)

	input_filename = filename
	u_vec = np.loadtxt("output/" + input_filename + ".txt")
	u.vector()[:] = u_vec

	input_filename += "_exit_data"
	exit_data = load_dict("output", input_filename)

	sampler = ReferenceSampler(N)

	certify_reference(u, mesh, N, n, sampler, exit_data, input_filename)
