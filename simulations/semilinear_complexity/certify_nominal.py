"""The file is used to check optimality of the nominal solution.

This code only depends on the implementation of prox.
"""

from dolfin import *
from dolfin_adjoint import *
import moola

from stats import save_dict

import numpy as np
from prox import prox_box_l1

set_log_level(30)

def certify_nominal(u, mesh, n, exit_data, input_filename, beta_filename):
	"""Computes a criticality measure to check whether u is a solution.

	We check whether the norm of u-prox(-1/alpha * nabla f(u)) is less
	than or equal to alpha*grad_norm, where grad_norm is the norm of the
	normal map at the final iterate.

	A file with ending "certify_reference" is printed to the directory
	of the solution u.

	The code should not be run in parallel.
	"""

	alpha = 1e-3
	beta = np.loadtxt("output/" + beta_filename + ".txt")
	ub = 10.
	lb = -10.


	V = FunctionSpace(mesh, "CG", 1)
	W = FunctionSpace(mesh, "DG", 0)

	y = Function(V)
	v = TestFunction(V)

	yd_expr = Expression('(0.25 < x[0] && x[0] < 0.75 && 0.25 < x[1] && x[1] < 0.75) ? -1.0 : 1.0', \
			degree=0, domain=mesh, mpi_comm=mesh.mpi_comm())
	yd = Function(W)
	yd.interpolate(yd_expr)

	bcs = DirichletBC(V, 0.0, "on_boundary")

	f = Constant(1.0)
	g = Constant(1.0)

	# Implementation of kappa based on that in https://fenicsproject.org/pub/tutorial/sphinx1/._ftut1005.html
	kappa = Expression('x[0] <= 0.5 ? exp(0.0) : 1.5', degree=0,  domain=mesh, mpi_comm=mesh.mpi_comm())

	F = (kappa*inner(grad(y), grad(v)) + y**3*v - g*u*v - f*v) * dx
	solve(F == 0, y, bcs)

	J = assemble(0.5 * inner(y - yd, y - yd) * dx)

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
	from stats import load_dict

	filename = sys.argv[1]
	filename_split = filename.split("_")
	n = int(filename_split[-1].split("=")[-1])
	beta_filename = sys.argv[2]

	mesh = UnitSquareMesh(n, n)

	W = FunctionSpace(mesh, "DG", 0)
	u = Function(W)

	input_filename = filename
	u_vec = np.loadtxt("output/" + input_filename + ".txt")
	u.vector()[:] = u_vec

	input_filename += "_exit_data"
	exit_data = load_dict("output", input_filename)

	certify_nominal(u, mesh, n, exit_data, input_filename, beta_filename)
