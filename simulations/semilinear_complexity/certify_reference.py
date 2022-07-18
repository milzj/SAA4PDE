"""The file is used to check optimality of the reference solution.

This code only depends on the implementation of prox and the random field.
"""

from dolfin import *
from dolfin_adjoint import *
import moola

from stats import save_dict

import numpy as np
from prox import prox_box_l1

set_log_level(30)

from random_fields import RHSExpr, RCCExpr, KappaExpr

def certify_reference(u, mesh, N, n, sampler, exit_data, input_filename, beta_filename):
	"""Computes a criticality measure to check whether u is a solution.

	We check whether the norm of u-prox(-1/alpha * nabla f(u)) is less
	than or equal to alpha*grad_norm, where grad_norm is the norm of the
	normal map at the final iterate.

	A file with ending "certify_reference" is printed to the directory
	of the solution u.

	The code should not be run in parallel.
	"""

	alpha = 1e-3
	ub = 10.
	lb = -10.

	# update beta
	beta = np.loadtxt("output/" + beta_filename + ".txt")


	V = FunctionSpace(mesh, "CG", 1)
	W = FunctionSpace(mesh, "DG", 0)

	y = Function(V)
	v = TestFunction(V)


	yd_expr = Expression('(0.25 < x[0] && x[0] < 0.75 && 0.25 < x[1] && x[1] < 0.75) ? -1.0 : 1.0', degree=0, domain=mesh)
	yd = Function(W)
	yd.interpolate(yd_expr)

	bcs = DirichletBC(V, 0.0, "on_boundary")


	m = 25
	kappa_expr, kappa_parameters = KappaExpr(m, mesh, W.ufl_element(), mpi_comm=mesh.mpi_comm())
	f_expr, f_parameters = RHSExpr(m, mesh, W.ufl_element(), mpi_comm=mesh.mpi_comm())
	g_expr, g_parameters = RCCExpr(m, mesh, W.ufl_element(), mpi_comm=mesh.mpi_comm())

	J = 0.0
	for i in range(N):

		sample = sampler.sample(i)

		sample1, sample2, sample3, sample4 = sample

		for k in range(m):
			kappa_parameters["p{}".format(k)] = sample1[k]
			kappa_parameters["q{}".format(k)] = sample2[k]
			g_parameters["r{}".format(k)] = sample3[k]
			f_parameters["t{}".format(k)] = sample4[k]

		kappa_expr.user_parameters.update(kappa_parameters)
		g_expr.user_parameters.update(g_parameters)
		f_expr.user_parameters.update(f_parameters)

		kappa = Function(W)
		kappa.interpolate(kappa_expr)

		g = Function(W)
		g.interpolate(g_expr)

		f = Function(W)
		f.interpolate(f_expr)

		F = (kappa*inner(grad(y), grad(v)) + y**3*v - g*u*v - f*v) * dx
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
		"is_solution": alpha*error <= (1+1e-14)*grad_norm,
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
	beta_filename = sys.argv[2]

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

	certify_reference(u, mesh, N, n, sampler, exit_data, input_filename, beta_filename)
