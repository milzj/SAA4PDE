import pytest


from dolfin import *

from dolfin_adjoint import *

set_log_level(30)

import moola

from algorithms import SemismoothNewtonCG
from problem import NonsmoothFunctional
from base import norm_Linf

def semilinear_problem(n):

	alpha = 1e-3
	beta = 1e-3

	mesh = UnitSquareMesh(n, n)

	V = FunctionSpace(mesh, "CG", 1)
	W = FunctionSpace(mesh, "DG", 0)

	u = Function(W)
	u.vector()[:] = -1.0
	y = Function(V, name='State')
	v = TestFunction(V)

	F = (inner(grad(y), grad(v)) - u * v) * dx
	bc = DirichletBC(V, 0.0, "on_boundary")
	solve(F == 0, y, bc)

	x = SpatialCoordinate(mesh)
	w = Expression("sin(pi*x[0])*sin(pi*x[1])", degree=3)
	d = 1 / (2 * pi ** 2)
	d = Expression("d*w", d=d, w=w, degree=3)


	J = assemble(0.5 * inner(y - d, y - d) * dx)
	control = Control(u)

	rf = ReducedFunctional(J, control)

	v_moola = moola.DolfinPrimalVector(u)
	u_moola = moola.DolfinPrimalVector(u)
	nrf = NonsmoothFunctional(rf, v_moola, alpha, beta)

	problem = moola.Problem(nrf)

	return problem, u_moola


@pytest.mark.parametrize("n", [32])

def test_solver(n):

	problem, u_moola = semilinear_problem(n)

	solver = SemismoothNewtonCG(problem, u_moola, options={'gtol': 1e-10,
                                                   'maxiter': 20,
                                                   'display': 3,
                                                   'ncg_hesstol': 0,
						   'line_search': 'fixed'})

	solver.solve()


@pytest.mark.parametrize("n", [32])

def test_zero_solution(n):
	"""If beta is sufficiently large, then zero
	should be a stationary point.
	"""
	problem, u_moola = semilinear_problem(n)

	# Update beta (u_moola = 0)
	problem.obj(u_moola)
	grad = problem.obj.derivative(u_moola).primal()
	beta = norm_Linf(grad.data)
	problem.obj.beta = beta

	solver = SemismoothNewtonCG(problem, u_moola, options={'gtol': 1e-10,
                                                   'maxiter': 20,
                                                   'display': 3,
                                                   'ncg_hesstol': 0,
						   'line_search': 'fixed'})
	sol = solver.solve()
	u_opt = sol['control'].data

	assert norm(u_opt, "L2") == 0.

