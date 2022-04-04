import pytest

from dolfin import *

from dolfin import *
from dolfin_adjoint import *
set_log_level(30)

import moola

from algorithms import SemismoothNewtonCG
from problem import NonsmoothFunctional

from example73 import Solution, Parameter, State, LaplaceAdjoint, DesiredState

import numpy as np

def convergence_rates(E_values, eps_values, show=True):
	"""
	Source https://github.com/dolfin-adjoint/pyadjoint/blob/master/pyadjoint/verification.py
	"""
	from numpy import log
	r = []
	for i in range(1, len(eps_values)):
		r.append(log(E_values[i] / E_values[i - 1])
		/ log(eps_values[i] / eps_values[i - 1]))
	if show:
		print("Computed convergence rates: {}".format(r))
	return r

def lstsq_rates(E_values, h_values):
	"""
	Applies least squares to log transformation of errors = c1 pow(h, c2).
	"""

	X = np.ones((len(h_values), 2)); X[:, 1] = np.log(h_values)

	x, residuals, rank, s = np.linalg.lstsq(X, np.log(E_values), rcond = None)

	return x[1], np.exp(x[0])


def example73(n=32, alpha=1e-4, beta=1.5e-4):
	"""

	Note:
	-----
	We approximate desired_state using interpolation rather than projection
	because project would yield the error
	AttributeError: 'DesiredState' object has no attribute 'block_variable'
	"""


	degree = 0
	mesh = UnitSquareMesh(n, n)
	element = FiniteElement("DG", mesh.ufl_cell(), degree)
	U = FunctionSpace(mesh, element)
	V = FunctionSpace(mesh, "CG", 1)

	params = Parameter(alpha, beta)
	lb = Constant(params.lb)
	ub = Constant(params.ub)
	g = Constant(0.0)

	# solution
	solution = Solution(params.lb, params.ub,\
				element = element,\
				domain = mesh)

	desired_state = DesiredState(params, element = element, domain = mesh)


	yd = Function(U)
	yd.interpolate(desired_state)

	u = Function(U)
	y = Function(V, name='State')
	v = TestFunction(V)

	F = (inner(grad(y), grad(v)) - u * v) * dx
	bc = DirichletBC(V, 0.0, "on_boundary")
	solve(F == 0, y, bc)

	J = assemble(0.5 * inner(y - yd, y - yd) * dx)
	control = Control(u)

	rf = ReducedFunctional(J, control)

	v_moola = moola.DolfinPrimalVector(Function(U))
	u_moola = moola.DolfinPrimalVector(Function(U))

	nrf = NonsmoothFunctional(rf, v_moola, alpha, beta, lb=params.lb, ub=params.ub)


	problem = moola.Problem(nrf)

	solver = SemismoothNewtonCG(problem, u_moola, options={'gtol': 1e-10,
       	                                   'maxiter': 20,
                                                  'display': 3,
                                                  'ncg_hesstol': 0,
						   'line_search': 'fixed',
						"restrict": True})


	return solver, solution, mesh

def test_example73_convergence_rates():
	"""Code verification

	Tests whether the computed orders-of-accuracy are of the
	order of the theoretical order-of-accuracy.
	"""

	p = 8
	degree_rise = 0

	gtol = 1e-12
	quadratic, u_ref, mesh_ref = example73(2**p)

	errors = []
	values = []

	for i in range(2, p):

		solver, _, _ = example73(n=2**i)

		sol = solver.solve()

		u_opt = sol["control"].data

		error = errornorm(u_ref, u_opt, degree_rise = degree_rise, mesh = mesh_ref)

		errors.append(error)
		values.append(1.0/2**i)

	rate = np.median(convergence_rates(errors, values))
	assert np.isclose(rate, 1.0, rtol=.2)

	lstsq_rate, _ = lstsq_rates(errors, values)
	assert np.isclose(lstsq_rate, 1.0, rtol=.2)

