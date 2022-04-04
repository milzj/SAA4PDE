import pytest

from fenics import *
from dolfin_adjoint import *

set_log_level(30)

import moola

from algorithms import SemismoothNewtonCG
from problem import NonsmoothFunctional

import numpy as np

def convergence_rates(E_values, eps_values, show=True):
	"""
	Source https://github.com/dolfin-adjoint/pyadjoint/blob/master/pyadjoint/verification.py

	Added print("Computed convergence rates: {}".format(r))
	"""

	from numpy import log
	r = []
	for i in range(1, len(eps_values)):
		r.append(log(E_values[i] / E_values[i - 1])
			/ log(eps_values[i] / eps_values[i - 1]))
	if show:
		print("Residuals:{}".format(E_values))
		print("Computed convergence rates: {}".format(r))
	return r

@pytest.mark.parametrize("n", [32, 64])
@pytest.mark.parametrize("alpha", [1e-3, 1e-2, 1e-1])
@pytest.mark.parametrize("beta", [0.0, 1e-6, 1e-5, 1e-3, 1.0])

def test_derivative_normal_map(n,alpha,beta):
	"""Performs a Taylor test for the normal map.

	If the normal map F is Lipschitz continuous at x, then
	norm(F(x+d) - F(x)) = O(norm(d)).

	If the normal map F is semismooth at x, 
	then F(x+d) - F(x) - DF(x+d)d = o(norm(d)).

	If F is semismooth at x of order alpha, then
	then F(x+d) - F(x) - DF(x+d)d = O(norm(d)^(1+alpha)).
	"""

	mesh = UnitSquareMesh(n, n)

	V = FunctionSpace(mesh, "CG", 1)
	W = FunctionSpace(mesh, "DG", 0)

	u = Function(W)
	y = Function(V, name='State')
	v = TestFunction(V)

	F = (inner(grad(y), grad(v)) + exp(y)*v - u * v) * dx
	bc = DirichletBC(V, 0.0, "on_boundary")
	solve(F == 0, y, bc)

	yd = Expression("exp(2.0*x[0])*sin(2.0*pi*x[0])*sin(2.0*pi*x[1])/6.0", degree=1)

	J = assemble(0.5 * (y-yd)**2 * dx)
	control = Control(u)
	rf = ReducedFunctional(J, control)

	v_moola = moola.DolfinPrimalVector(Function(W))
	u_moola = moola.DolfinPrimalVector(Function(W))
	h_moola = moola.DolfinPrimalVector(Function(W))

	nrf = NonsmoothFunctional(rf, v_moola, alpha, beta)

	v_moola.data.interpolate(Expression("sin(pi*x[0])*sin(pi*x[1])", degree = 1))
	h_moola.data.interpolate(Expression("cos(pi*x[0])*cos(pi*x[1])", degree = 1))

	Jm = nrf.normal_map(u_moola, v_moola).primal()

	lipschitz_residuals = []
	derivative_residuals = []
	epsilons = [0.01 / 2**i for i in range(4)]

	v_moola_new = moola.DolfinPrimalVector(Function(W))

	for eps in epsilons:

		v_moola_new.assign(v_moola)
		v_moola_new.axpy(eps, h_moola)

		Jp = nrf.normal_map(u_moola, v_moola_new).primal()
		DJph = nrf.derivative_normal_map(u_moola, v_moola_new)(h_moola).primal()

		# norm(F(x+d) - F(x)) = O(norm(d))
		lipschitz_res = errornorm(Jp.data, Jm.data, degree_rise = 0, norm_type = "L2")

		DJph.scale(eps)
		DJph.axpy(1.0, Jm)
		# F(x+d) - F(x) - DF(x+d)d = O(norm(d)^(1+alpha))
		derivative_res = errornorm(Jp.data, DJph.data, degree_rise = 0, norm_type = "L2")

		lipschitz_residuals.append(lipschitz_res)
		derivative_residuals.append(derivative_res)


	r = np.median(convergence_rates(lipschitz_residuals, epsilons))
	assert np.isclose(r, 1.0, atol = 0.2)
	
	if np.median(derivative_residuals) < 1e-12:
		r = 2
	else:
		r = np.median(convergence_rates(derivative_residuals, epsilons))
	assert r > 1.2
	

if __name__ == "__main__":

	test_derivative_normal_map(64, 1e-3, 1e-6)

