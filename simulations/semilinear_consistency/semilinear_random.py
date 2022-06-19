"""Implements the control problem considered in sect. 4.2 in Ref. [1]
but with feasible set defined by lower bound equal to -7 and upper bound
equal to 7.

References:
-----------
[1] C. Geiersbach and T. Scarinci, Stochastic proximal gradient methods for nonconvex
    problems in Hilbert spaces, Comput. Optim. Appl., 78 (2021), pp. 705–740, https:
    //doi.org/10.1007/s10589-020-00259-y.
"""

import os
outdir = "output/"
if not os.path.exists(outdir):
	os.makedirs(outdir)
from datetime import datetime

from dolfin import *
from dolfin_adjoint import *
import moola

import numpy as np
from random_field import RandomField

set_log_level(30)

from algorithms import SemismoothNewtonCG
from problem import NonsmoothFunctional

def semilinear_random_problem(N, n):

	mesh = UnitSquareMesh(n, n)

	alpha = 0.001
	beta = 0.008

	V = FunctionSpace(mesh, "CG", 1)
	W = FunctionSpace(mesh, "DG", 0)

	kappa = RandomField()
	g = RandomField()

	kappa.function_space = V
	g.function_space = V

	kappa.version = 1
	g.version = 10000

	u_expr = Expression("sin(4.0*pi*x[0])*sin(4.0*pi*x[1])", degree = 1)

	u = Function(W)
	u.interpolate(u_expr)

	y = Function(V)
	v = TestFunction(V)


	yd = Expression("sin(2.0*pi*x[0])*sin(2.0*pi*x[1])*exp(2.0*x[0])/6.0", degree = 1)
	bc = DirichletBC(V, 0.0, "on_boundary")

	J = 0.0
	for i in range(N):

		kappas = kappa.sample()
		gs = g.sample()

		F = kappas*inner(grad(y), grad(v))*dx + gs*y**3*v*dx - u * v * dx
		solve(F == 0, y, bc)

		J += (1/N)*assemble(0.5 * inner(y - yd, y - yd) * dx)

	rf = ReducedFunctional(J, Control(u))
	v_moola = moola.DolfinPrimalVector(u)

	ub = 7.0
	lb = -7.0

	nrf = NonsmoothFunctional(rf, v_moola, alpha, beta, lb = lb, ub = ub)

	nproblem = moola.Problem(nrf)
	v_moola.zero()


	solver = SemismoothNewtonCG(nproblem, v_moola, options={'gtol': 1e-8,
	                                                   'maxiter': 20,
	                                                   'display': 3,
	                                                   'ncg_hesstol': 0,
							   'line_search': 'fixed', "restrict": True})

	sol = solver.solve()

	u_opt = sol['control'].data
	now = datetime.now().strftime("%d-%B-%Y-%H-%M-%S")

	filename = outdir + now + "_solution_N={}_n={}.txt".format(N,n)
	np.savetxt(filename, u_opt.vector()[:])

