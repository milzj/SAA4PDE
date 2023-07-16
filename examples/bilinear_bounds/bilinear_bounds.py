"""Implements a bilinear control problem from Borzı̀ et al. (2011).

References:
----------

Alfio Borzı̀ and Volker Schulz. Computational Optimization of Systems Governed
by Partial Differential Equations. Comput. Sci. Eng. 8. SIAM, Philadelphia, PA,
2011.

Michelle Vallejos. MGOPT with gradient projection method for solving bilinear
elliptic optimal control problems. Computing, 87(1-2):21–33, 2010.
"""
from dolfin import *
from dolfin_adjoint import *
import moola

import numpy as np
import matplotlib.pyplot as plt

set_log_level(30)

from algorithms import SemismoothNewtonCG
from problem import NonsmoothFunctional

n = 2**7
mesh = UnitSquareMesh(n, n)

V = FunctionSpace(mesh, "CG", 1)
W = FunctionSpace(mesh, "DG", 0)

u = Function(W)
y = Function(V)
v = TestFunction(V)


f = Expression("sin(2.0*pi*x[0])*sin(2*pi*x[1])", degree = 1)
yd = Expression("1.0+f", f=f, degree = 1)

F = (inner(grad(y), grad(v)) - y*u*v - f*v) * dx
bc = DirichletBC(V, 0.0, "on_boundary")
solve(F == 0, y, bc)


alpha = 1e-4
beta = 0.0
lb = -4.0
ub = 4.0

J = assemble(0.5 * inner(y - yd, y - yd) * dx)

rf = ReducedFunctional(J, Control(u))

v_moola = moola.DolfinPrimalVector(u)
nrf = NonsmoothFunctional(rf, v_moola, alpha, beta, lb = lb, ub = ub)

nproblem = moola.Problem(nrf)
v_moola.zero()

solver = SemismoothNewtonCG(nproblem, v_moola, options={'gtol': 1e-10,
                                                   'maxiter': 20,
                                                   'display': 3,
                                                   'ncg_hesstol': 0,
						   'line_search': 'fixed'})

sol = solver.solve()

u_opt = sol["control"].data

plot(u_opt)
plt.savefig("solution.pdf")
plt.close()

file = File("solution.pvd")
file << u_opt

w_opt = Function(V)
w_opt.interpolate(u_opt)

file = File("p1solution.pvd")
file << w_opt

u.assign(u_opt)
solve(F == 0, y, bc)
file = File("state.pvd")
file << y


