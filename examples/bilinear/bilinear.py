"""Implements the bilinear control problem from Vallejos et al. (2010).

References:
----------
M. Vallejos and A. Borzı̀. Multigrid optimization methods for linear and
bilinear elliptic optimal control problems. Computing, 82(1):31–52
"""
from dolfin import *
from dolfin_adjoint import *
import moola

import numpy as np
import matplotlib.pyplot as plt

set_log_level(30)

from algorithms import SemismoothNewtonCG, TrustRegionSemismoothNewtonCG
from problem import NonsmoothFunctional

u_vec = np.loadtxt("initial_point.txt")

n = 2**6
mesh = UnitSquareMesh(n, n)

V = FunctionSpace(mesh, "CG", 1)
W = FunctionSpace(mesh, "DG", 0)

u = Function(W)
y = Function(V)
v = TestFunction(V)

yd_expr = Expression('(0.25 < x[0] && x[0] < 0.75 && 0.25 < x[1] && x[1] < 0.75) ? 2.0 : 1.0', degree=0, domain=mesh)
yd = Function(W)
yd.interpolate(yd_expr)


f = Constant(1.0)
F = (inner(grad(y), grad(v)) - y*u*v - f*v) * dx
bc = DirichletBC(V, 0.0, "on_boundary")
solve(F == 0, y, bc)


alpha = 1e-4
beta = 0.0
lb = -np.inf
ub = np.inf

J = assemble(0.5 * inner(y - yd, y - yd) * dx)

rf = ReducedFunctional(J, Control(u))

v_moola = moola.DolfinPrimalVector(u)
nrf = NonsmoothFunctional(rf, v_moola, alpha, beta, lb = lb, ub = ub)

nproblem = moola.Problem(nrf)
v_moola.zero()
v_moola.data.vector()[:] = u_vec

solver = SemismoothNewtonCG(nproblem, v_moola, options={'gtol': 1e-9,
                                                   'maxiter': 100,
                                                   'display': 3})


sol = solver.solve()

u_opt = sol["control"].data


plot(u_opt)
plt.savefig("solution.pdf")
plt.close()


plot(yd)
plt.savefig("yd.pdf")
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
