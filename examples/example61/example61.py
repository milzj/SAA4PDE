"""Implements the Example 6.1 from Kunisch et al. (2010).

Using norm(u-ud)**2 = norm(u)**2  - 2(u,ud) + norm(ud)**2,
we can define a problem equivalent to eq. (6.1) in
Kunisch et al. (2010).


References:
----------
K. Kunisch, W. Liu, Y. Chang, N. Yan, and R. Li, Adaptive finite element
approximation for a class of parameter estimation problems, J. Comput. Math., 28
(2010), pp. 645–675.
"""

from dolfin import *
from dolfin_adjoint import *
import moola

import numpy as np
import matplotlib.pyplot as plt

set_log_level(30)

from algorithms import SemismoothNewtonCG
from problem import NonsmoothFunctional

n = 2**6
mesh = UnitSquareMesh(n, n)

V = FunctionSpace(mesh, "CG", 1)
W = FunctionSpace(mesh, "DG", 0)

u = Function(W)
y = Function(V)
v = TestFunction(V)

yd = Expression("sin(pi*x[0])*sin(pi*x[1])", degree = 1, domain = mesh)
ud = Expression("x[0]+x[1] > 1.0 ? 1.0 : 0.0", degree = 1, domain = mesh)
f = Expression("2*pi*pi*a+a*b", a=yd, b=ud, degree = 1)
F = (inner(grad(y), grad(v)) + y*u*v - f*v) * dx
bc = DirichletBC(V, 0.0, "on_boundary")
solve(F == 0, y, bc)


alpha = 1e-2
beta = 0.0
lb = 0.0
ub = np.inf

J = assemble(0.5 * inner(y - yd, y - yd) * dx - alpha*u*ud*dx)

rf = ReducedFunctional(J, Control(u))

v_moola = moola.DolfinPrimalVector(u)
nrf = NonsmoothFunctional(rf, v_moola, alpha, beta, lb = lb, ub = ub)

nproblem = moola.Problem(nrf)
v_moola.zero()

solver = SemismoothNewtonCG(nproblem, v_moola, options={'gtol': 1e-10,
                                                   'maxiter': 20,
                                                   'display': 3,
                                                   'ncg_hesstol': 0})

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
