"""Implements Example 6.1 in De los Reyes (2015).

References:
----------

J. C. De los Reyes, Numerical PDE-Constrained Optimization, SpringerBriefs
Optim., Springer, Cham, 2015, https://doi.org/10.1007/978-3-319-13395-9
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

F = (inner(grad(y), grad(v)) + y**3*v - u * v) * dx
bc = DirichletBC(V, 0.0, "on_boundary")
solve(F == 0, y, bc)


alpha = 1e-3
beta = 0.008

yd = Expression("sin(2.0*pi*x[0])*sin(2.0*pi*x[1])*exp(2.0*x[0])/6.0", degree = 1)
J = assemble(0.5 * inner(y - yd, y - yd) * dx)

rf = ReducedFunctional(J, Control(u))

v_moola = moola.DolfinPrimalVector(u)
nrf = NonsmoothFunctional(rf, v_moola, alpha, beta)

nproblem = moola.Problem(nrf)
v_moola.zero()

solver = SemismoothNewtonCG(nproblem, v_moola, options={'gtol': 1e-7,
                                                   'maxiter': 20,
                                                   'display': 3})

sol = solver.solve()

u_opt = sol['control'].data
nz = sum(u_opt.vector()[:] == 0.0)/W.dim()
print("nz=",nz)



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

plot(y)
plt.savefig("state.pdf")
plt.close()
