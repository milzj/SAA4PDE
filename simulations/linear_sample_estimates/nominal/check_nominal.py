from dolfin import *
from dolfin_adjoint import *
import moola

import numpy as np
import matplotlib.pyplot as plt

set_log_level(30)

from algorithms import SemismoothNewtonCG
from problem import NonsmoothFunctional

import os
outdir = "output/"
if not os.path.exists(outdir):
	os.makedirs(outdir)

n = 144
mesh = UnitSquareMesh(n, n)

V = FunctionSpace(mesh, "CG", 1)
W = FunctionSpace(mesh, "DG", 0)

u = Function(W)
y = Function(V)
v = TestFunction(V)

F = (inner(grad(y), grad(v)) - u * v) * dx
bc = DirichletBC(V, 0.0, "on_boundary")
solve(F == 0, y, bc)


alpha = 1e-3
beta = 0.01
lb = -6.
ub = 6.

yd = Expression('(0.25 < x[0] && x[0] < 0.75 && 0.25 < x[1] && x[1] < 0.75) ? -1.0 : 1.0', degree=0, domain=mesh)
J = assemble(0.5 * inner(y - yd, y - yd) * dx)

rf = ReducedFunctional(J, Control(u))

v_moola = moola.DolfinPrimalVector(u)
nrf = NonsmoothFunctional(rf, v_moola, alpha, beta, lb, ub)

nproblem = moola.Problem(nrf)
v_moola.zero()

solver = SemismoothNewtonCG(nproblem, v_moola, options={'gtol': 1e-8,
                                                   'maxiter': 20,
                                                   'display': 3})

sol = solver.solve()

u_opt = sol['control'].data

plot(u_opt)
p = plot(u_opt)
plt.colorbar(p)
plt.savefig(outdir + "nominal_solution.pdf")
plt.savefig(outdir + "nominal_solution.png")
plt.close()

