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
mesh = UnitIntervalMesh(n)

V = FunctionSpace(mesh, "CG", 1)
W = FunctionSpace(mesh, "DG", 0)

u = Function(W)
y = Function(V)
v = TestFunction(V)

F = (inner(grad(y), grad(v)) - u * v) * dx
bc = DirichletBC(V, 0.0, "on_boundary")
solve(F == 0, y, bc)


alpha = 0.002
beta = 0.01
lb = -6.
ub = 6.

yd_expr = Expression('0.25+sin(20.0*x[0]/pi)', degree=0, domain=mesh)
yd = Function(W)
yd.interpolate(yd_expr)

J = assemble(0.5 * inner(y - yd, y - yd) * dx)

rf = ReducedFunctional(J, Control(u))

v_moola = moola.DolfinPrimalVector(u)
nrf = NonsmoothFunctional(rf, v_moola, alpha, beta, lb, ub)

nproblem = moola.Problem(nrf)
v_moola.zero()

solver = SemismoothNewtonCG(nproblem, v_moola, options={'gtol': 1e-8,
                                                   'maxiter': 10,
                                                   'display': 3})

sol = solver.solve()

u_opt = sol['control'].data

plot(u_opt)
plt.savefig(outdir + "nominal_solution.pdf")
plt.savefig(outdir + "nominal_solution.png")
plt.close()

