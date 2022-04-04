from dolfin import *
from dolfin_adjoint import *
import moola

import numpy as np
import matplotlib.pyplot as plt

set_log_level(30)

from algorithms import SemismoothNewtonCG
from problem import NonsmoothFunctional

from ccdist1 import Adjoint, Solution, DesiredState, UncontrolledForce, BoundaryCoefficient

n = 2**6

mesh = UnitSquareMesh(n, n)

V = FunctionSpace(mesh, "CG", 1)
element = FiniteElement("DG", mesh.ufl_cell(), 0)
W = FunctionSpace(mesh, element)

u = Function(W)
y = Function(V)
v = TestFunction(V)

force = UncontrolledForce(element = element, domain = mesh)
f = Function(W)
f.interpolate(force)

F = (inner(grad(y), grad(v)) + y*v - f*v - u * v) * dx
solve(F == 0, y)


alpha = .5
beta = 0.0
lb = 0.0
ub = 1.0

desired_state = DesiredState(element = element, domain = mesh)
yd = Function(W)
yd.interpolate(desired_state)

bd = BoundaryCoefficient()

J = assemble(0.5 * inner(y - yd, y - yd) * dx) - assemble(12.0*y*ds)

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
solve(F == 0, y)
file = File("state.pvd")
file << y
