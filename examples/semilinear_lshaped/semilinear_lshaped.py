"""Examples taken from sect. 6 in Garreis and Ulbrich (2019)

S. Garries and M. Ulbrich. An inexact trust-region algorithm for constrained
problems in Hilbert space and its application to the adaptive solution of
optimal control problems with PDEs. Preprint, Technische Universität München,
München, 2019. URL:
https://www-m1.ma.tum.de/foswiki/pub/M1/Lehrstuhl/PublikationenUlbrich/GarreisUlbrich-2019-1.pdf
"""

from dolfin import *
from dolfin_adjoint import *
import moola

import numpy as np
import matplotlib.pyplot as plt

set_log_level(30)

from algorithms import SemismoothNewtonCG, TrustRegionSemismoothNewtonCG
from problem import NonsmoothFunctional


# L-shaped domain as subset of unit square
mesh = Mesh("lshape.xml.gz")
# TODO: Is this transformation okay?
mesh.coordinates()[:] = -2*mesh.coordinates()+1.0

nrefine = 3

for i in range(nrefine):
	mesh = refine(mesh)

V = FunctionSpace(mesh, "CG", 1)
W = FunctionSpace(mesh, "DG", 0)

u = Function(W)
y = Function(V)
v = TestFunction(V)

F = (inner(grad(y), grad(v)) + y**3*v - u * v) * dx
bc = DirichletBC(V, 0.0, "on_boundary")
solve(F == 0, y, bc)

alpha = 1e-3
beta = 0.0
ub = 14.
lb = -np.inf

setup = 1
if setup == 1:
	yd = Constant(1.0)
elif setup == 2:
	yd = Expression("9.0*(pow(x[0],3)-x[0])*(pow(x[1],3)-x[1])", degree = 1)

J = assemble(0.5 * inner(y - yd, y - yd) * dx)

rf = ReducedFunctional(J, Control(u))

v_moola = moola.DolfinPrimalVector(u)
nrf = NonsmoothFunctional(rf, v_moola, alpha, beta, lb = lb, ub = ub)

nproblem = moola.Problem(nrf)
v_moola.zero()


if False:

	solver = TrustRegionSemismoothNewtonCG(nproblem, v_moola, options={'gtol': 1e-7,
                                                   'maxiter': 22,
                                                   'display': 3})

else:
	solver = SemismoothNewtonCG(nproblem, v_moola, options={'gtol': 1e-7,
                                                   'maxiter': 20,
                                                   'display': 3,
                                                   'ncg_hesstol': 0,
						"restrict": False,
						"correction_step": False})

sol = solver.solve()

u_opt = sol["control"].data

plot(u_opt)
plt.savefig("solution_setup={}.pdf".format(setup))
plt.close()

file = File("solution_setup={}.pvd".format(setup))
file << u_opt


w_opt = Function(V)
w_opt.interpolate(u_opt)


file = File("p1solution_setup={}.pvd".format(setup))
file << w_opt

u.assign(u_opt)
solve(F == 0, y, bc)
file = File("state_setup={}.pvd".format(setup))
file << y
