if __name__ == "__main__":

	from dolfin import *

	from dolfin_adjoint import *

	set_log_level(30)

	import moola

	from semismooth_newton_cg import SemismoothNewtonCG
	from femsaa import NonsmoothFunctional

	n = 256
	mesh = UnitSquareMesh(n, n)

	V = FunctionSpace(mesh, "CG", 1)
	W = FunctionSpace(mesh, "DG", 0)

	u = Function(W)
	u.vector()[:] = -1.0
	y = Function(V, name='State')
	v = TestFunction(V)


	p = Constant(1.0)

	F = (p*inner(grad(y), grad(v)) - u * v) * dx
	bc = DirichletBC(V, 0.0, "on_boundary")
	solve(F == 0, y, bc)

	x = SpatialCoordinate(mesh)
	w = Expression("sin(pi*x[0])*sin(pi*x[1])", degree=3)
	d = 1 / (2 * pi ** 2)
	d = Expression("d*w", d=d, w=w, degree=3)


	alpha = 1e-3
	J = assemble(0.5 * inner(y - d, y - d) * dx)
	control = Control(u)

	rf = ReducedFunctional(J, control)


	v_moola = moola.DolfinPrimalVector(u)
	print(rf(u))
	beta = 1e-3
	nrf = NonsmoothFunctional(rf, v_moola, alpha, beta)
	print(nrf(v_moola))
	print(nrf(v_moola))
	d = nrf.derivative(v_moola)
#	u_moola = v_moola.copy()
	u_moola = moola.DolfinPrimalVector(u)
	nrf.normal_map(u_moola, v_moola)

	H = nrf.derivative_normal_map(u_moola, v_moola)

	Hv = H(v_moola)
	print(Hv.apply(v_moola))

	problem = moola.Problem(nrf)

	solver = SemismoothNewtonCG(problem, u_moola, options={'gtol': 1e-10,
                                                   'maxiter': 20,
                                                   'display': 3,
                                                   'ncg_hesstol': 0,
						   'line_search': 'fixed'})

	solver.solve()


