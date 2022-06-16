from dolfin import *
from dolfin_adjoint import *
import moola
import numpy as np
import time
import matplotlib.pyplot as plt

set_log_level(30)

def signif(beta):
	"Returns three significant figures of input float."
	return np.format_float_positional(beta, precision=3, unique=True, trim="k", fractional=False)

def lsqs_label(constant, rate, variable):
  "Least squares label"
  constant = signif(constant)
  rate = signif(rate)
  return r"${}\cdot {}^{}$".format(constant, variable, "{"+ str(rate)+"}")


def poisson_mother(n):

	set_working_tape(Tape())

	mesh = UnitSquareMesh(n, n)
	V = FunctionSpace(mesh, "CG", 1)
	W = FunctionSpace(mesh, "DG", 0)

	u = Function(W)
	y = Function(V)
	v = TestFunction(V)

	bcs = DirichletBC(V, 0.0, "on_boundary")


	f = Expression("10.0*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 1, 2)))", degree=5)
	yd = Constant(1.0)
	alpha = 1e-3
	gtol = 1e-8
	jtol = 1e-10

	F = inner(grad(y), grad(v))*dx - u*v*dx - f*v*dx
	solve(F == 0, y, bcs)

	J = assemble(0.5 * inner(y - yd, y - yd) * dx + .5*Constant(alpha)*u**2*dx)

	rf = ReducedFunctional(J, Control(u))
	problem = MoolaOptimizationProblem(rf)
	u_moola = moola.DolfinPrimalVector(u)

	solver = moola.BFGS(problem, u_moola, options={'gtol': gtol,
						   "jtol": jtol, "rjtol": jtol,
                                                   'maxiter': 20,
                                                   'display': 3})

	sol = solver.solve()
	u_opt = sol['control'].data

	return u_opt

def poisson(n):
	"""We construct a DG0 function using the solution to the Poisson equation."""
	mesh = UnitSquareMesh(n, n)
	V = FunctionSpace(mesh, "CG", 1)

	bcs = DirichletBC(V, 0.0, "on_boundary")

	u = TrialFunction(V)
	v = TestFunction(V)
	f = Expression("10.0*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 1, 2)))", degree=5)
	a = inner(grad(u), grad(v))*dx
	L = f*v*dx

	u = Function(V)
	solve(a == L, u, bcs)

	W = FunctionSpace(mesh, "DG",0)
	w = Function(W)
	w.interpolate(u)

	return w

def error_norm(u, uh, n, nh):

	_n = np.lcm(n, nh)
	_mesh = UnitSquareMesh(_n, _n)
	return errornorm(u, uh, degree_rise = 0, mesh=_mesh)

def _plot(u, n):

	p = plot(u)
	plt.colorbar(p)
	plt.savefig("output/n={}".format(n), bbox_inches="tight")


def test_error_computation():

	n_ref = 144
	u_ref = poisson_mother(n_ref)
	_plot(u_ref, n_ref)

	n_vec = np.array([2**n for n in range(3, 7)])
	n_vec = np.array([int(np.ceil(2**(i/2))) for i in range(6, 14)])

	n_vec = []
	for i in range(8, 144):
		if np.lcm(i, 144) <= 256:
			n_vec.append(i)

	n_vec.pop(1)
	n_vec.pop(3)

	n_vec = np.array(n_vec)


	error_indicators = [lambda u, uh, n, nh: error_norm(u, uh, n, nh),
			lambda u, uh, n, nh: errornorm(u, uh, degree_rise = 0),
			lambda u, uh, n, nh: errornorm(uh, u, degree_rise = 0),
			lambda u, uh, n, nh: errornorm(u, uh, degree_rise = 3)]

	errors  = {"error_norm": [], "errornorm_uuh_degree_rise=0": [], "errornorm_uhu_degree_rise=0": [], "errornorm_degree_rise=3": []}

	keys = []
	for e in errors:
		keys.append(e)


	for n in n_vec:

		u = poisson_mother(n)
		i = 0
		for error_indicator in error_indicators:
			error = error_indicator(u_ref, u, n_ref, n)
			errors[keys[i]].append(error)
			i += 1

	for key in keys:

		fig, ax = plt.subplots()
		ax.plot(n_vec, errors[key])
		ax.scatter(n_vec, errors[key])
		ax.set_xscale("log", base=2)
		ax.set_yscale("log", base=2)

		e = np.array(errors[key])
		X = np.ones((len(n_vec), 2)); X[:, 1] = np.log(1/n_vec)
		x, residudals, rank, s = np.linalg.lstsq(X, np.log(e), rcond=None)

		t, s = np.exp(x[0]), x[1]

		y = np.exp(x[0])*(1/n_vec)**x[1]
		ax.plot(n_vec, y, color="black", linestyle="--", label=lsqs_label(t, s, "h"))
		plt.legend()
		plt.savefig("output/{}.pdf".format(key))



if __name__ == "__main__":

	import os

	outdir = "output/"
	if not os.path.exists(outdir):
		os.makedirs(outdir)


	test_error_computation()
