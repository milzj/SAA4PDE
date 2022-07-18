from dolfin import *
from dolfin_adjoint import *

#set_log_level(30)

from random_problem import RandomProblem

from random_fields import RHSExpr, RCCExpr, KappaExpr

class RandomSemilinearProblem(RandomProblem):

	def __init__(self, n):

		set_working_tape(Tape())

		self.n = n
		self.alpha = 1e-3
		self.beta = 0.0
		self.lb = -10.0
		self.ub = 10.0
		degree = 0

		mesh = UnitSquareMesh(MPI.comm_self, n, n)
		V = FunctionSpace(mesh, "CG", 1)
		U = FunctionSpace(mesh, "DG", 0)


		self.V = V
		self.U = U

		self.y = Function(V)
		self.v = TestFunction(V)

		self.u = Function(U)

		yd_expr = Expression('(0.25 < x[0] && x[0] < 0.75 && 0.25 < x[1] && x[1] < 0.75) ? -1.0 : 1.0', \
			degree=0, domain=mesh, mpi_comm=mesh.mpi_comm())

		yd = Function(U)
		yd.interpolate(yd_expr)
		self.yd = yd

		self.bcs = DirichletBC(self.V, 0.0, "on_boundary")

		m = 25
		self.m = m

		self.kappa_expr, self.kappa_parameters = KappaExpr(m, mesh, U.ufl_element(), mpi_comm=mesh.mpi_comm())
		self.f_expr, self.f_parameters = RHSExpr(m, mesh, U.ufl_element(), mpi_comm=mesh.mpi_comm())
		self.g_expr, self.g_parameters = RCCExpr(m, mesh, U.ufl_element(), mpi_comm=mesh.mpi_comm())

	@property
	def control_space(self):
		return self.U


	def state(self, y, v, u, sample):


		sample1, sample2, sample3, sample4 = sample

		kappa_expr, kappa_parameters = self.kappa_expr, self.kappa_parameters
		f_expr, f_parameters = self.f_expr, self.f_parameters
		g_expr, g_parameters = self.g_expr, self.g_parameters

		for k in range(self.m):
			kappa_parameters["p{}".format(k)] = sample1[k]
			kappa_parameters["q{}".format(k)] = sample2[k]
			g_parameters["r{}".format(k)] = sample3[k]
			f_parameters["t{}".format(k)] = sample4[k]

		kappa_expr.user_parameters.update(kappa_parameters)
		g_expr.user_parameters.update(g_parameters)
		f_expr.user_parameters.update(f_parameters)

		kappa = Function(self.U)
		kappa.interpolate(kappa_expr)

		g = Function(self.U)
		g.interpolate(g_expr)

		f = Function(self.U)
		f.interpolate(f_expr)

		F = (kappa*inner(grad(y), grad(v)) + y**3*v - g*u*v - f*v) * dx

		solve(F == 0, y, bcs=self.bcs)

	def __call__(self, u, sample):

		y = self.y
		yd = self.yd
		y.vector().zero()
		v = self.v
		alpha = self.alpha
		self.state(y, v, u, sample)

		return 0.5 * assemble((y-yd) ** 2 * dx)


