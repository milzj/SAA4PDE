from dolfin import *
from dolfin_adjoint import *

#set_log_level(30)

from random_problem import RandomProblem
from mknrandom_field import MKNRandomField

class RandomLinearProblem(RandomProblem):

	def __init__(self, n, sampler):

		set_working_tape(Tape())

		self.n = n
		self.alpha = 0.001
		self.beta = 0.01
		self.lb = -6.
		self.ub = 6.

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

		kappa = MKNRandomField(mean=0.0, var=1.0)
		kappa.function_space = V
		self.kappa = kappa

		self.sampler = sampler

	@property
	def control_space(self):
		return self.U


	def state(self, y, v, u, sample):

		kappas = self.kappa.sample(sample=sample)

		F = (kappas*inner(grad(y), grad(v)) - u * v) * dx
		solve(F == 0, y, bcs=self.bcs)

	def __call__(self, u, sample):

		y = self.y
		yd = self.yd
		y.vector().zero()
		v = self.v
		alpha = self.alpha
		self.state(y, v, u, sample)

		return 0.5 * assemble((y-yd) ** 2 * dx)


