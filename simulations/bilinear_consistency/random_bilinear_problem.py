import numpy as np
from dolfin import *
from dolfin_adjoint import *

#set_log_level(30)

from random_problem import RandomProblem

from random_fields.random_diffusion_coefficient import RandomDiffusionCoefficient
from random_fields.random_nonnegative_coefficient import RandomNonnegativeCoefficient

class RandomBilinearProblem(RandomProblem):

	def __init__(self, n, sampler):

		set_working_tape(Tape())

		self.n = n
		self.alpha = 1e-3
		self.beta = 0.0
		self.lb = 0.0
		self.ub = 100.0

		mesh = UnitSquareMesh(MPI.comm_self, n, n)
		V = FunctionSpace(mesh, "CG", 1)
		U = FunctionSpace(mesh, "DG", 0)


		self.V = V
		self.U = U

		self.y = Function(V)
		self.v = TestFunction(V)

		self.u = Function(U)

		yd_expr = Expression("sin(2.0*pi*x[0])*sin(2.0*pi*x[1])*exp(2.0*x[0])/6.0", \
			degree=0, domain=mesh, mpi_comm=mesh.mpi_comm())

		yd = Function(U)
		yd.interpolate(yd_expr)
		self.yd = yd

		self.bcs = DirichletBC(self.V, 0.0, "on_boundary")

		self.kappa = RandomDiffusionCoefficient(mpi_comm=mesh.mpi_comm())
		self.kappa.function_space = U
		self.g = RandomNonnegativeCoefficient(mpi_comm=mesh.mpi_comm())
		self.g.function_space = U

		self.f = Constant(-1.0)


		self.sampler = sampler



	@property
	def control_space(self):
		return self.U


	def state(self, y, v, u, sample):

		a, b, = np.split(sample, 2)
		kappas = self.kappa.sample(a)
		gs = self.g.sample(b)

		f = self.f

		F = (kappas*inner(grad(y), grad(v)) + gs*y*u*v - f*v) * dx
		solve(F == 0, y, bcs=self.bcs)

	def __call__(self, u, sample):

		y = self.y
		yd = self.yd
		y.vector().zero()
		v = self.v
		alpha = self.alpha
		self.state(y, v, u, sample)

		return 0.5 * assemble((y-yd) ** 2 * dx)


