import numpy as np
from fenics import *
from dolfin_adjoint import *

from random_fields import RandomField

class RandomDiffusionCoefficient(RandomField):


	def __init__(self, mpi_comm=MPI.comm_self):

		l = 0.15
		mean = 0.0
		m = 200

		super().__init__(mean=mean, l=l, m=m, mpi_comm=mpi_comm)


	def sample(self, sample):

		y = self._sample(sample)

		f = Function(self.function_space)
		f.vector()[:] =  np.exp(y)

		return f



