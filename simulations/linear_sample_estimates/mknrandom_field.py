import numpy as np
import itertools
from fenics import *
from dolfin_adjoint import *

class MKNRandomField(object):

	def __init__(self, mean=1.0, var=np.exp(-1.125)):
		"""
		Implements the random field in sect. 8 in Martin, Krumscheid, and Nobile (2021).


		Arguments:
		----------
		mean : float
			scalar mean
		var : float
			standard deviation

		References:
		----------

		M. Martin, S. Krumscheid, and F. Nobile, Complexity analysis of stochastic gradient methods for PDE-
		constrained optimal control problems with uncertain parameters, ESAIM Math. Model. Numer. Anal., 55
		(2021), pp. 1599–1633, https://doi.org/10.1051/m2an/2021025.
		"""
		self.seed = 1000

		self.mean = mean
		self.var = var



	def bump_seed(self):
		self.seed += 1

	@property
	def function_space(self):
		return self._function_space

	@function_space.setter
	def function_space(self, function_space):
		self._function_space = function_space
		mpi_comm = function_space.mesh().mpi_comm()
		self.g = Expression("xi1*cos(1.1*pi*x[0])+xi2*cos(1.2*pi*x[0])+xi3*sin(1.3*pi*x[1])+xi4*sin(1.4*pi*x[1])",
				xi1=0.0, xi2 = 0.0, xi3 = 0.0, xi4 = 0.0, degree = 1, mpi_comm=mpi_comm)
		self.a = Expression("m+exp(g)", m=self.mean, g=self.g, degree = 1, mpi_comm=mpi_comm)

	def sample(self, sample=None):
		return self.realization(sample=sample)

	def realization(self, sample=None):
		"""Computes a realization of the random field.

		If sample=None, then four uniform random variables
		are generated and thee seed is increased by one.
		"""
		xi = sample

		self.g.xi1 = self.var*xi[0]
		self.g.xi2 = self.var*xi[1]
		self.g.xi3 = self.var*xi[2]
		self.g.xi4 = self.var*xi[3]

		f = Function(self.function_space)
		f.interpolate(self.a)

		return f
