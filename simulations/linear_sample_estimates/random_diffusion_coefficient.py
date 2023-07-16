import numpy as np
import itertools
from fenics import *
from dolfin_adjoint import *

class RandomDiffusionCoefficient(object):

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

		self.degree = 3
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
		g_str = "a*v*cos(1.1*pi*x[0])+b*v*cos(1.2*pi*x[0])+c*v*sin(1.3*pi*x[1])+d*v*sin(1.4*pi*x[1])"
		self.g = Expression(g_str, a=0.0, b=0.0, c=0.0, d=0.0, v=self.var, degree=self.degree, mpi_comm=mpi_comm)
		self.kappa = Expression("m+exp(g)", m=self.mean, g=self.g, degree = self.degree, mpi_comm=mpi_comm)

	def sample(self, sample=None):
		return self.realization(sample=sample)

	def realization(self, sample=None):
		"""Computes a realization of the random field.

		If sample=None, then four uniform random variables
		are generated and thee seed is increased by one.
		"""

		self.g.a = sample[0]
		self.g.b = sample[1]
		self.g.c = sample[2]
		self.g.d = sample[3]

		return self.kappa
