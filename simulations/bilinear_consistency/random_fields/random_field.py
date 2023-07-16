import numpy as np
import itertools
from fenics import *
from dolfin_adjoint import *

class RandomField(object):

	def __init__(self, mean=0.0, l=0.1, m=20, mpi_comm=MPI.comm_self):
		"""Implements an adaption of the random field define  in eq. (9.50)  in Ref. [1].
		Arguments:
		----------
			mean : float
				scalar mean (default .5)
			l : float
				correlation length (positive float, default 0.1)
			m : int
				number of addends (default 20)
			mpi_comm : MPI communicator
				MPI communicator (default dolfin's MPI.comm_self)

		References:
		----------

		[1]	G. J. Lord, C. E. Powell, and T. Shardlow, An Introduction to Computational Stochastic
			PDEs, Cambridge Texts Appl. Math. 50, Cambridge University Press, Cambridge, 2014,
			https://doi.org/10.1017/CBO9781139017329

		"""

		self.mean = mean
		self.l = l
		self.m = m

		self.mpi_comm = mpi_comm

		indices = list(itertools.product(range(1,self.m), repeat=2))
		indices = sorted(indices, key=lambda x:x[0]**2+x[1]**2)
		self.indices = indices[0:self.m]

	@property
	def function_space(self):
		return self._function_space

	@function_space.setter
	def function_space(self, function_space):
		self._function_space = function_space
		self.eigenpairs()

	def eigenpairs(self):
		"""Computes eigenvalues and eigenfunctions."""

		function_space = self.function_space

		eigenvalues = np.zeros(len(self.indices))
		eigenfunctions = []

		for i, pair in enumerate(self.indices):
			j, k = pair[0], pair[1]

			eigenvalue = .25*np.exp(-np.pi*(j**2+k**2)*self.l**2)
			eigenvalues[i] = eigenvalue

			fun = Expression("2.0*cos(pi*j*x[0])*cos(pi*k*x[1])", j=j, k=k, degree=0, mpi_comm=self.mpi_comm)

			eigenfunction = interpolate(fun, function_space)
			eigenfunctions.append(eigenfunction)

		self.eigenvalues = eigenvalues
		self.eigenfunctions = eigenfunctions


	def plot_eigenvalues(self, outdir):

		import matplotlib.pyplot as plt
		from base.savefig import savefig

		m = self.m
		e = self.eigenvalues
		plt.scatter(range(1,m+1), e, s=0.5)
		plt.xlabel("Index of eigenvalue")
		plt.ylabel("Eigenvalue")

		savefig(outdir  + "eigenvalues")

		plt.scatter(range(m), e, s=0.5)
		plt.xlabel("Index of eigenvalue")
		plt.ylabel("Eigenvalue")
		plt.yscale("log")

		savefig(outdir + "log_eigenvalues")


	def _sample(self, sample):
		"""Computes a realization of the random field given a sample of the random variables.

		The sum defining the truncated KL expansion is evaluated from smallest
		to largest eigenvalue (to allow the potentially small values to contribute to the sum).
		"""

		y = np.zeros(self._function_space.dim())

		for i in np.argsort(self.eigenvalues):

			value = self.eigenvalues[i]
			fun = self.eigenfunctions[i]

			y += np.sqrt(value)*sample[i]*fun.vector().get_local()

		# add mean
		y += self.mean

		return y

