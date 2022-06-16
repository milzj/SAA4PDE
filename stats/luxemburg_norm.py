#!/usr/bin/env python

from scipy import optimize
import numpy as np

class LuxemburgNorm(object):
	"""Estimate the Luxemburg-2-norm.

	We use the fact that the Luxemburg-2-norm
	can be obtained by solving a one-dimensional
	equation (provided that the Luxemburg norm
	is between zero and infinity).

	The Luxemburg norm is computed using Brent's method.

	Youngs' function is given by exp(x^2)-1.

	Methods
	-------

	evaluate
		Computes Luxemburg-2-norm

	fun
		Defines root function

	"""

	def __init__(self):
		self.atol = 1e-14

	def fun(self, tau):
		"""Define root function."""

		return np.mean(np.exp(np.power(self.Z, 2.0)/tau**2)) - 2.0

	def evaluate(self, Z):
		"""Compute Luxemburg norm using Brent's method."""

		if isinstance(Z, np.ndarray) == False and isinstance(Z, list) == False:
			raise TypeError('Input Z should be an np.ndarray or a list.')

		self.Z = Z

		x0 = np.sqrt(np.mean(np.power(self.Z, 2.0)))
		x1 = np.max(np.absolute(self.Z))/np.sqrt(np.log(2.))

		if x1 <= self.atol:
			return x1
		elif np.abs(self.fun(x1)) <= self.atol and np.abs(self.fun(x0)) <= self.atol:
			return (x0+x1)/2.
		else:
			return optimize.brentq(self.fun, x0, x1)

