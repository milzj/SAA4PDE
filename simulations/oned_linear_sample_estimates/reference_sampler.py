import numpy as np


class ReferenceSampler(object):

	def __init__(self, lb=-1.0, ub = 1.0, grid_points=12):
		"""Approximates the uniform distribution on [-lb,ub]^4

		Approximates each interval [-lb,ub] in [-lb,ub]^4 using
		a uniform mesh with n grid points.

		Parameters:
		-----------
		lb, ub : float
			Lower and upper bounds used to define intervals.
		n : int
			Number of grid points used to discretize [lb, ub].
		"""

		x = np.linspace(lb, ub, grid_points)
		a, b, c, d = np.meshgrid(x,x,x,x)

		self.a = np.ndarray.flatten(a)
		self.b = np.ndarray.flatten(b)
		self.c = np.ndarray.flatten(c)
		self.d = np.ndarray.flatten(d)

	def sample(self, sample_index):
		"""Generates 'samples' from a discrete distribution."""
		return [self.a[sample_index], self.b[sample_index],
			self.c[sample_index], self.d[sample_index]]



if __name__ == "__main__":

	sampler = ReferenceSampler()

	assert np.all(np.array(sampler.sample(0)) == -1.0)
	assert np.all(np.array(sampler.sample(144**2-1)) == 1.0)



