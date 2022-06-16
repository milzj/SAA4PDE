import numpy as np

from reference_sampler import ReferenceSampler

class DiscreteSampler(ReferenceSampler):

	def __init__(self, lb=-1.0, ub = 1.0, grid_points=12):
		"""
		Approximates each interval [-1,1] in [-1,1]^4 using
		a uniform mesh with n grid points.
		"""

		super().__init__(lb = lb, ub = ub, grid_points = grid_points)


		self._seed = 1
		self.__seed = []

	@property
	def seed(self):
		return self._seed

	@seed.setter
	def seed(self, seed):
		self._seed = seed
		assert np.sum([self.__seed[i] == self._seed for i in range(len(self.__seed))]) == 0
		self.__seed.append(seed)


	def bump_seed(self):
		self._seed += 4
		assert np.sum([self.__seed[i] == self._seed for i in range(len(self.__seed))]) == 0
		self.__seed.append(self._seed)

	def sample(self, sample_index, N=1, replace=False):
		"""Generates a random sample from a discrete distribution.

		Parameters:
		----------
		N : int
			sample size
		seed : int
			random seed
		replace : boolean, optional
			Default is False.


		Returns:
		--------


		"""
		self.bump_seed()

		a = self.a
		b = self.b
		c = self.c
		d = self.d

		np.random.seed(self.seed)
		aN = np.random.choice(a, size=N, replace=replace)
		np.random.seed(self.seed+1)
		bN = np.random.choice(b, size=N, replace=replace)
		np.random.seed(self.seed+2)
		cN = np.random.choice(c, size=N, replace=replace)
		np.random.seed(self.seed+3)
		dN = np.random.choice(d, size=N, replace=replace)

		return [aN, bN, cN, dN]


if __name__ == "__main__":

	sampler = DiscreteSampler()
	sample = sampler.sample(0, N=1)
	assert len(sample) == 4
