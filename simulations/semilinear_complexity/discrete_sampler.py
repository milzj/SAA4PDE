import numpy as np

from reference_sampler import ReferenceSampler

class DiscreteSampler(object):

	def __init__(self):
		"""
		Approximates each interval [-1,1] in [-1,1]^4 using
		a uniform mesh with n grid points.
		"""

		super().__init__()


		self._seed = 10000
		self.__seed = []

		reference_sampler = ReferenceSampler(2**13)

		self.samples = reference_sampler.samples


	@property
	def seed(self):
		return self._seed

	@seed.setter
	def seed(self, seed):
		self._seed = seed
		assert np.sum([self.__seed[i] == self._seed for i in range(len(self.__seed))]) == 0
		self.__seed.append(seed)


	def bump_seed(self):
		self._seed += 1
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

		np.random.seed(self._seed)
		Z = np.random.uniform(-1., 1., 25*4)
#		idx = np.random.choice(range(0,2**13), 1)[0]
#		Z = self.samples[idx]

		return np.split(Z, 4)


if __name__ == "__main__":

	sampler = DiscreteSampler()
	sample = sampler.sample(0, N=1)
	print(sample)

	sample = sampler.sample(1, N=1)
	print(sample)
