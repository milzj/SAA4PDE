import numpy as np

class BilinearSampler(object):

	def __init__(self):

		self._seed = 1
		self._mean = np.zeros(2*200)

	@property
	def seed(self):
		return self._seed


	def bump_seed(self):
		self._seed += 400

	def sample(self, sample_index, N=1):

		self.bump_seed()
		np.random.seed(self._seed)
		Z = np.random.uniform(-np.sqrt(3), np.sqrt(3), 400)

		return Z

	@property
	def mean(self):
		return self._mean


