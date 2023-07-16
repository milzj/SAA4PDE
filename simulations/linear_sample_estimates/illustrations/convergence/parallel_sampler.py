import numpy as np
import sys, os

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
parentparentdir = os.path.dirname(parentdir)
sys.path.append(parentparentdir)

from reference_sampler import ReferenceSampler

class ParallelSampler(ReferenceSampler):

	def __init__(self, N, seed):

		super().__init__()
		np.random.seed(seed)
		self._a = np.random.choice(self.a, size=N)
		np.random.seed(seed+1)
		self._b = np.random.choice(self.b, size=N)
		np.random.seed(seed+2)
		self._c = np.random.choice(self.c, size=N)
		np.random.seed(seed+3)
		self._d = np.random.choice(self.d, size=N)

	def sample(self, sample_index):
		"""Generates 'samples' from a discrete distribution."""
		return [self._a[sample_index], self._b[sample_index],
			self._c[sample_index], self._d[sample_index]]



if __name__ == "__main__":

	N = 100
	seed = 0
	sampler = ParallelSampler(N, seed)

	print(sampler.sample(0))
	print(sampler.sample(5))
	print(sampler.sample(N-1))


