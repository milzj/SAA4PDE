import numpy as np
from scipy.stats import qmc

class ReferenceSampler(object):

	def __init__(self, Nref):

		p = 25
		self.mean = np.split(np.zeros(4*p), 4)

		sampler = qmc.Sobol(d=100, scramble=False)

		if not ((Nref & (Nref-1) == 0) and Nref != 0):
			raise ValueError("Nref is not 2**m for some natural number m.")

		self.samples = 2*sampler.random_base2(m=int(np.log2(Nref)))-1



	def sample(self, sample_index):
		"""Generates 'samples' from a discrete distribution."""
		return np.split(self.samples[sample_index], 4)





if __name__ == "__main__":

	sampler = ReferenceSampler()

	print(sampler.sample(0))
	print(sampler.sample(2499))



