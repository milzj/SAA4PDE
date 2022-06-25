import numpy as np

from bilinear_sampler import BilinearSampler

class ReferenceSampler(object):

        def __init__(self, N):

                sampler = BilinearSampler()

                self._seed = 1
                sample_index = 0

                Z_vec = []
                for i in range(N):
                        Z = sampler.sample(sample_index)
                        Z_vec.append(Z)
                self.Z_vec = Z_vec


        def sample(self, sample_index):
                return self.Z_vec[sample_index]






