import numpy as np
import itertools
from fenics import *
from dolfin_adjoint import *

from avg_mknrandom_field import avg_mknrandom_field

class AVGMKNRandomField(object):

    def __init__(self):
        return None

    @property
    def function_space(self):
        return self._function_space

    @function_space.setter
    def function_space(self, function_space):

        self._function_space = function_space
        mpi_comm = function_space.mesh().mpi_comm()

        self.f = avg_mknrandom_field.avg_mknrandom_field(function_space)

    def sample(self, sample=None):
        return self.realization(sample=sample)

    def realization(self, sample=None):
        return self.f
