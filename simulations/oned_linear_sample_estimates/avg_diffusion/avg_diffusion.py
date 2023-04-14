import numpy as np
import itertools
from fenics import *
from dolfin_adjoint import *

from avg_field import avg_mknrandom_field

class AVGMKNRandomField(object):

    def __init__(self, var=1.):
        self.var = var
        return None

    @property
    def function_space(self):
        return self._function_space

    @function_space.setter
    def function_space(self, function_space):

        self._function_space = function_space
        mpi_comm = function_space.mesh().mpi_comm()
        var = self.var

        self.f = avg_mknrandom_field.avg_mknrandom_field(function_space, var=var)

    def sample(self, sample=None):
        return self.realization(sample=sample)

    def realization(self, sample=None):
        return self.f
