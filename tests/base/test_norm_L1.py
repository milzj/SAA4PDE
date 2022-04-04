import pytest

from dolfin import *
import numpy as np
from base import norm_L1

"""
References:
-----------
Jan Blechta (2014), https://fenicsproject.org/qa/3329/ufl-l-1-norm-of-a-function/
"""

import dolfin
import numpy as np
from base import norm_L1

@pytest.mark.parametrize("seed", [1234, 12345, 123456])
def test_norm_L1(seed):

	n = 256
	mesh = UnitSquareMesh(dolfin.MPI.comm_self, n, n)

	W = FunctionSpace(mesh, "DG", 0)
	u = Function(W)

	np.random.seed(seed)
	u_vec = np.random.randn(W.dim())
	u.vector().set_local(u_vec)
	L1 = norm_L1(u)

	F = abs(u)*dx(None, {'quadrature_degre': 5})
	f = assemble(F)

	rel_err = (f-L1)/(1.0+f)
	assert rel_err == 0.0
