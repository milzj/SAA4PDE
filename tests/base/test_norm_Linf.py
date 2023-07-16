import pytest

import dolfin
import numpy as np
from base import norm_Linf


@pytest.mark.parametrize("seed", [1234, 12345, 123456])
def test_norm_Linf(seed):

	n = 128
	mesh = dolfin.UnitSquareMesh(dolfin.MPI.comm_self, n, n)

	# both function spaces are nodal
	V = dolfin.FunctionSpace(mesh, "CG", 1)
	W = dolfin.FunctionSpace(mesh, "DG", 0)

	u = dolfin.Function(W)
	v = dolfin.Function(V)

	np.random.seed(seed)
	u_vec = np.random.randn(W.dim())
	u.vector().set_local(u_vec)
	u_inf = norm_Linf(u)
	assert u_inf == np.linalg.norm(u_vec,ord=np.inf)

	np.random.seed(seed)
	v_vec = np.random.rand(V.dim())
	v.vector().set_local(v_vec)
	v_inf = norm_Linf(v)
	assert v_inf == np.linalg.norm(v_vec,ord=np.inf)


