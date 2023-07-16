import pytest

import numpy as np
import dolfin

from problem import Regularizer

def relerrornorm(a,b, ord=np.inf):

	return np.linalg.norm(a-b)/np.linalg.norm(a)


@pytest.mark.parametrize("n", [32, 64, 128, 256])

def test_regularizer(n):
	"""Test implementation of regularizer and its derivatives."""

	degree_rise = 0

	atol = 1e-8
	rtol = 1e-13

	mesh = dolfin.UnitSquareMesh(dolfin.MPI.comm_self, n, n)
	U = dolfin.FunctionSpace(mesh, "DG", 0)

	u = dolfin.Function(U)
	u.vector()[:] = np.random.randn(U.dim())

	regularizer = Regularizer(u)

	u2_norm_true = .5*dolfin.norm(u, norm_type = "L2")**2
	assert np.isclose(u2_norm_true, regularizer(u), rtol = rtol, atol=atol)

	assert np.isclose(regularizer.derivative().inner(u.vector()), 2.0*u2_norm_true, rtol = rtol, atol=atol)

	v = dolfin.TestFunction(U)
	assert relerrornorm(dolfin.assemble(u*v*dolfin.dx), regularizer.derivative()) < rtol
