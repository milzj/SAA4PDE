import pytest

import numpy as np
import dolfin
import moola

def problem(n):

	mesh = dolfin.UnitSquareMesh(dolfin.MPI.comm_self, n,n)

	U = dolfin.FunctionSpace(mesh, "DG", 0)

	u = dolfin.Function(U)
	v = dolfin.Function(U)
	w = dolfin.Function(U)

	u_vec = np.random.randn(U.dim())
	v_vec = np.random.randn(U.dim())
	w_vec = np.multiply(u_vec, v_vec)

	u.vector()[:] = u_vec
	v.vector()[:] = v_vec

	w.vector()[:] = w_vec

	return u, v, w


@pytest.mark.parametrize("n", [32, 64, 128, 256])
def test_dolfin_scale(n):
	"""Tests if dolfin's command *= applies componentwise."""

	atol = 1e-15
	degree_rise = 0

	u, v, w = problem(n)

	u_vec = u.vector()
	u_vec *= v.vector()

	assert dolfin.errornorm(u, w, degree_rise = degree_rise) < atol


@pytest.mark.parametrize("n", [32, 64, 128, 256])
def test_moola_scale(n):
	"""Tests if moola's scale applies componentwise."""

	atol = 1e-15
	degree_rise = 0

	u, v, w = problem(n)

	u_moola = moola.DolfinPrimalVector(u)
	v_moola = moola.DolfinPrimalVector(v)
	w_moola = moola.DolfinPrimalVector(w)

	u_moola.scale(v_moola.data.vector())

	assert dolfin.errornorm(u_moola.data, w_moola.data, degree_rise = degree_rise) < atol

