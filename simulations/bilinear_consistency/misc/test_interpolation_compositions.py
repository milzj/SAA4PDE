"""
Let fh be a FE function (represented in a nodal basis) defined on the FE space Vh and g continuous.

If we interpolate g(fh) onto Vh we should obtain the same coefficients
as g(fh.vector()[:]).

If Vh is DG0 and we project g(fh) onto Vh we should obtain the same coefficients
as g(fh.vector()[:]).

We verify these facts empirically.
"""

import numpy as np
import fenics
import pytest

@pytest.mark.parametrize("n", [32, 64, 128])
@pytest.mark.parametrize("fun", [("exp(f)", lambda x: np.exp(x)), ("max(0.0, f)", lambda x: np.maximum(x, 0.0))])
@pytest.mark.parametrize("degree_element", [(0, "DG"), (1, "CG")])
def test_interpolation(n, fun, degree_element):

	fun_str = fun[0]
	fun_np = fun[1]
	degree, element = degree_element
	atol = 1e-15


	mesh = fenics.UnitSquareMesh(n,n)
	V = fenics.FunctionSpace(mesh, element, degree)

	f_str = "sin(2.0*pi*x[0])*sin(2.0*pi*x[1])*exp(2.0*x[0])"
	f_expr = fenics.Expression(f_str, degree = degree)
	# Define field as composition
	g_expr = fenics.Expression("{}".format(fun_str), f=f_expr, degree = degree)

	h_str = "{}".format(fun_str)
	h_str = h_str.replace("f", f_str)
	h_expr = fenics.Expression("{}".format(fun_str), f=f_expr, degree = degree)
	h = fenics.Function(V)
	h.interpolate(h_expr)

	f = fenics.Function(V)
	f.interpolate(f_expr)

	# Interpolate fun expression field
	g = fenics.Function(V)
	g.interpolate(g_expr)

	# Apply fun_np to interpolated expression
	v = fenics.Function(V)
	v.vector()[:] = fun_np(f.vector()[:])

	# Are h and g equal?
	assert fenics.errornorm(v, g, degree_rise = 0) < atol
	assert fenics.errornorm(g, v, degree_rise = 0) < atol
	assert fenics.errornorm(v, h, degree_rise = 0) < atol
	assert fenics.errornorm(h, v, degree_rise = 0) < atol

	assert fenics.norm(v) > 0.0


@pytest.mark.parametrize("n", [32, 64, 128])
@pytest.mark.parametrize("fun", [("exp(f)", lambda x: np.exp(x)), ("max(0.0, f)", lambda x: np.maximum(x, 0.0))])
@pytest.mark.parametrize("degree", [0])
@pytest.mark.parametrize("element", ["DG"])
def test_projection(n, fun, degree, element):

	fun_str = fun[0]
	fun_np = fun[1]

	rtol = 1e-15

	mesh = fenics.UnitSquareMesh(n,n)
	V = fenics.FunctionSpace(mesh, element, degree)

	f_str = "sin(2.0*pi*x[0])*sin(2.0*pi*x[1])*exp(2.0*x[0])"
	f_expr = fenics.Expression(f_str, degree = degree)
	# Define field as composition
	g_expr = fenics.Expression("{}".format(fun_str), f=f_expr, degree = degree)

	h_str = "{}".format(fun_str)
	h_str = h_str.replace("f", f_str)
	h_expr = fenics.Expression("{}".format(fun_str), f=f_expr, degree = degree)
	h = fenics.Function(V)
	h = fenics.project(h_expr, V)

	f = fenics.Function(V)
	f = fenics.project(f_expr, V)

	# Interpolate fun expression field
	g = fenics.Function(V)
	g = fenics.project(g_expr, V)

	# Apply fun_np to interpolated expression
	v = fenics.Function(V)
	v.vector()[:] = fun_np(f.vector()[:])

	# Are h and g equal?
	assert fenics.errornorm(v, g, degree_rise = 0)/fenics.norm(v) < rtol
	assert fenics.errornorm(g, v, degree_rise = 0)/fenics.norm(v) < rtol
	assert fenics.errornorm(v, h, degree_rise = 0)/fenics.norm(v) < rtol
	assert fenics.errornorm(h, v, degree_rise = 0)/fenics.norm(v) < rtol




if __name__ == "__main__":


	test_interpolation(32, ("exp(f)", lambda x: np.exp(x)), 0, "DG")
