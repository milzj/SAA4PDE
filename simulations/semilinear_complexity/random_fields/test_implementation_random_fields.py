import pytest


import numpy as np
from dolfin import *

from random_diffusion_coefficient import Kappa, KappaExpr
from random_righthand_side import RHS, RHSExpr
from random_control_coefficient import RCC, RCCExpr

def test_implementation_kappa():

	n = 64
	atol = 1e-14

	mesh = UnitSquareMesh(n,n)
	V = FunctionSpace(mesh, "DG", 0)

	m = 25
	kappa = Kappa(m=m, element=V.ufl_element(), domain = mesh, degree = 0)

	kappa_expr, parameters = KappaExpr(m, mesh, V.ufl_element())


	for i in range(10):

		sample1 = np.random.uniform(-1, 1, m)
		sample2 = np.random.uniform(-1, 1, m)
		kappa._sample1 = sample1
		kappa._sample2 = sample2

		for k in range(m):
			parameters["p{}".format(k)] = sample1[k]
			parameters["q{}".format(k)] = sample2[k]


		kappa_expr.user_parameters.update(parameters)

		assert errornorm(kappa_expr, kappa, degree_rise = 0, mesh=mesh) < atol



def test_implementation_g():

	n = 64
	mesh = UnitSquareMesh(n,n)
	V = FunctionSpace(mesh, "DG", 0)

	atol = 1e-14

	m = 25
	g = RCC(m=m, element=V.ufl_element(), domain = mesh)


	g_expr, parameters = RCCExpr(m, mesh, V.ufl_element())

	for i in range(10):

		sample3 = np.random.uniform(-1, 1, m)
		g._sample3 = sample3

		for k in range(m):
			parameters["r{}".format(k)] = sample3[k]

		g_expr.user_parameters.update(parameters)

		assert errornorm(g_expr, g, degree_rise = 0, mesh=mesh) < atol



def test_implementation_b():

	n = 64
	mesh = UnitSquareMesh(n,n)
	V = FunctionSpace(mesh, "DG", 0)

	atol = 1e-14

	m = 25
	g = RHS(m=m, element=V.ufl_element(), domain = mesh)

	g_expr, parameters = RHSExpr(m, mesh, V.ufl_element())

	for i in range(10):

		sample4 = np.random.uniform(-1, 1, m)
		g._sample4 = sample4

		for k in range(m):
			parameters["t{}".format(k)] = sample4[k]

		g_expr.user_parameters.update(parameters)

		assert errornorm(g_expr, g, degree_rise = 0, mesh=mesh) < atol

