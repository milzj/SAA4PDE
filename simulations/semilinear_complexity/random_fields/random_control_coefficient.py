import numpy as np

from dolfin import *
from dolfin_adjoint import *

class RCC(UserExpression):

	def __init__(self, m=25, **kwargs):

		super(RCC, self).__init__(**kwargs)
		self.m = m
		self._sample3 = np.zeros(m)

	def eval(self, value, x):

		m = self.m
		s = self._sample3
		v = 0.0

		for k in range(m):
			v += 10.0*s[k]*np.sin((5+k)*s[k]*x[0]*x[1])*np.cos((5+k)*s[k]*x[0]*x[1])/(k+1)**2

		value[0] = np.maximum(1.0, v)

	def value_shape(self):

		return (1,)


def RCCExpr(m=25, mesh=None, element=None, mpi_comm=MPI.comm_world):

	g_str = ""

	R = ["r{}".format(i) for i in range(m)]

	for k in range(m):

		g_str += "+"
		g_str += "10.0*s[k]*sin((5+{})*s[k]*x[0]*x[1])*cos((5+{})*s[k]*x[0]*x[1])/pow({}+1,2)".format(k,k,k)
		g_str = g_str.replace("s[k]", R[k])

	g_str = "max(1.0, {})".format(g_str)

	parameters = {str : 0.0 for str in R}

	g = Expression(g_str, degree=0, domain=mesh, element=element, mpi_comm = mpi_comm, **parameters)

	return g, parameters




if __name__ == "__main__":

	import matplotlib.pyplot as plt
	from base.savefig import savefig
	from stats.surface_function import surface_function

	n = 64
	mesh = UnitSquareMesh(n,n)
	V = FunctionSpace(mesh, "DG", 0)
	v = Function(V)

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

		v.interpolate(g)

		c = plot(v)
		plt.colorbar(c)

		savefig("output/g={}".format(i))

		surface_function(v, n)
		savefig("output/g_surface={}".format(i))
