import numpy as np

from dolfin import *
from dolfin_adjoint import *

class RHS(UserExpression):

	def __init__(self, m=25, **kwargs):

		super(RHS, self).__init__(**kwargs)
		self.m = m
		self._sample4 = np.zeros(m)

	def eval(self, value, x):

		m = self.m
		s = self._sample4
		v = 0.0

		if x[0] <= 0.75 + 0.5*s[0]:

			for k in range(m):
				v += 5.0*s[k]*x[0]*x[1]*np.cos(4.0*np.pi*(k+1)*x[0])*np.sin(4.0*(k+1)*np.pi*x[1])/(k+1)**2

		else:

			for k in range(m):
				v += np.abs(3.0*x[1]*s[k]*np.sin(3*np.pi*x[1])*np.cos(3.0*np.pi*(x[0]-(k+1)*s[k]*x[1])))/(k+1)**2


		value[0] = v + 1.0

	def value_shape(self):

		return (1,)


def RHSExpr(m=25, mesh=None, element=None, mpi_comm=MPI.comm_world):

	g_1_str = ""
	g_2_str = ""

	T = ["t{}".format(i) for i in range(m)]

	for k in range(m):

		g_1_str += "+"
		g_1_str += "5.0*s[k]*x[0]*x[1]*cos(4.0*pi*({}+1)*x[0])*sin(4.0*({}+1)*pi*x[1])/pow({}+1,2)".format(1.0*k, 1.0*k, 1.0*k)
		g_1_str = g_1_str.replace("s[k]", T[k])

		g_2_str += "+"
		g_2_str += "abs(3.0*x[1]*s[k]*sin(3*pi*x[1])*cos(3.0*pi*(x[0]-({}+1)*s[k]*x[1])))/pow({}+1,2)".format(1.0*k, 1.0*k)
		g_2_str = g_2_str.replace("s[k]", T[k])

	g_str = "x[0] <= 0.75 + 0.5*{} ? 1.0 + {} : 1.0 + {}".format("t0", g_1_str, g_2_str)

	parameters = {str : 0.0 for str in T}

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
	g = RHS(m=m, element=V.ufl_element(), domain = mesh)

	g_expr, parameters = RHSExpr(m, mesh, V.ufl_element())

	for i in range(10):

		sample4 = np.random.uniform(-1, 1, m)
		g._sample4 = sample4

		for k in range(m):
			parameters["t{}".format(k)] = sample4[k]

		g_expr.user_parameters.update(parameters)

		assert errornorm(g_expr, g, degree_rise = 0, mesh=mesh) < atol

		v.interpolate(g)

		c = plot(v)
		plt.colorbar(c)

		savefig("output/b={}".format(i))

		surface_function(v, n)
		savefig("output/b_surface={}".format(i))
