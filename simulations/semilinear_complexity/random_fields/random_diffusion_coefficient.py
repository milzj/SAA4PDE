import numpy as np

from dolfin import *
from dolfin_adjoint import *

class Kappa(UserExpression):

	def __init__(self, m=25, **kwargs):

		super(Kappa, self).__init__(**kwargs)
		self.m = m
		self._sample1 = np.zeros(m)
		self._sample2 = np.zeros(m)

	def eval(self, value, x):

		m = self.m
		s1 = self._sample1
		s2 = self._sample2
		v = 0.0

		if x[0] <= 0.5:

			for k in range(m):
				v += 2.5*np.sin(4*(k+1)*s1[k]*np.pi*x[0])*np.sin(4*(k+1)*np.pi*s2[k]*x[1])/(k+1)**2

			v = np.exp(v)
		else:

			for k in range(m):
				v += s2[k]*np.cos((10+s2[k])*x[0]*x[1])/(k+1)**2

			v = np.abs(10.0*v) + 1.5

		value[0] = v


	def value_shape(self):

		return (1,)


def KappaExpr(m=25, mesh=None, element=None, mpi_comm=MPI.comm_world):

	kappa_1_str = ""
	kappa_2_str = ""

	P = ["p{}".format(i) for i in range(m)]
	Q = ["q{}".format(i) for i in range(m)]

	for k in range(m):

		kappa_1_str += "+"
		kappa_1_str += "2.5*sin(4*({}+1)*s1[k]*pi*x[0])*sin(4*({}+1)*pi*s2[k]*x[1])/pow({}+1,2)".format(1.0*k, 1.0*k, 1.0*k)
		kappa_1_str = kappa_1_str.replace("s1[k]", P[k])
		kappa_1_str = kappa_1_str.replace("s2[k]", Q[k])

		kappa_2_str += "+"
		kappa_2_str += "10.0*s2[k]*cos((10+s2[k])*x[0]*x[1])/pow({}+1,2)".format(1.0*k)
		kappa_2_str = kappa_2_str.replace("s2[k]", Q[k])

	kappa_1_str = "exp({})".format(kappa_1_str)
	kappa_2_str = "abs({})+1.5".format(kappa_2_str)


	kappa_str = "x[0] <= 0.5 ? {} : {}".format(kappa_1_str, kappa_2_str)

	parameters = {str : 0.0 for str in P}
	parameters.update({str: 0.0 for str in Q})

	kappa = Expression(kappa_str, degree=0, domain=mesh, element=element, mpi_comm = mpi_comm, **parameters)

	return kappa, parameters




if __name__ == "__main__":

	import matplotlib.pyplot as plt
	from base.savefig import savefig
	from stats.surface_function import surface_function

	n = 64

	atol = 1e-14

	mesh = UnitSquareMesh(n,n)
	V = FunctionSpace(mesh, "DG", 0)
	v = Function(V)

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

		v.interpolate(kappa)

		kappa_expr.user_parameters.update(parameters)

		assert errornorm(kappa_expr, kappa, degree_rise = 0, mesh=mesh) < atol

		c = plot(v)
		plt.colorbar(c)

		savefig("output/kappa={}".format(i))

		surface_function(v, n)
		savefig("output/kappa_surface={}".format(i))



