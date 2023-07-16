"""Test problem ccdist1

References:
-----------

OPTPDE - A Collection of Problems in PDE-Constrained Optimization
http://www.optpde.net/ccdist1

sect. 2.9.2 in

F. Tröltzsch. Optimal Control of Partial Differential Equations,
volume 112 of Graduate Studies in Mathematics. 
American Mathematical Society, Providence, 2010.

"""


from dolfin import *
import numpy as np


class Adjoint(UserExpression):
	"""Implements the optimal adjoint state.

	Note: We implement the negative `p`.

	TODO: Implement as Expression.
	"""

	def __init__(self,  **kwargs):

		super(Adjoint, self).__init__(**kwargs)

	def eval(self, value, x):

		r2 = (.5 - x[0])**2 + (.5 - x[1])**2

		value[0] = 12.0*r2-1.0/3.0

	def value_shape(self):

		return (1,)


class Solution(UserExpression):
	"""Implements optimal control.

	The implementation is based on the MATLAB implementation
	provided on the OPTPDE webpage, that is, we project the
	optimal adjoint state onto the feasible set. See also
	p. 82 in Tröltzsch (2010).
	"""

	def __init__(self, **kwargs):

		super(Solution, self).__init__(**kwargs)

		self.adjoint = Adjoint(**kwargs)

	def eval(self, value, x):


		val = [0.0]
		self.adjoint.eval(val, x)

		value[0] = np.clip(val[0], 0.0, 1.0)

	def value_shape(self):

		return (1,)



def DesiredState(element, domain):

	return Expression("-142.0/3.0 + 12.0*pow(.5-x[0],2) + 12.0*pow(.5-x[1],2)",
			element = element, domain = domain)

def BoundaryCoefficient():
	"boundary observation coefficient"

	return Constant(-12.0)


class UncontrolledForce(UserExpression):
	"""Implements the uncontrolled force term."""

	def __init__(self, **kwargs):

		super(UncontrolledForce, self).__init__(**kwargs)

		self.solution = Solution(**kwargs)

	def eval(self, value, x):

		val = [0.0]
		self.solution.eval(val, x)

		value[0] = 1.0 - val[0]

	def value_shape(self):

		return (1,)

