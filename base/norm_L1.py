import fenics

def norm_L1(v):
	"""Evaluate L1 norm of FE function.

	Function requires v.function_space()
	is the FE space of piecewise constant functions
	with nodal basis.

	norm(v, L1) should be norm(M_h v_vec, l1), where
	M_h is the stiffness matrix and v_vec are the coefficients
	used to define the FE function v.

	See [Pieper2015, pp. 97--98 and p. 141] for the
	computations.


	References:
	-----------

	K. Pieper, Finite element discretization and efficient numerical solution of el-
	liptic and parabolic sparse control problems, Dissertation, TUM, Munich 2015,
	http://mediatum.ub.tum.de/?id=1241413
	"""

	V = v.function_space()
	element = V.element()
	signature = element.signature()

	if signature == "FiniteElement('Discontinuous Lagrange', triangle, 0)":

		d = fenics.assemble(fenics.derivative(.5*v**2*fenics.dx, v))
		d.abs()
		return d.sum()

	else:
		raise ValueError("Finite element of v is {} but DG0 is required.".format(signature))
