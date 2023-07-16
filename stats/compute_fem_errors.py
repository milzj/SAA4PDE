import dolfin

def compute_fem_errors(u, uh, degree_rise = 3, mesh=None):
	"""Compute and return several norms of the error u-uh.

	The code uses using dolfin.errornorm to compute the
	norms of u-uh.

	Parameters:
	-----------
	u : dolfin.Function
		"Exact" solution
	uh : dolfin.Function
		Approximate solution
	degree_rise : int (optional)
		default is 3.
	mesh : dolfin.Mesh (optional)
		default is None


	Returns:
	--------
		errors : dictionary

	Note:
	-----

	TODO: Implement Linfty-norm computation.
	"""

	# L2 norm
	E1 = dolfin.errornorm(u, uh, norm_type="L2", degree_rise=degree_rise, mesh=mesh)

	# H1 seminorm
	E2 = dolfin.errornorm(u, uh, norm_type="H10", degree_rise=degree_rise, mesh=mesh)

	# Linfty norm
	E3 = -1.0

	# Collect error measures in a dictionary with self-explanatory keys
	errors = {"L2": E1,
		"H10": E2,
		"Linf": E3}

	return errors


