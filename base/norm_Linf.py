def norm_Linf(v):
	"""Compute L^infinity norm of v.

	Function assumes that v.function_space() is nodal.
	"""

	v_max = v.vector().max()
	v_min = v.vector().min()

	return max(abs(v_max),abs(v_min))
