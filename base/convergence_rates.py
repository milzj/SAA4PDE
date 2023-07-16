def convergence_rates(E_values, eps_values, show=True):
	"""
	Source https://github.com/dolfin-adjoint/pyadjoint/blob/master/pyadjoint/verification.py

	Added print("Computed convergence rates: {}".format(r))
	"""

	from numpy import log
	r = []
	for i in range(1, len(eps_values)):
		r.append(log(E_values[i] / E_values[i - 1])
			/ log(eps_values[i] / eps_values[i - 1]))
	if show:
		print("Residuals:{}".format(E_values))
		print("Computed convergence rates: {}".format(r))
	return r
