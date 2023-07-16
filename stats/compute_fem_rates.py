import numpy as np

def compute_fem_rates(E, h, norm_types = {"L2"}, num_drop=0):
	"""Computes convergence rates.

	Computes c and r assuming E = c h^r given a list of errors E and of mesh widths h.

	Parameters:
	----------
	E: list, ndarray

	h: list, ndarray

	norm_types : list (optional)
		default is {"L2"}
	num_drop: int (optional)
		Number of data points to be dropped.

	Returns:
	-------
	rates : Dict

	rate_constant : Dict

	"""

	rates = {}
	rate_constant = {}
	num_levels = len(h)

	assert len(E) == len(h)

	# output
	e = np.zeros(num_levels)
	# design matrix
	X = np.ones((num_levels, 2)); X[:, 1] = np.log(h)

	for norm_type in sorted(norm_types):

		rates[norm_type] = []
		rate_constant[norm_type] = []

		for l in range(1, num_levels):

			r = np.log(E[l]/E[l-1]) / np.log(h[l]/h[l-1])
			rates[norm_type].append(r)

		mean = np.mean(rates[norm_type])
		median = np.median(rates[norm_type])

		# Least squares applied to log(error) = log(c)  + r log(h)

		_nl = num_levels-num_drop

		e[:] = [E[l] for l in range(num_levels)]

		e = e[0:_nl]
		X = X[0:_nl, ::]
		x, residudals, rank, s = np.linalg.lstsq(X, np.log(e), rcond=None)

		# x[1] = r and c = np.exp(x[0))
		rate_constant[norm_type] = [x[1], np.exp(x[0])]

	return rates, rate_constant
