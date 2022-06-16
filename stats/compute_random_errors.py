from stats import LuxemburgNorm
import numpy as np

def compute_random_errors(errors):
	"""Compute statistics

	Compute for each key in errors, a number of statitics.

	Args:
		errors : dict

	Returns:
		errors_stats

	"""
	errors_stats = {}

	lux_norm = LuxemburgNorm()

	# Inline function won't do here (we want a descriptive *.__name__)
	def root_mean_squared(x):
		return np.sqrt(np.mean(np.power(x, 2)))

	def luxemburg_norm(x):
		return lux_norm.evaluate(x)

	Functions = [	np.mean,\
			root_mean_squared,\
			np.max,\
			np.median,\
			luxemburg_norm]

	# Compute statistics of error
	for e in errors.keys():

		_errors_stats = {}

		for func in Functions:
			_errors_stats[func.__name__] = func(errors[e])

		errors_stats[e] = _errors_stats


	return errors_stats
