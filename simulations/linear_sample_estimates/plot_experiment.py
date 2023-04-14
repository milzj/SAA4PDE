from stats import load_dict, compute_fem_rates, compute_random_errors

from base import signif, lsqs_label, savefig

import numpy as np
from matplotlib import pyplot as plt
import itertools

from stats import figure_style
plt.rcParams['legend.loc'] = "upper right"

def plot_experiment(stats, outdir, filename, norm_type="L2", error_types=["luxemburg_norm", "mean"], n_drop=0, N_drop=0):
	"""Plot output

	Note: Does not check whether norm_type and error_types are available.
	"""

	labels = {"luxemburg_norm": "Luxemburg norm",
		"mean": r"$\mathbb{E}[\|u_{h, N}^*-u^*\|_{L^2(D)}]$"}

	labels = {"luxemburg_norm": "Luxemburg norm",
		"mean": "mean"}

	data_label = {"data": r"$\|u_{h, N}^*-u^*\|_{L^2(D)}$"}

	ylabel = ""
	filename = filename.replace(".", "_")

	experiments = {}
	errors = {}
	if len(error_types) == 2:
		markers = itertools.cycle(('s', 'o'))
		linestyles = itertools.cycle(('--', '-.'))
	else:
		markers = itertools.cycle(('o'))
		linestyles = itertools.cycle(('--'))

	# Find experiments, meaning all tuples (n, N)
	replications = stats.keys()
	
	for r in replications:
		i = 0
		experiments[r] = {}
		for n in sorted (stats[r].keys()):
	
			for N in sorted (stats[r][n].keys()):
				i += 1
				experiments[r][i] = (n, N)
	

	# Check if experiments are the same for each replication
	for r in list(replications)[:-1]:
		assert list(experiments[r].values()) == list(experiments[r+1].values())

	# Find all replications for each experiment (assumes at least one experiment)
	for e in experiments[1].values():

		n, N = e
		errors[e] = []
	
		for r in replications:
	
			s = stats[r][n][N][norm_type]
			errors[e].append(s)
	

	errors_stats = compute_random_errors(errors)
	# Find all unique n and N
	n_vec = []
	N_vec = []
	
	for e in errors.keys():
		n, N = e
		n_vec.append(n)
		N_vec.append(N)
	
	n_vec = sorted(list(set(n_vec)))
	N_vec = sorted(list(set(N_vec)))

	# Compute convergence rates
	rates_N = {}
	lsqs_rates_N = {}

	error_type = error_types[0]

	for error_type in error_types:

		rates_N[error_type] = {}
		lsqs_rates_N[error_type] = {}

		for N in N_vec:
			h = []
			for (n_, N_) in errors.keys():
				if N == N_:
					h.append(n_)
			h = np.array(sorted(h))
			E = [errors_stats[(n, N)][error_type] for n in h]
			h = 1.0/np.array(sorted(h))
			rates, rate_constant = compute_fem_rates(E, h, num_drop = n_drop)


			lsqs_rates_N[error_type][N] = rate_constant
			rates_N[error_type][N] = rate_constant[norm_type][0]


	rates_n = {}
	lsqs_rates_n = {}

	for error_type in error_types:

		rates_n[error_type] = {}
		lsqs_rates_n[error_type] = {}

		for n in n_vec:
			_N_vec = {N for (_, N) in [(n, N) for _,N in errors.keys()]}
			_N_vec = []
			for (n_, N_) in errors.keys():
				if n == n_:
					_N_vec.append(N_)

			_N_vec = np.array(sorted(_N_vec))

			E = [errors_stats[(n, N)][error_type] for N in _N_vec]
			rates, rate_constant = compute_fem_rates(E, _N_vec, num_drop = N_drop)
			lsqs_rates_n[error_type][n] = rate_constant
			rates_n[error_type][n] = rate_constant[norm_type][0]



	# N = n**2
	rates_n_nn = {}
	lsqs_rates_n_nn = {}

	tuples = {}
	for (_n, _N) in errors.keys():
		if _N == _n**2:
			tuples[(_n, _N)] = (_n, _N)

	for error_type in error_types:

		E = [errors_stats[e][error_type] for e in tuples]
		h = [1.0/_n for (_n, _N) in tuples]
		rates, rate_constant = compute_fem_rates(E, h)
		lsqs_rates_n_nn[error_type] = rate_constant


		rates_n_nn[error_type] = np.median(rates[norm_type])
		rates_n_nn[error_type] = rate_constant[norm_type][0]



	# Plot data
	# For each N, plot all n
	for N in N_vec:
	
		fig, ax = plt.subplots()
		tuples = {}
		for (_n, _N) in errors.keys():
			if _N == N:
				tuples[(_n, _N)] = (_n, _N)

		for e in tuples:
	
			n, _ = e
			for error_type in error_types:
				y = errors_stats[e][error_type]
				ax.scatter(n, y, marker=next(markers), color="black",
					label=labels[error_type])
			y = errors[e]
			n = n*np.ones(len(y))
	
			ax.scatter(n, y, marker="o", color="black", s=2, label=data_label["data"])

		# Plot least squares fit
		for error_type in error_types:

			s, t = lsqs_rates_N[error_type][N][norm_type]
			_n_vec = {n for (n,_) in [(j, N) for j,_ in errors.keys()]}
			_n_vec = np.array(sorted(_n_vec))
			_h_vec = 1.0/_n_vec
			y = t*_h_vec**s
			ax.plot(_n_vec, y, color="black", linestyle=next(linestyles),
				label=lsqs_label(t, s, "h"))

			ax.set_xscale("log", base=2)
			ax.set_yscale("log", base=2)
			ax.set_xlabel(r"$1/h$ (with $N={}$)".format(N,rates_N[error_type][N]))
			ax.set_ylabel(ylabel)


		# https://stackoverflow.com/questions/13588920/stop-matplotlib-repeating-labels-in-legend
		_handles, _labels = plt.gca().get_legend_handles_labels()
		by_label = dict(zip(_labels, _handles))
		plt.legend(by_label.values(), by_label.keys())

		figure_name = outdir + "/" + filename + "_N={}".format(N)
		savefig(figure_name)
	
	# For each n, plot all N
	for n in n_vec:
	
		fig, ax = plt.subplots()

		tuples = {}
		for (_n, _N) in errors.keys():
			if _n == n:
				tuples[(_n, _N)] = (_n, _N)

		x = []

		for e in sorted (tuples):
	
			_n, N = e
			# Plot realizations of error
			for error_type in error_types:
				y = errors_stats[e][error_type]
				ax.scatter(N, y, marker=next(markers), color="black",\
					label=labels[error_type])
			# Plot statistic of error
			y = errors[e]
			N = N*np.ones(len(y))
			ax.scatter(N, y, marker="o", color="black", s=2, label=data_label["data"])


		# Plot least squares fit
		for error_type in error_types:

			s, t = lsqs_rates_n[error_type][n][norm_type]
			_N_vec = {N for (_, N) in [(n, N) for _,N in errors.keys()]}
			_N_vec = np.array(sorted(_N_vec))
			y = t*_N_vec**s
			ax.plot(_N_vec, y, color="black", linestyle=next(linestyles),
					label=lsqs_label(t, s, "N"))

			ax.set_xscale("log", base=2)
			ax.set_yscale("log", base=2)
			ax.set_xlabel(r"$N$ (with $1/h={}$)".format(_n, rates_n[error_type][n]))


		_handles, _labels = plt.gca().get_legend_handles_labels()
		by_label = dict(zip(_labels, _handles))
#		plt.legend(by_label.values(), by_label.keys(), loc="upper right")
		plt.legend(by_label.values(), by_label.keys())

		figure_name = outdir + "/" + filename + "_n={}".format(n) + "_" + error_type
		savefig(figure_name)


	# For each n with N = n**2
	tuples = {}
	for (_n, _N) in errors.keys():
		if _N == _n**2:
			tuples[(_n, _N)] = (_n, _N)


	fig, ax = plt.subplots()

	for e in sorted (tuples):
	
		_n, _N = e
		# Plot realizations of error
		for error_type in error_types:
			y = errors_stats[e][error_type]
			ax.scatter(_n, y, marker=next(markers), color="black",\
				label=labels[error_type])

		# Plot statistic of error
		y = errors[e]
		_n_vec = _n*np.ones(len(y))
		ax.scatter(_n_vec, y, marker="o", color="black", s=2, label=data_label["data"])


	# Plot least squares fit
	for error_type in error_types:
		s, t = lsqs_rates_n_nn[error_type][norm_type]
		_n_vec = {_n for (_n, _) in tuples}
		_n_vec = np.array(sorted(_n_vec))
		_h_vec = 1.0/_n_vec
		y = t*_h_vec**s

		ax.plot(_n_vec, y, color="black", linestyle=next(linestyles), label=lsqs_label(t, s, "h"))

	ax.set_xscale("log", base=2)
	ax.set_yscale("log", base=2)
	ax.set_xlabel(r"$1/h$ (with $N=(1/h)^2$)")
	ax.set_ylabel(ylabel)

	handles, labels = plt.gca().get_legend_handles_labels()
	by_label = dict(zip(labels, handles))
	plt.legend(by_label.values(), by_label.keys(), loc="upper right")

	figure_name = outdir + "/" + filename + "_N=nn".format(n)
	savefig(figure_name)


if __name__ == "__main__":

	import sys

	N_drop = int(sys.argv[1])
	n_drop = int(sys.argv[2])
	outdir = sys.argv[3]

	filename = outdir.split("/")
	filename = filename[-1]


	try:
		stats = load_dict(outdir, filename)
	except FileNotFoundError:
		stats = {}
		for rank in range(48):
			_filename = filename + "_mpi_rank=" + str(rank)
			_stats = load_dict(outdir, _filename)
			stats.update(_stats)


	plot_experiment(stats, outdir, filename, N_drop=N_drop, n_drop=n_drop)
