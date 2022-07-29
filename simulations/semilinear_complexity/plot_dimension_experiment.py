
from stats import load_dict, compute_random_errors
from base import lsqs_label
import warnings
import numpy as np
import itertools, os

from plot_experiment import load_experiment
from stats import figure_style

from matplotlib import pyplot as plt

plt.rcParams.update({"legend.frameon": True, "legend.loc": "lower left"})
plt.rcParams.update({"legend.columnspacing": 1.0})

def plot_dimension_experiment(outdirs):
	"""Generate convergence plots.

	Parameters:
	----------
		outdir : string
			directory of experiment
		ndrop : int (optional)
			number of data points to be dropped for computing
			convergence rates using least squares.

	"""

	Stats = {}
	ncol = 1

	markers = itertools.cycle(('1', '+', 'x', '.'))
	line_styles = itertools.cycle(('solid', 'dotted', 'dashed'))

	# Plot
	fig, ax = plt.subplots()

	X_vec = {}
	Y_vec = {}

	Dates = ""

	for k in outdirs.keys():
		outdir = outdirs[k]
		stats = load_experiment(outdir)

		experiment_name = outdir.split("/")[-1].split("_")
		# remove date
		date = experiment_name[-1]
		Dates = Dates + "_" + date
		experiment_name.pop(-1)
		experiment_name = "_".join(experiment_name)
		experiment = load_dict(outdir, experiment_name)
		experiment = experiment[experiment_name]
		experiments = experiment[('n_vec', 'N_vec', 'alpha_vec')]

		x_id = 1 # N_vec
		xlabel = r"$N$"
		base = 2
		lsqs_base = "N"
		alpha = experiment[('n_vec', 'N_vec', 'alpha_vec')][0][2]
		n = experiment[('n_vec', 'N_vec', 'alpha_vec')][0][0]
		empty_label = r"($\alpha={}$)".format(alpha)
		set_ylim = False
		ndelete = 0

		marker = next(markers)
		line_style = next(line_styles)

		replications = sorted(stats.keys())

		errors = {}

		# Find all replications for each experiment
		for e in experiment[('n_vec', 'N_vec', 'alpha_vec')]:
			errors[e] = []

			for r in replications:
				s = stats[r][e]
				errors[e].append(s)


		# Compute statistics
		errors_stats = compute_random_errors(errors)

		# Find "x" values
		x_vec = []
		for e in errors.keys():
			n, N, alpha = e
			x_vec.append(e[x_id])


		# Compute convergence rates
		y_vec = [errors_stats[e]["mean"] for e in experiments]

		X_vec[k] = x_vec
		Y_vec[k] = y_vec

		ax.plot([], [], " ", label=empty_label)

		# Plot mean of realizations
		ax.scatter(x_vec, y_vec, marker=marker, color="black",
			label=r"$\widehat{\mathrm{E}}[\widetilde \chi_n(\bar{u}_{N,\alpha,n})]$, " + r"$n={}$".format(n))

		# Legend and labels
		ax.set_xlabel(xlabel)
		ax.set_xscale("log", base=base)
		ax.set_yscale("log", base=base)



	# least squares fit
	x_vec = np.array(X_vec[1])
	y_vec = np.mean(list(Y_vec.values()), axis=0)
	ndrop = 0

	X = np.ones((len(x_vec[ndrop::]), 2)); X[:, 1] = np.log(x_vec[ndrop::]) # design matrix
	x, residudals, rank, s = np.linalg.lstsq(X, np.log(y_vec[ndrop::]), rcond=None)

	rate = x[1]
	constant = np.exp(x[0])

	X = x_vec[ndrop::]
	Y = constant*X**rate
	ax.plot(X, Y, color="black", linestyle="--", label=lsqs_label(rate=rate, constant=constant, base=lsqs_base))

	## Legend with unique entries
	_handles, _labels = plt.gca().get_legend_handles_labels()
	by_label = dict(zip(_labels, _handles))
	plt.legend(by_label.values(), by_label.keys(), ncol=ncol)

	# Create outdir
	outdir = "output/Experiments/Monte_Carlo_Rates" + Dates

	if not os.path.exists(outdir):
		os.makedirs(outdir)

	plt.tight_layout()
	plt.savefig(outdir + "/" + "Monte_Carlo_Rates" + Dates + ".pdf")
	plt.close()


if __name__ == "__main__":

	import sys

	outdirs = {}
	for k in range(1, 5):

		try:
			outdirs[k] = sys.argv[k]
		except:
			break


	plot_dimension_experiment(outdirs)
