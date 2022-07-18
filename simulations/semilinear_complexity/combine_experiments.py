from plot_experiment import load_experiment
from stats import save_dict, compute_fem_errors, load_dict
import numpy as np
import sys, os

# We combine the simulation output of two experiments

outdir_one = "output/Experiments/Monte_Carlo_Rate_08-Jul-2022-10-31-52"
outdir_two = "output/Experiments/Monte_Carlo_Rate_10-Jul-2022-19-02-23"

filename_one = outdir_one.split("_")
filename_one = filename_one[-1]

filename_two = outdir_two.split("_")
filename_two = filename_two[-1]


filename = "Monte_Carlo_Rate_08-Jul-2022-10-31-52-10-Jul-2022-19-02-23"
outdir = "output/Experiments" + "/" + filename

if not os.path.exists(outdir):
	os.makedirs(outdir)


for rank in range(48):

	_filename_one = filename_one + "_mpi_rank=" + str(rank)
	_filename_two = filename_two + "_mpi_rank=" + str(rank)
	_filename = filename_one + "-" + filename_two + "_mpi_rank=" + str(rank)

	_stats_one = load_dict(outdir_one, _filename_one)
	_stats_two = load_dict(outdir_two, _filename_two)
	print(_stats_one)
	print(_stats_two)
	print(rank)

	i = rank + 1
	_stats_one[i][(64, 2048, 0.001)] = _stats_two[i][(64, 2048, 0.001)]

	save_dict(outdir, _filename, _stats_one)


np.savetxt(outdir  + "/" + filename  + "_filename.txt", np.array([outdir]), fmt = "%s")

outdir_one = "output/Experiments/Dimension_Dependence_small_alpha_14-Jul-2022-16-53-38"
outdir_two = "output/Experiments/Dimension_Dependence_small_alpha_14-Jul-2022-12-47-13"

filename_one = outdir_one.split("_")
filename_one = filename_one[-1]

filename_two = outdir_two.split("_")
filename_two = filename_two[-1]


filename = "Dimension_Dependence_small_alpha_14-Jul-2022-16-53-38-14-Jul-2022-12-47-13"
outdir = "output/Experiments" + "/" + filename

if not os.path.exists(outdir):
	os.makedirs(outdir)


for rank in range(48):

	_filename_one = filename_one + "_mpi_rank=" + str(rank)
	_filename_two = filename_two + "_mpi_rank=" + str(rank)
	_filename = filename_one + "-" + filename_two + "_mpi_rank=" + str(rank)

	_stats_one = load_dict(outdir_one, _filename_one)
	_stats_two = load_dict(outdir_two, _filename_two)
	print(_stats_one)
	print(_stats_two)
	print(rank)

	i = rank + 1
	_stats_one[i][(128, 256, 1e-05)] = _stats_two[i][(128, 256, 1e-05)]

	save_dict(outdir, _filename, _stats_one)


np.savetxt(outdir  + "/" + filename  + "_filename.txt", np.array([outdir]), fmt = "%s")
