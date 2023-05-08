# nominal solution
python plot_control.py Nominal_Simulation_n=64_date=01-Jul-2022-13-26-50/01-Jul-2022-13-26-50_nominal_solution_n=64

# reference solution
python plot_control.py Reference_Simulation_n=64_N=8192_date=04-Jul-2022-22-09-44/04-Jul-2022-22-09-44_reference_solution_mpi_rank=0_N=8192_n=64

# Monte Carlo Rate
python plot_experiment.py output/Experiments/Monte_Carlo_Rate_08-Jul-2022-10-31-52-10-Jul-2022-19-02-23 4 -1

# Regularization Parameter
python plot_experiment.py output/Experiments/Regularization_Parameter_09-Jul-2022-11-27-21 4 -1
python plot_experiment.py output/Experiments/Tikhonov_n=64_N=256_03-May-2023-18-59-33 4 0
python plot_experiment.py output/Experiments/Tikhonov_n=64_N=256_03-May-2023-18-59-33 -1 1
python plot_experiment.py output/Experiments/Tikhonov_n=64_N=256_03-May-2023-18-59-33 -1 2

# Dimension Dependence
python plot_experiment.py output/Experiments/Dimension_Dependence_08-Jul-2022-10-39-02

# Dimension Dependence large alpha
python plot_experiment.py output/Experiments/Dimension_Dependence_large_alpha_19-Jul-2022-21-23-24-19-Jul-2022-21-16-20


# alpha = 1e-3

expone="output/Experiments/Monte_Carlo_Rate_n_16_23-Jul-2022-10-25-13"
exptwo="output/Experiments/Monte_Carlo_Rate_n_32_23-Jul-2022-13-14-35"
expthree="output/Experiments/Monte_Carlo_Rate_08-Jul-2022-10-31-52"

python plot_dimension_experiment.py $expone $exptwo $expthree

expone="output/Experiments/Monte_Carlo_Rate_n_16_alpha_01_25-Jul-2022-12-09-27"
exptwo="output/Experiments/Monte_Carlo_Rate_n_32_alpha_01_25-Jul-2022-14-02-26"
expthree="output/Experiments/Monte_Carlo_Rate_n_64_alpha_01_26-Jul-2022-13-34-56"
# alpha = 1e-1
python plot_dimension_experiment.py $expone $exptwo $expthree
