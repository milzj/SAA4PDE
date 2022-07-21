import numpy as np

class Experiments(object):

	def __init__(self):

		self._experiments = {}

		name = "Monte_Carlo_Rate"
		N_vec = [2**i for i in range(3, 10+1)]

		n_vec = 64*np.ones(len(N_vec), dtype=np.int64)
		alpha_vec = 1e-3*np.ones(len(N_vec), dtype=np.float64)

		self.add_experiment(name, n_vec, N_vec, alpha_vec)


		name = "Regularization_Parameter"
		num=6
		alpha_vec = np.linspace(-num, 1, num+2)
		alpha_vec = 10.0**alpha_vec

		n_vec = 64*np.ones(len(alpha_vec), dtype=np.int64)
		N_vec = 256*np.ones(len(alpha_vec), dtype=np.int64)

		self.add_experiment(name, n_vec, N_vec, alpha_vec)

		name = "Dimension_Dependence"

		n_vec = [2**i for i in range(3, 7+1)]
		alpha_vec = 1e-3*np.ones(len(n_vec), dtype=np.float64)
		N_vec = 256*np.ones(len(n_vec), dtype=np.int64)

		self.add_experiment(name, n_vec, N_vec, alpha_vec)

		name = "Dimension_Dependence_large_alpha"

		n_vec = [2**i for i in range(3, 7+1)]
		alpha_vec = 1e-1*np.ones(len(n_vec), dtype=np.float64)
		N_vec = 256*np.ones(len(n_vec), dtype=np.int64)

		self.add_experiment(name, n_vec, N_vec, alpha_vec)

	def add_experiment(self, name, n_vec, N_vec, alpha_vec):

		key = ("n_vec", "N_vec", "alpha_vec")
		items = list(zip(np.array(n_vec),np.array(N_vec),np.array(alpha_vec)))

		self._experiments[name] = {key: items}
		self._experiments[name + "_Synthetic"] = {key: items}


	def __call__(self, experiment_name):

		return self._experiments[experiment_name]


if __name__ == "__main__":

	experiments = Experiments()
	print(experiments._experiments)

	experiment_name = "Dimension_Dependence"

	for e in experiments(experiment_name)[("n_vec", "N_vec", "alpha_vec")]:
		print(e)
		n, N, alpha = e
