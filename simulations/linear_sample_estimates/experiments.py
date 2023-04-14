import numpy as np

class Experiments(object):

	def __init__(self):

		self._experiments = {}

		name = "n_N_variable"
		n_vec = [8, 12, 16, 24, 36, 48, 72]
		N_vec = [n**2 for n in n_vec]

		self.add_experiment(name, n_vec, N_vec)

		name = "N_fixed"
		n_vec = [8, 12, 16, 24, 36, 48, 72]
		N_vec = 16**2*np.ones(len(n_vec), dtype=np.int64)

		self.add_experiment(name, n_vec, N_vec)

		name = "n_fixed"
		vec = [8, 12, 16, 24, 36, 48, 72]
		N_vec = [n**2 for n in vec]
		n_vec = 72*np.ones(len(n_vec), dtype=np.int64)

		self.add_experiment(name, n_vec, N_vec)

		name = "combinations"
		_n_vec = [8, 12, 16, 24, 36, 48, 72]
		_N_vec = [n**2 for n in _n_vec]

		n_vec = []
		N_vec = []

		for n in _n_vec:
			for N in _N_vec:
				n_vec.append(n)
				N_vec.append(N)

		self.add_experiment(name, n_vec, N_vec)


		name = "test"
		_n_vec = [8, 12]
		_N_vec = [n**2 for n in _n_vec]

		n_vec = []
		N_vec = []

		for n in _n_vec:
			for N in _N_vec:
				n_vec.append(n)
				N_vec.append(N)

		self.add_experiment(name, n_vec, N_vec)

	def add_experiment(self, name, n_vec, N_vec):

		key = ("n_vec", "N_vec")
		items = list(zip(np.array(n_vec),np.array(N_vec)))

		self._experiments[name] = {key: items}

	def __call__(self, experiment_name):

		return self._experiments[experiment_name]


if __name__ == "__main__":

	experiments = Experiments()

	for experiment_name in ["n_N_variable", "N_fixed", "n_fixed", "combinations"]:
		experiment = experiments(experiment_name)

		E = {}
		_n = -1
		for e in experiment[("n_vec", "N_vec")]:
			n, N = e
			if n != _n:
				E[n] = {}
			E[n][N] = -1
			_n = n

		print(E)
