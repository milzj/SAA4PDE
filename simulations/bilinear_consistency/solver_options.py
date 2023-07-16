class SolverOptions(object):

	def __init__(self):

		self._options={'gtol': 1e-10,
	        	'maxiter': 20,
	                'display': 3,
			"restrict": True,
                        'ncg_hesstol': 0,
			 'line_search': 'fixed',
			"correction_step": False}


	@property
	def options(self):
		return self._options
