class SolverOptions(object):

	def __init__(self):

		self._options={'gtol': 1e-8,
	        	'maxiter': 20,
	                'display': 3,
                        'ncg_hesstol': 0,
			 'line_search': 'fixed',
			"correction_step": True,
            "restrict": True}


	@property
	def options(self):
		return self._options
