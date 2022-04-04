"""
This file is a modification of

https://github.com/funsim/moola/blob/master/moola/algorithms/newton_cg.py

funsim/moola is licensed under the GNU Lesser General Public License v3.0

Changes made to https://github.com/funsim/moola/blob/master/moola/algorithms/newton_cg.py
(commit: https://github.com/funsim/moola/commit/79ffc8d86dff383762063970358b8aebfe373896)


	- __name__ was changed to SemismoothNewtonCG

	options:
	-------
	- added "correction_step", "restrict", "globalization"

	main loop and step computation:
	-------------------------------
	- gradient computation was replaced by normal_map.primal() computation
	- Hessian computation was replaced by derivative_normal_map computation

	main loop:
	----------
	- Removed line search options. Use globalization instead
	- We implemented the CG method with final step, Algorithm 1 (CG with final step), developed by Pieper (2015).
	- The optimization variable z was added. z is the actural iterate. Upon calling
	normal_map, the prox of z is stored in x.
	- The method's default cg iterations is set to 20.
	- Deleted ll. 122 -- 141
	- Added restriction and correction steps

		globalization:
		-------------
		- We use a simple globalization technique adapted from a monotonicity test described in Deufhard (2011).
		- The default value for sigma_min is taken from Mannel and Rund (2020).
		- A convergence analysis of a globalized semismooth Newton method is provided in
		Gerdts et al. (2017).

	header:
	-------
	- sqrt of numpy is used
	- moola tools are loaded.

	References:
	-----------

	Details about the final step (also called correction step) are provided in
	[Pieper2015, sects. 3.2.2, 3.2.3, 3.4.1, and 3.5.2]. The final step as introduced in [Pieper2015, p. 61]
	is called correction step in [Mannel2020, sect. 5].


	References:
	-----------
	S. Funke: moola, https://github.com/funsim/moola

	Sebastian K. Mitusch, Simon W. Funke, and Jørgen S. Dokken (2019). dolfin-adjoint 2018.1:
	automated adjoints for FEniCS and Firedrake, Journal of Open Source Software, 4(38), 1292,
	doi:10.21105/joss.01292.

	K. Pieper, Finite element discretization and efficient numerical solution of el-
	liptic and parabolic sparse control problems, Dissertation, TUM, Munich 2015,
	http://mediatum.ub.tum.de/?id=1241413

	F. Mannel and A. Rund, A hybrid semismooth quasi-Newton method for non-
	smooth optimal control with PDEs, Optim. Eng., (2020),
	https://doi.org/10.1007/s11081-020-09523-w.

	M. Gerdts, S. Horn, and S.-J. Kimmerle, Line search globalization of a semismooth
	Newton method for operator equations in Hilbert spaces with applications in optimal
	control, J. Ind. Manag. Optim., 13 (2017), pp. 47–62, https://doi.org/10.3934/
	jimo.2016003.

	P. Deuflhard, Newton Methods for Nonlinear Problems, Springer Ser. Comput.
	Math. 35, Springer, Berlin, 2011, https://doi.org/10.1007/978-3-642-23899-4
"""


from moola.algorithms.optimisation_algorithm import *
from moola.algorithms.bfgs import LinearOperator, dual_to_primal

from numpy import sqrt
import warnings

class SemismoothNewtonCG(OptimisationAlgorithm):
	'''
	Initialises the Hybrid CG method. The valid options are:
		* options: A dictionary containing additional options for the steepest descent algorithm. Valid options are:
		- tol: Not supported yet - must be None.
		- maxiter: Maximum number of iterations before the algorithm terminates. Default: 200.
		- disp: dis/enable outputs to screen during the optimisation. Default: True
		- gtol: Gradient norm stopping tolerance: ||grad j|| < gtol.
		- line_search: defines the line search algorithm to use. Default: strong_wolfe
		- line_search_options: additional options for the line search algorithm. The specific options read the help
		for the line search algorithm.
		- an optional callback method which is called after every optimisation iteration.
	'''

	__name__ = 'SemismoothNewtonCG'
	def __init__(self, problem, initial_point = None, precond=LinearOperator(dual_to_primal), options = {}):
		'''
		Initialises the Hybrid CG method. The valid options are:
		 * options: A dictionary containing additional options for the steepest descent algorithm. Valid options are:
		    - tol: Not supported yet - must be None.
		    - maxiter: Maximum number of iterations before the algorithm terminates. Default: 200.
		    - disp: dis/enable outputs to screen during the optimisation. Default: True
		    - gtol: Gradient norm stopping tolerance: ||grad j|| < gtol.
		    - line_search: defines the line search algorithm to use. Default: strong_wolfe
		    - line_search_options: additional options for the line search algorithm. The specific options read the help
		      for the line search algorithm.
		    - an optional callback method which is called after every optimisation iteration.
		  '''

		# Set the default options values
		self.problem = problem
		self.set_options(options)
		self.linesearch = get_line_search_method(self.options['line_search'], self.options['line_search_options'])
		self.data = {'control'   : initial_point,
		             'iteration' : 0,
		             'precond'   : precond }

	def __str__(self):
		s = "Semismooth Newton CG method.\n"
		s += "-"*30 + "\n"
		s += "Line search:\t\t %s\n" % self.options['line_search']
		s += "Maximum iterations:\t %i\n" % self.options['maxiter']
		return s

	# set default parameters
	@classmethod
	def default_options(cls):
		# this is defined as a function to prevent defaults from being changed at runtime.
		default = OptimisationAlgorithm.default_options()
		default.update(
		# generic parameters:
		{"jtol"                   : None,
		"gtol"                   : 1e-4,
		"maxiter"                :  200,
		"display"                :    3,
		"line_search"            : "fixed",
		"line_search_options"    : {"start_stp": 1},
		"callback"               : None,
		"record"                 : ("grad_norm", "objective"),

		# method specific parameters:
		"ncg_reltol"             :  .5,
		"ncg_maxiter"            : 20,
		"ncg_hesstol"            : "default",
		"correction_step"	 : False,
		"restrict"		 : False,
		"globalization"		 : "monotonicity_test"
		})
		return default

	def solve(self):
		'''
		    Arguments:
		     * problem: The optimisation problem.

		    Return value:
		      * solution: The solution to the optimisation problem
		 '''
		self.display( self.__str__(), 1)

		objective = self.problem.obj
		options = self.options

		B = self.data['precond']
		x = self.data['control']
		i = self.data['iteration']

		# Compute initial value z as proposed by Mannel and Rund (2020)
		J = objective(x)
		r = objective.derivative(x)
#		z = r.primal() # B*r
#		z.scale(-1.0/objective.alpha)
		z = x.copy()

		# compute initial normal map
		r = objective.normal_map(x, z)  # initial residual ( with dk = 0)
		r.scale(-1.0)

		self.update({'objective' : J,
		             'grad_norm' : r.primal_norm()})

		self.record_progress()

		if options['ncg_hesstol'] == "default":
			import numpy
			eps = numpy.finfo(numpy.float64).eps
			ncg_hesstol = eps*numpy.sqrt(len(z))
		else:
			ncg_hesstol = options['ncg_hesstol']

		# Start the optimisation loop
		while self.check_convergence() == 0:
			self.display(self.iter_status, 2)
			p = Br = (B * r) # mapping residual to primal space
			# TODO: Why are p and r not defined prior to loop?
			d = p.copy().zero()

			TBr = Br.copy()
			Tp = p.copy()
			if self.options["restrict"] == True:
				objective.restrict(TBr) # compute T*Br

			rBr = r.apply(TBr) # T*Br

			if rBr == 0: # Temporary fix?
				rBr = r.apply(Br)

			H = objective.derivative_normal_map(x, z) # in L(X, X*)

			# CG iterations
			cg_tol =  min(options['ncg_reltol']**2, sqrt(rBr))*rBr
			cg_iter  = 0
			cg_break = 0
			while cg_iter < options['ncg_maxiter'] and rBr >= cg_tol:
				print('TEST:', rBr, cg_tol)

				Hp  = H(p)
				if self.options["restrict"] == True:
					objective.restrict(Tp) # compute T*p
				pHp = Hp.apply(Tp)

				if pHp == 0: # Temporary fix?
					pHp = Hp.apply(p)

				if pHp <= 0:
					warnings.warn("Hessian not positive definite pHp={}".format(pHp))


				self.display('cg_iter = {}\tcurve = {}\thesstol = {}'.format(cg_iter, pHp, ncg_hesstol), 3)

				# Standard CG iterations
				alpha = rBr / pHp
				d.axpy(alpha, p)            # update cg iterate
				r.axpy(-alpha, Hp)          # update residual

				Br = B*r
				TBr.assign(Br)
				if self.options["restrict"] == True:
					objective.restrict(TBr) # compute T*Br
				t = r.apply(TBr)
				rBr, beta = t, t / rBr,

				p.scale(beta)
				p.axpy(1., Br)
				Tp.assign(p)

				cg_iter +=1

			# final/correction step
			norm_correction_step = 1.0/objective.alpha*Br.norm()
			if options["correction_step"] == True:
				print("Norm of correction step={}".format(norm_correction_step))
				d.axpy(1.0/objective.alpha, Br)

			# Globalize using monotonicity test
			if self.options["globalization"] == "monotonicity_test":
				rnew = objective.normal_map(x, z+d)
				r = objective.normal_map(x,z)
				sigma_min = .5**10
				sigma = 1.0
				while rnew.primal_norm() > (1.0-sigma/100.0+2e-4)*r.primal_norm() and sigma >= sigma_min:
					sigma /= 2.0
					d.scale(.5)
					rnew = objective.normal_map(x, z+d)
				if sigma <= sigma_min:
					warnings.warn("Step size sigma = {} is smaller than minimum step size = {}.".format(sigma, sigma_min))

			# Update iterate z, x and residual
			z.axpy(1.0, d)
			r = objective.normal_map(x, z)

			# evaluate normal map at the new point and compute prox
			r.scale(-1)
			J, oldJ = objective(x), J

			i += 1

			if options['callback'] is not None:
				options['callback'](J, r)

			# store current iteration variables
			self.update({'iteration' : i,
			         'control'   : x,
			         'grad_norm' : r.primal_norm(), # norm of normal mapping
			         'delta_J'   : oldJ-J,
			         'objective' : J,
			         'lbfgs'     : B,
			 	 "norm_correction_step": norm_correction_step,
				 "Dprox_norm": objective.T.norm()})
			self.record_progress()

		self.display(self.convergence_status, 1)
		self.display(self.iter_status, 1)
		return self.data

