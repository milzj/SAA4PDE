from .prox_l1 import prox_l1
from .proj_box import proj_box


def diag_grad_prox_box_l1(v, lb, ub, lam):
	"""Compute proximal operator for box constaints and l1-norm and 
	diagonal entries of its generalized derivative.

	Computes proximal operator and an element of the subdifferential.
	The proximal operator is computed using a composition formula.

	TODO: Should we use (abs(v) >= lam) instead of (abs(v) > lam)?

	Parameters:
	-----------
		v : ndarray
			input array
		lb, ub : ndarray or float
			lower and upper bound
		lam : float
			parameter

	Returns:
	--------
		(proximal_operator, diagonal of subgradient) : (ndarray, ndarray)
			proximal operator and diagonal entries of one of its generalized subgradients.

	References:
	-----------

	Examples 3.2.8, 3.2.9 and 4.2.17 in

	A. Milzarek, Numerical methods and second order theory for nonsmooth problems,
	Dissertation, TUM, Munich, http://mediatum.ub.tum.de/?id=1289514
	"""

	w = prox_l1(v, lam)

	delta = 1.0*(w > lb) * (w < ub) * (abs(v) >= lam)

	return proj_box(w, lb, ub), delta
