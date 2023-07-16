from .convergence_rates import convergence_rates
import numpy as np
import warnings

def semismooth_taylor_test_numpy(F, x, d, DFd=None, norm=None):
	"""Performs a Taylor test for a semismooth mapping F

	Parameters:
	-----------
	F : Function
	x : ndarry
		input vector
	d : ndarray
		direction
	DFd : Function (optional)
		DFd(y,h) evaluates generalized derivative
		at y in direction h
	norm: Function (optional)
		norm (default np.linalg.norm)

	Note:
	-----

	If F is Lipschitz continuous at x, then
	F(x+d) - F(x) = O(norm(d)).

	If F is semismooth at x,
	then F(x+d) - F(x) - DF(x+d)d = o(norm(d)).

	If F is semismooth at x of order alpha, then
	then F(x+d) - F(x) - DF(x+d)d = O(norm(d)^(1+alpha)).
	"""

	tol = 1e-12
	residuals = []
	epsilons = [0.01 / 2**i for i in range(5)]

	if norm == None:
		norm = lambda x: np.linalg.norm(x)

	Fx = F(x)

	for eps in epsilons:

		y = x + eps*d
		Fy = F(y)

		if DFd is not None:
			DFyd = DFd(y,d)
		else:
			DFyd = 0.0

		res = norm(Fy-Fx-eps*DFyd)
		residuals.append(res)



	if np.median(residuals) < tol and DFd is not None:
		warnings.warn("Mapping might be affine-linear at x.")
		r = 2
	else:
		r = np.median(convergence_rates(residuals, epsilons))

	return r
