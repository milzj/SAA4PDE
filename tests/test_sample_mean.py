import numpy as np
import random
import pytest

def mean_onepass12(Z):
	"""Computes the sample mean using a one-pass algorithm.

	Computes the sample mean using formula M_12 in Ling (1974).

	References:
	-----------

	R. F. Ling, Comparison of Several Algorithms for Computing Sample Means and Variances,
	Journal of the American Statistical Association
	Vol. 69, No. 348 (Dec., 1974), pp. 859-866 (8 pages)
	https://doi.org/10.2307/2286154

	"""
	N = len(Z)
	return Z[1] + np.sum(Z-Z[1])/N


def mean_onepass14(Z):
	"""Computes the sample mean using a one-pass algorithm.

	Computes the sample mean using formula M_14 in Ling (1974).

	References:
	-----------

	D. Assencio, Numerically stable computation of arithmetic means, 2015, blog post,
	https://diego.assencio.com/?index=c34d06f4f4de2375658ed41f70177d59

	R. F. Ling, Comparison of Several Algorithms for Computing Sample Means and Variances,
	Journal of the American Statistical Association
	Vol. 69, No. 348 (Dec., 1974), pp. 859-866 (8 pages)
	https://doi.org/10.2307/2286154

	"""
	N = len(Z)
	M = 0.0
	for i in range(N):
		M += 1.0/(i+1.0)*(Z[i]-M)
	return M

def relative_error(a,b):
	return abs(a-b)/max(1.0, abs(b))

def absolute_error(a,b):
	return abs(a-b)

def perturb(Z):
	"Data perturbation based on that in Ling (1974)."
	N = len(Z)
	for i in range(N):
		Z[i] += 100.0*round(10.0*(i-1)/1000.0)
	random.shuffle(Z)
	return Z


testdata = [
	perturb(1000.0+np.random.randn(1000)),
	perturb(np.random.rand(1000)),
	perturb(np.exp(5.0*np.random.randn(1000)))
]

@pytest.mark.parametrize("Z", testdata)
def test_sample_mean(Z):

	N = len(Z)
	mean = Z.mean()
	rtol = 1e-6
	atol = 1e-2

	m = 0.0
	for i in range(N):
		m += (1/N)*Z[i]

	assert relative_error(m,mean) <= rtol
	assert absolute_error(m,mean) <= atol

	M12 = mean_onepass12(Z)

	assert relative_error(M12,mean) <= rtol
	assert absolute_error(M12,mean) <= atol

	M14 = mean_onepass14(Z)

	assert relative_error(M14,mean) <= rtol
	assert absolute_error(M14,mean) <= atol

	Mean = Z.sum()/N
	assert relative_error(Mean,mean) <= rtol
	assert absolute_error(Mean,mean) <= atol

	# Is M12 more accurate than M14?
	assert relative_error(M12,mean) <= max(1.0,relative_error(M14,mean))

	# Is M14 more accurate then m?
	assert relative_error(M14,mean) <= max(1.0,relative_error(m,mean))
	assert absolute_error(M14,mean) <= absolute_error(m,mean) + atol

