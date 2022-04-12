from stats import LuxemburgNorm

import pytest
import numpy as np


@pytest.mark.parametrize("sigma", [0.1, 1., 5.])
@pytest.mark.parametrize("seed", range(10))
def test_luxemburg_norm_gaussian(sigma, seed):
	"""Compare empirical estimate of Luxemburg norm with exact norm.

	For s=3/4, we have 1/sqrt(1-s) = 2
	(see https://www.wolframalpha.com/input?i=1%2Fsqrt%281-s%29+%3D+2).

	"""

	N = 10000
	np.random.seed(seed)
	Z = sigma*np.random.randn(N)
	rtol = 0.05
	s = 3.0/4.

	luxnorm = LuxemburgNorm()

	exact_luxnorm = 1.0/np.sqrt(.5*s/sigma**2)
	approx_luxnorm = luxnorm.evaluate(Z)

	rerror = abs(exact_luxnorm-approx_luxnorm)/exact_luxnorm
	assert rerror <= rtol


def test_luxemburg_norm_rademacher():

	N = 10
	rtol = 1e-14

	Z = np.random.uniform(size=N)
	Z[Z >=0.5] = 1.
	Z[Z < 0.5] = -1.

	luxnorm = LuxemburgNorm()

	exact_luxnorm = np.sqrt(1.0/np.log(2.0))
	approx_luxnorm = luxnorm.evaluate(Z)

	assert abs(exact_luxnorm-approx_luxnorm)/exact_luxnorm <= rtol


def test_luxemburg_norm_one_sample():

	Z = np.ones(1)
	rtol = 1e-14

	luxnorm = LuxemburgNorm()

	exact_luxnorm = np.sqrt(1.0/np.log(2.0))
	approx_luxnorm = luxnorm.evaluate(Z)

	assert abs(exact_luxnorm-approx_luxnorm)/exact_luxnorm <= rtol


@pytest.mark.parametrize("seed", range(10))
def test_luxemburg_norm_uniform(seed):
	"""
	We have int exp(s x^2) dx = sqrt(pi) erfi(sqrt(s))/2/sqrt(s)
	(see https://www.wolframalpha.com/input?i=integrate+exp%28sx%5E2%29+for+x%3D0..1)

	For s = 1.67482492851261717921985615,
	we have int exp(s x^2) dx = 2
	(see https://www.wolframalpha.com/input?i=%28Sqrt%5BPi%5D+Erfi%5BSqrt%5Bs%5D%5D%29%2F%282+Sqrt%5Bs%5D%29+%3D+2)

	"""
	rtol = 1e-2
	N = 10000

	np.random.seed(seed)
	Z = np.random.rand(N)

	s = 1.67482492851261717921985615
	exact_luxnorm = 1.0/np.sqrt(s)

	luxnorm = LuxemburgNorm()

	approx_luxnorm = luxnorm.evaluate(Z)

	rerror = abs(exact_luxnorm-approx_luxnorm)/exact_luxnorm
	assert rerror <= rtol
