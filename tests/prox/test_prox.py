import pytest

from prox import diag_grad_prox_box_l1, grad_prox_box_l1
import numpy as np
from base import semismooth_taylor_test_numpy

def fun1(x):
	return np.sin(x)
def grad1(x):
	return np.cos(x)
def fun2(x):
	return x**2
def grad2(x):
	return 2.0*x

@pytest.mark.parametrize("grad_prox", [diag_grad_prox_box_l1,grad_prox_box_l1])
@pytest.mark.parametrize("lam", [1e-6, 1e-2, 1.0])
@pytest.mark.parametrize("lb", [-10.0, -5.0, -1.0])
@pytest.mark.parametrize("fungrad", [[fun1,grad1], [fun2,grad2]])
@pytest.mark.filterwarnings("ignore:Mapping")
def test_prox(grad_prox, lam, lb, fungrad):


	atol = 0.2

	ub = -lb

	n = 10

	np.random.randn(1234)
	x = np.random.randn(n)
	np.random.randn(12345)
	d = np.random.randn(n)

	def F(x):
		return diag_grad_prox_box_l1(x, lb, ub, lam)[0]

	def DFd(x, d):
		return np.multiply(diag_grad_prox_box_l1(x, lb, ub, lam)[1], d)

	# Check Lipschitz continuity
	assert np.isclose(semismooth_taylor_test_numpy(F, x, d), 1.0, atol=atol)
	# Check semismoothness
	assert np.isclose(semismooth_taylor_test_numpy(F, x, d, DFd=DFd), 2.0, atol=atol)

	fun = fungrad[0]
	grad = fungrad[1]

	def F(x):
		return fun(diag_grad_prox_box_l1(x, lb, ub, lam)[0])

	def DFd(x, d):
		return grad(diag_grad_prox_box_l1(x, lb, ub, lam)[0])* \
			np.multiply(diag_grad_prox_box_l1(x, lb, ub, lam)[1], d)


	# Check Lipschitz continuity
	assert np.isclose(semismooth_taylor_test_numpy(F, x, d), 1.0, atol=atol)
	# Check semismoothness
	assert np.isclose(semismooth_taylor_test_numpy(F, x, d, DFd=DFd), 2.0, atol=atol)
