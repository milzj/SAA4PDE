import pytest
import numpy as np

def pytest_runtest_setup(item):
	np.random.seed(1234)
