from pathlib import Path
import sys


import pytest
import numpy as np

here = Path(__file__).parent
sys.path.insert(0, str(here.parent) + "/problem")
sys.path.insert(0, str(here.parent) + "/prox")
sys.path.insert(0, str(here.parent) + "/algorithms")
sys.path.insert(0, str(here.parent) + "/base")


def pytest_runtest_setup(item):
	np.random.seed(1234)
