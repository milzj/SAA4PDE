from semilinear_random import semilinear_random_problem
import numpy as np


#N = 10
#n = 32
#semilinear_random_problem(N, n)


N = 1000
n = int(np.sqrt(9800/2))
semilinear_random_problem(N, n)

