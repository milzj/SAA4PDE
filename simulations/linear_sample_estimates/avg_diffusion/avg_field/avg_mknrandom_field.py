from dolfin import *
from dolfin_adjoint import *
import matplotlib.pyplot as plt
import numpy as np

def avg_mknrandom_field(function_space):
    """Average of MKN random field.

    See mknrandom_field.py

    """

    avg_kappa = Expression("sinh(cos(1.1*pi*x[0]))/cos(1.1*pi*x[0])"+
                            "*sinh(cos(1.2*pi*x[0]))/cos(1.2*pi*x[0])"+
                            "*sinh(sin(1.3*pi*x[1]))/sin(1.3*pi*x[1])"+
                            "*sinh(sin(1.4*pi*x[1]))/sin(1.4*pi*x[1])", degree = 1)

    v = Function(function_space)
    v.interpolate(avg_kappa)

    # use continuity
    w = v.vector()[:]
    idx = np.isnan(w)
    w[idx] = 1.0
    v.vector()[:] = w[:]

    return v



