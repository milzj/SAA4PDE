import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.colors import LinearSegmentedColormap

from dolfin import *

try:
	from .boxfield import *
except ImportError as error:
	print("boxfield.py is not located in stats/base.")
	print("boxfield.py is available at https://github.com/hplgit/fenics-tutorial/blob/master/src/vol1/python/boxfield.py")
	raise error


def mesh2triang(mesh):
	"""Create a triangulation given the input mesh.

	References:
	---------
	Chris Richardson: https://fenicsproject.org/qa/5795/plotting-dolfin-solutions-in-matplotlib/
	J. B. Haga and F. Valdmanis: https://fenicsproject.org/olddocs/dolfinx/dev/python/_modules/dolfin/plotting.html
	"""

	xy = mesh.coordinates()
	return tri.Triangulation(xy[:, 0], xy[:, 1], mesh.cells())

def cmap_blue_orange():
	"Color map inspired by cm.coolwarm."


	return LinearSegmentedColormap.from_list(name="cmap_BlueOrange",
                                          colors =["tab:blue", "lightgrey", "tab:orange"],
                                            N=256)

def surface_function(obj, n, division=None, uniform_mesh=None, figsize=(5.0, 5.0), cmap=cmap_blue_orange()):
	"""Create surface plot of a fenics function defined on a structured mesh using boxfield.

	Parameters:
	----------
		obj : fenics function defined on UnitIntervalMesh or UnitSquareMesh
		n : int
			number of cells in each direction
		cmap : color map (matplotlib.colors.LinearSegmentedColormap)
			default is cmap_BlueOrange
		figsize : tuple
			default is (5, 5)

	References:
	----------

	H. P. Langtangen, A. Logg: Solving PDEs in Python: The FEniCS Tutorial I, Springer, Cham, 2016

	"""

	if type(n) == type(1):
		division = (n,n)

	u_box = FEniCSBoxField(obj, division=division, uniform_mesh=uniform_mesh)
	u_ = u_box.values

	cv = u_box.grid.coorv
	X = cv[0]
	Y = cv[1]
	Z = u_

	fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=figsize)


	mesh = obj.function_space().mesh()

	tri = mesh2triang(mesh)
	ax.plot_trisurf(X.flatten(), Y.flatten(), Z.flatten(), triangles=tri.triangles,\
		 	shade=False, antialiased=False, cmap = cmap)

	return fig, ax

