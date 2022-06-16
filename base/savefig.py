import matplotlib.pyplot as plt


def savefig(figure_name, formats=["pdf", "png"]):
	"""Save matplotlib plots using multiple formats.

	The function closes the current figure window.

	Parameters:
	----------
		figure_name : str
			Figure's name
		formats : list of strings (optional)
			default is ["pdf", "png"]

	"""

	for f in formats:
		plt.tight_layout()
		plt.savefig(figure_name + ".{}".format(f))
	plt.close()
