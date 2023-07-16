try:
	from .compute_fem_errors import compute_fem_errors
except ImportError:
	print("ImportError compute_fem_errors.")

from .compute_fem_rates import compute_fem_rates
from .save_dict import save_dict
from .load_dict import load_dict
from .luxemburg_norm import LuxemburgNorm
from .compute_random_errors import compute_random_errors

