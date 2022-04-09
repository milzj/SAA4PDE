from pathlib import Path
import sys

here = Path(__file__).parent
sys.path.insert(0, str(here.parent) + "/problem")
sys.path.insert(0, str(here.parent) + "/prox")
sys.path.insert(0, str(here.parent) + "/algorithms")
sys.path.insert(0, str(here.parent) + "/base")
