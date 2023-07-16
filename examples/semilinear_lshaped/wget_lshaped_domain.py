# The script downloads an L-shaped mesh
from urllib import request

remote_url = "https://fenicsproject.org/pub/data/meshes/lshape.xml.gz"
local_file = "lshape.xml.gz"

request.urlretrieve(remote_url, local_file)
