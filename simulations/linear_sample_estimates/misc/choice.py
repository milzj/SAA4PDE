import numpy as np
import matplotlib.pyplot as plt

from base.signif import signif

import os

outdir = "output/"
if not os.path.exists(outdir):
	os.makedirs(outdir)


grid_points = 12
x = np.linspace(-1, 1, grid_points)

N = 10000
z = np.random.choice(x, N)

fig, (ax1,ax2) = plt.subplots(1,2)

ax1.hist(z, weights=np.ones(N)/N, bins=grid_points)

z = []
for i in range(N):
	z.append(np.random.choice(x,1))

z = np.array(z)
ax2.hist(z, weights=np.ones(N)/N, bins=grid_points)
fig.suptitle(r"Relative frequency = {}".format(signif(1/12)))

plt.savefig("output/choice.pdf")
