import cupy as cp
import numpy as np
from gamma.simulator.gamma import domain_mgr
cp.cuda.Device(0).use()

domain = domain_mgr(filename='thinwall.k')
nodes = domain.nodes.get()
base = np.isclose(nodes[:, 2], -4.0)          # base of substrate
idx = np.where(base)[0]
xc, yc = nodes[base, 0].mean(), nodes[base, 1].mean()
d = (nodes[base, 0] - xc)**2 + (nodes[base, 1] - yc)**2
node = int(idx[np.argmin(d)])
print("number of base nodes (z=-4.0):", int(base.sum()))
print("chosen base-center node index:", node)
print("its (x, y, z):", nodes[node])
