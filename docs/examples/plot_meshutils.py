"""
``meshutils`` to create complicated meshes
==========================================

The examples demonstrate some of the tools in `utils.meshutils`.
"""

# %matplotlib notebook
import discretize
import numpy as np
import matplotlib.pyplot as plt
import sys

if sys.version_info[0] < 3:
    print("This example only runs on Python 3")
    sys.exit(0)

###############################################################################
# 1. Get stretched grid with one stretching factor for various restrictions
# -------------------------------------------------------------------------
#
# - Source at 500 => Domain is centered around this point
# - Frequency 10 Hz, Resistivity 1 Ohm m
# - 2^5 cells
h_min, domain = discretize.utils.meshutils.get_domain(x0=500, freq=10, rho=1)

hx = discretize.utils.meshutils.get_stretched_h(h_min, domain, nx=2**5, x0=500)
grid = discretize.TensorMesh([hx, 1], x0=[domain[0], 0])
grid.plotGrid()

###############################################################################
# 1.b As before, but with following additional restrictions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# - My model goes from -800 to 3500, so I want this extent at least.
# - I want the minimum cell width to be within 5-100 meters, no smaller, no
#   wider.
# - From 300 to 800 meter I want regular spacing with min cell width.

h_min, domain = discretize.utils.meshutils.get_domain(
    x0=500, freq=10, rho=1, limits=[-800, 3500], min_width=[5, 100])

hx = discretize.utils.meshutils.get_stretched_h(
        h_min, domain, nx=2**5, x0=300, x1=800)
grid = discretize.TensorMesh([hx, 1], x0=[domain[0], 0])
grid.plotGrid()
mesh.plot_3d_slicer(Lpout)

###############################################################################
# 1.c As before, but for ``f=0.1`` instead
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

h_min, domain = discretize.utils.meshutils.get_domain(
    x0=500, freq=0.1, rho=1, limits=[-800, 3500], min_width=[5, 100])

hx = discretize.utils.meshutils.get_stretched_h(
        h_min, domain, nx=2**5, x0=300, x1=800)
grid = discretize.TensorMesh([hx, 1], x0=[domain[0], 0])
grid.plotGrid()
