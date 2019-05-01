"""
Tensor meshes
=============

Tensor meshes are the most basic class of meshes that can be created with
discretize. They belong to the class (:class:`~discretize.TensorMesh`).
Tensor meshes can be defined in 1, 2 or 3 dimensions. Here we demonstrate:

    - How to create basic tensor meshes
    - How to include padding cells
    - How to plot tensor meshes
    - How to extract properties from meshes

"""

###############################################
#
# Import Packages
# ---------------
#
# Here we import the packages required for this tutorial.
#

from discretize import TensorMesh
import numpy as np

# sphinx_gallery_thumbnail_number = 3

###############################################
# Basic Example
# -------------
#
# The easiest way to define a tensor mesh is to define the cell widths in
# x, y and z as 1D numpy arrays. And to provide the position of the bottom
# southwest corner of the mesh. We demonstrate this here for a 2D mesh (thus
# we do not need to consider the z-dimension).

ncx = 10     # number of core mesh cells in x
ncy = 15     # number of core mesh cells in y
dx = 15      # base cell width x
dy = 10      # base cell width y
hx = dx*np.ones(ncx)
hy = dy*np.ones(ncy)

x0 = 0
y0 = -150

mesh = TensorMesh([hx, hy], x0=[x0, y0])

mesh.plotGrid()


###############################################
# Padding Cells
# -------------
#
# For practical purposes, the user may want to define a region where the cell
# widths are increasing/decreasing in size. For example, padding is often used
# to define a large domain while reducing the total number of mesh cells.
# Here we demonstrate how to create tensor meshes that have padding cells.
#

ncx = 10      # number of core mesh cells in x
ncy = 15      # number of core mesh cells in y
dx = 15       # base cell width x
dy = 10       # base cell width y
npad_x = 4    # number of padding cells in x
npad_y = 4    # number of padding cells in y
exp_x = 1.25  # expansion rate of padding cells in x
exp_y = 1.25  # expansion rate of padding cells in y

# Use a list of tuples to define cell widths in each direction. Each tuple
# contains the cell with, number of cells and the expansion factor (+ve/-ve).
hx = [(dx, npad_x, -exp_x), (dx, ncx), (dx, npad_x, exp_x)]
hy = [(dy, npad_y, -exp_y), (dy, ncy), (dy, npad_y, exp_y)]

# We can use flags 'C', '0' and 'N' to define the xyz position of the mesh.
mesh = TensorMesh([hx, hy], x0='CN')

mesh.plotGrid()

###############################################
# Extracting Mesh Properties
# --------------------------
#
# Once the mesh is created, you may want to extract certain properties. Here,
# we show some properties that can be extracted from 2D meshes.
#

ncx = 10      # number of core mesh cells in x
ncy = 15      # number of core mesh cells in y
dx = 15       # base cell width x
dy = 10       # base cell width y
npad_x = 4    # number of padding cells in x
npad_y = 4    # number of padding cells in y
exp_x = 1.25  # expansion rate of padding cells in x
exp_y = 1.25  # expansion rate of padding cells in y

hx = [(dx, npad_x, -exp_x), (dx, ncx), (dx, npad_x, exp_x)]
hy = [(dy, npad_y, -exp_y), (dy, ncy), (dy, npad_y, exp_y)]

mesh = TensorMesh([hx, hy], x0='C0')

# The bottom west corner
x0 = mesh.x0

# The total number of cells
nC = mesh.nC

# An (nC, 2) array containing the cell-center locations
cc = mesh.gridCC

# A boolean array specifying which cells lie on the boundary
bInd = mesh.cellBoundaryInd

# The cell areas (2D "volume")
s = mesh.vol
mesh.plotImage(s, grid=True)

###############################################
# 3D Example
# ----------
#
# Here we show how the same approach can be used to create and extract
# properties from a 3D tensor mesh.
#

nc = 10      # number of core mesh cells in x, y and z
dh = 10      # base cell width in x, y and z
npad = 5     # number of padding cells
exp = 1.25   # expansion rate of padding cells

h = [(dh, npad, -exp), (dh, nc), (dh, npad, exp)]
mesh = TensorMesh([h, h, h], x0='C00')

# The bottom southwest corner
x0 = mesh.x0

# The total number of cells
nC = mesh.nC

# An (nC, 3) array containing the cell-center locations
cc = mesh.gridCC

# A boolean array specifying which cells lie on the boundary
bInd = mesh.cellBoundaryInd

# The cell volumes
v = mesh.vol

mesh.plotImage(v, grid=True)
