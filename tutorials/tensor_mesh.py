"""
Tensor meshes
=============

Tensor meshes (:class:`~discretize.TensorMesh`) are defined by a vectors of
cell widths in each dimension. They can be defined in 1, 2, or 3 dimensions.
"""

import numpy as np
import discretize

###############################################################################
# In the simplest case, you can instantiate a tensor mesh by providing the
# number of cells in each dimension. This assumes that the extent of each
# dimension is unity.

ncx = 16
ncy = 16
tensor_mesh = discretize.TensorMesh([ncx, ncy])

tensor_mesh.plotGrid()

###############################################################################
# If instead, you want a mesh that has 16 cells in each dimension and the width
# of each cell is 1 (so that the extent of the domain is 16), you can instead
# specify the vector of cell widths.

hx2 = np.ones(ncx)
hy2 = np.ones(ncy)
tensor_mesh2 = discretize.TensorMesh([hx2, hy2])

tensor_mesh2.plotGrid()

###############################################################################
# In one lin, the above mesh can be specified by providing tuples to TensorMesh
# (cell_size, number_of_cells)

cell_size = 1
tensor_mesh2b = discretize.TensorMesh([
    [(cell_size, ncx)],  # hx
    [(cell_size, ncy)]  # hy
])

###############################################################################
# The widths of the cells can be variable. For example when solving problems
# in electromagnetics, it is necessary to add "padding cells" which expand
# near the boundary so that we satisfy the boundary conditions

n_padding_cells = 4
padding_factor = 1.3
h_padding = padding_factor * np.arange(n_padding_cells)
h_core = np.ones(ncx)

hx3 = np.hstack([np.flipud(h_padding), h_core, h_padding])
hy3 = hx3

tensor_mesh3 = discretize.TensorMesh([hx3, hy3])

tensor_mesh3.plotGrid()

###############################################################################
# Since it is common to include padding in meshes for numerical simulations,
# we simplify the above steps by allowing you to provide a list of tuples
# describing parts of the mesh. When there is a third number in the tuple, it
# refers to the factor by which we increase the cell size. If the number is
# negative, the tensor is flipped so that the widest cells are first.

hx4 = [
    (cell_size, n_padding_cells, -padding_factor),  # padding on the left
    (cell_size, ncx),  # core part of the mesh (uniform cells)
    (cell_size, n_padding_cells, padding_factor),  # padding on the right
]
hy4 = hx4
tensor_mesh4 = discretize.TensorMesh([hx4, hy4])

tensor_mesh4.plotGrid()

###############################################################################
# The origin of the mesh can be moved by assigning the `x0` variable. When a
# mesh is instantiated, the `x0` can be provided as a keyword. Alternatively,
# it can be set by setting the `x0` property after a mesh has been created.
# Here, we will position the origin in the center of the mesh created in the
# previous step.

x0 = [-tensor_mesh4.hx.sum()/2, -tensor_mesh4.hy.sum()/2]
tensor_mesh4.x0 = x0

tensor_mesh4.plotGrid()

###############################################################################
# The above is equivalent to passing `x0='CC'` which states that you want both
# the x and y directions of the mesh to be centered. A 'N'
# will make the entire mesh negative, and a '0' (or a 0) will
# make the mesh start at zero.

tensor_mesh4b = discretize.TensorMesh([hx4, hx4], x0='CC').plotGrid()
