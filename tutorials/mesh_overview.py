"""
Overview of available meshes
============================

`discretize` provides a numerical grid on which to solve
differential equations. Each mesh type has a similar API to make switching
between different meshes relatively simple.

"""

import numpy as np
import discretize
import matplotlib.pyplot as plt

###############################################################################
# Tensor Mesh
# -----------
#
# Tensor meshes (:class:`~discretize.TensorMesh`) are defined by a vectors of cell widths in
# each dimension. They can be defined in 1, 2, or 3 dimensions.
#

ncx = 16
ncy = 16
tensor_mesh = discretize.TensorMesh([ncx, ncy])

tensor_mesh.plotGrid()


###############################################################################
# Cylindrical Meshes
# ------------------
#
# Cylindrical meshes are variation on the tensor mesh. The are still defined by
# a vector in each dimension, but those dimensions are now in cylindrical,
# rather than cartesian coordinates.
#
