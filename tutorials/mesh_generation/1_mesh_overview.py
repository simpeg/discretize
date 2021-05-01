r"""
Meshes Overview
===============

Here we provide a basic overview of the meshes supported by *discretize*
and define some useful terminology.
Detailed tutorials covering the construction and properties of each mesh type are found in
separate tutorials.


"""

###############################################################################
# Meshes Supported by Discretize
# ------------------------------
# 
# A multitude of mesh types are supported by the *discretize* package.
# Each mesh type has its advantages and disadvantages when being considered to solve
# differential equations with the finite volume approach.
# The user should consider the dimension (1D, 2D, 3D) and
# any spatial symmetry in the PDE when choosing a mesh type.
# Mesh types supported by *discretize* include:
# 
#     - **Tensor Meshes:** A mesh where the grid locations are organized according to tensor products
#     - **Tree Meshes:** A mesh where the dimensions of cells are :math:`2^n` larger than the dimension of the smallest cell size
#     - **Curvilinear Meshes:** A tensor mesh where the axes are curvilinear
#     - **Cylindrical Meshes:** A pseudo-2D mesh for solving 3D problems with perfect symmetry in the radial direction
# 
# .. figure:: ../../images/mesh_types.png
#     :align: center
#     :width: 700
# 
#     Examples of different mesh types supported by the *discretize* package.
#

###############################################################################
# Cells and Where Variables Live
# ------------------------------
# 
# The small volumes within the mesh are called *cells*.
# In *discretize*, we use a staggered mimetic finite volume approach :cite:`haber2014,HymanShashkov1999`.
# This approach requires discrete variables to live at cell-centers, nodes, faces, or edges.
# Below, we illustrate the valid locations for discrete quantities for a single cell where:
# 
#   - **Centers:** center locations of each cell
#   - **Nodes:** locations of intersection between grid lines defining the mesh
#   - **X, Y and Z edges:** edges whose tangent lines are parallel to the X, Y and Z axes, respectively
#   - **X, Y and Z faces:** faces which are normal to the orientation of the X, Y and Z axes, respectively
# 
# 
# .. figure:: ../../images/cell_locations.png
#     :align: center
#     :width: 700
# 
#     Locations of centers, nodes, faces and edges for 2D cells (left) and 3D cells (right).
#
# Below, we create a 2D tensor mesh and plot a visual representation of where discretized
# quantities are able to live.
#

import numpy as np
import discretize
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':14})

# sphinx_gallery_thumbnail_number = 1

hx = np.r_[3, 1, 1, 3]
hy = np.r_[3, 2, 1, 1, 1, 1, 2, 3]
tensor_mesh = discretize.TensorMesh([hx, hy])

# Plot
fig, axes = plt.subplots(1, 3, figsize=(14.5, 5))
tensor_mesh.plot_grid(ax=axes[0], nodes=True, centers=True)
axes[0].legend(("Nodes", "Centers"))
axes[0].set_title("Nodes and cell centers")

tensor_mesh.plot_grid(ax=axes[1], edges=True)
axes[1].legend(("X-edges", "Y-edges"))
axes[1].set_title("Cell edges")

tensor_mesh.plot_grid(ax=axes[2], faces=True)
axes[2].legend(("X-faces", "Y-faces"))
axes[2].set_title("Cell faces")
