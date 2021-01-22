r"""
Interpolating and Averaging Scalar Quantities in 2D and 3D
==========================================================

Interpolation is used when a discrete quantity is known on the mesh (centers, nodes, edges or faces),
but we would like to estimate its value at locations within the continuous domain.
For any mesh type, *discretize* allows the user to construct a sparse interpolation matrix
for a corresponding set of locations.

In *discretize*, averaging matrices are constructed when a discrete quantity must be mapped
between centers, nodes, edges or faces.
Averaging matrices are a property of the mesh and are constructed when called.

In this tutorial, we demonstrate:

	- how to construct interpolation and averaging matrices
	- how to apply the interpolation and averaging to scalar functions
    

"""

#############################################
# Background Theory
# -----------------
#
# We will begin by presenting the theory for 2D interpolation, then extend the
# theory to cover 3D interpolation. In 2D, the location of the interpolated
# quantity lies either within 4 nodes or cell centers.
# 
# .. figure:: ../../images/interpolation_2d.png
#     :align: center
#     :width: 300
# 
#     A tensor mesh in 2D denoting interpolation from nodes (blue) and cell centers (red).
# 
# Let :math:`(x^*, y^*)` be within a cell whose nodes are located at
# :math:`(x_1, y_1)`, :math:`(x_2, y_1)`, :math:`(x_1, y_2)` and :math:`(x_2, y_2)`.
# If we define :math:`u_0 = u(x_1, y_1)`, :math:`u_1 = u(x_2, y_1)`, :math:`u_2 = u(x_1, y_2)` and
# :math:`u_3 = u(x_2, y_2)`, then
# 
# .. math::
#     u(x^*, y^*) \approx a_0 u_0 + a_1 u_1 + a_2 u_2 + a_3 u_3
# 
# where :math:`a_0`, :math:`a_1`, :math:`a_2` and :math:`a_3` are coefficients determined from equations
# governing `bilinear interpolation <https://en.wikipedia.org/wiki/Bilinear_interpolation>`__ .
# These coefficients represent the 4 non-zero values within the corresponding row of the interpolation matrix :math:`\boldsymbol{P}`.
# 
# Where the values of :math:`u(x,y)` at all nodes are organized into a single vector :math:`\boldsymbol{u}`,
# and :math:`\boldsymbol{u^*}` is a vector containing the approximations of :math:`u(x,y)` at an arbitrary number of locations:
# 
# .. math::
#     \boldsymbol{u^*} \approx \boldsymbol{P\, u}
#     :label: operators_interpolation_general
# 
# In each row of :math:`\boldsymbol{P}`, the position of the non-zero elements :math:`a_0`, :math:`a_1`, :math:`a_2` and :math:`a_3`
# corresponds to the indecies of the 4 nodes comprising a specific cell.
# Once again the shape of :math:`\boldsymbol{P}` is the number of locations by the number of nodes.
# 
# **What if the function is defined at cell centers?**
# 
# A similar result can be obtained by interpolating a function define at cell centers.
# In this case, we let :math:`(x^*, y^*)` lie within 4 cell centers located at
# :math:`(\bar{x}_1, \bar{y}_1)`, :math:`(\bar{x}_2, \bar{y}_1)`, :math:`(\bar{x}_1, \bar{y}_2)` and :math:`(\bar{x}_2, \bar{y}_2)`.
# 
# .. math::
#     u(x^*, y^*) \approx a_0 \bar{u}_0 + a_1 \bar{u}_1 + a_2 \bar{u}_2 + a_3 \bar{u}_3
# 
# The resulting interpolation is defined similar to expression :eq:`operators_interpolation_general`.
# However the size of the resulting interpolation matrix is the number of locations by number of cells.
# 
# **What about for 3D case?**
# 
# The derivation for the 3D case is effectively the same, except 8 node or center locations must
# be used in the interpolation. Thus:
# 
# .. math::
#     u(x^*, y^*, z^*) \approx \sum_{k=0}^7 a_k u_k
# 
# This creates an interpolation matrix :math:`\boldsymbol{P}` with 8 non-zero entries per row.
# To learn how to compute the value of the coefficients :math:`a_k`,
# see `trilinear interpolation (3D) <https://en.wikipedia.org/wiki/Trilinear_interpolation>`__
#


################################################
# Import Packages
# ---------------
#

from discretize import TensorMesh, TreeMesh
from discretize.utils import refine_tree_xyz
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
mpl.rcParams.update({'font.size':14})

# sphinx_gallery_thumbnail_number = 1


###############################################################
# Constructing and Applying a 2D/3D Interpolation Matrix
# ------------------------------------------------------
#
# Here we discretize a scalar quantity to live at cell centers of a tree mesh.
# We then use the interpolation matrix to approximate the values of the
# scalar function along a profile. The approach for 2D and 3D meshes
# are essentially the same.
#

# Construct a tree mesh
h = 2* np.ones(128)
mesh = TreeMesh([h, h], x0="CC")

xy = np.c_[0., 0.]
mesh = refine_tree_xyz(mesh, xy, octree_levels=[8, 8, 8], method="radial", finalize=False)
mesh.finalize()  # Must finalize tree mesh before use

# Define the points along the profile
d = np.linspace(-100, 100, 21)  # distance along profile
phi = 35.                       # heading of profile
xp = d*np.cos(np.pi*phi/180.)
yp = d*np.sin(np.pi*phi/180.)

# Define a continuous 2D scalar function
def fun(x, y):
    return np.exp(-(x ** 2 + y ** 2) / 40 ** 2)

# Get all cell center locations from the mesh and evaluate function
# at the centers. Also compute true value at interpolation locations
centers = mesh.cell_centers
v_centers = fun(centers[:, 0], centers[:, 1])
v_true = fun(xp, yp)

# Create interpolation matrix and apply. When creating the interpolation matrix,
# we must define where the discrete quantity lives and where it is being
# interpolated to.
locations = np.c_[xp, yp]
P = mesh.get_interpolation_matrix(locations, 'CC')
v_interp = P * v_centers

# Plot mesh and profile line
fig = plt.figure(figsize=(14, 4.5))

ax1 = fig.add_axes([0.1, 0.15, 0.25, 0.75])
mesh.plot_grid(ax=ax1)
ax1.plot(xp, yp, 'ko')
ax1.set_xlim(np.min(mesh.nodes_x), np.max(mesh.nodes_x))
ax1.set_ylim(np.min(mesh.nodes_y), np.max(mesh.nodes_y))
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Tree mesh and profile line')

ax2 = fig.add_axes([0.43, 0.15, 0.5, 0.75])
ax2.plot(
    d, v_true, "k-",
    d, v_interp, "bo",
    d, np.c_[v_true - v_interp], "ro",
)
ax2.set_ylim([-0.1, 1.3])
ax2.set_title("Comparison plot")
ax2.set_xlabel("Position along profile")
ax2.legend((
    "true value", "interpolated from centers", "error"
))

fig.show()


#########################################################
# Constructing and Applying a 2D Averaging Matrix
# -----------------------------------------------
#
# Here we compute a scalar function at cell centers.
# We then create an averaging operator to approximated the function
# at the faces. We choose to define a scalar function that is 
# strongly discontinuous in some places
# to demonstrate how the averaging operator will smooth out
# discontinuities.
#

# Create mesh and obtain averaging operators
h = 2 * np.ones(50)
mesh = TensorMesh([h, h], x0="CC")

# Create a variable on cell centers
v = 100.0 * np.ones(mesh.nC)
xy = mesh.gridCC
v[(xy[:, 1] > 0)] = 0.0
v[(xy[:, 1] < -10.0) & (xy[:, 0] > -10.0) & (xy[:, 0] < 10.0)] = 50.0

# Create averaging operator
A = mesh.average_cell_to_face  # cell centers to faces

# Apply averaging operator
u = A*v

# Plot
fig = plt.figure(figsize=(11, 5))
ax1 = fig.add_subplot(121)
mesh.plot_image(v, ax=ax1)
ax1.set_title("Variable at cell centers")

ax2 = fig.add_subplot(122)
mesh.plot_image(u, ax=ax2, v_type="F")
ax2.set_title("Averaged to faces")

fig.show()

