r"""
Interpolation and Averaging in 1D
=================================

Interpolation is used when a discrete quantity is known on the mesh (centers, nodes, edges or faces),
but we would like to estimate its value at locations within the continuous domain.
For any mesh type, *discretize* allows the user to construct a sparse interpolation matrix
for a corresponding set of locations.

In *discretize*, averaging matrices are constructed when a discrete quantity must be mapped
between centers, nodes, edges or faces.
Averaging matrices are a property of the mesh and are constructed when called.

In this tutorial, we demonstrate:

	- how to construct interpolation and averaging matrices
	- how to apply the interpolation and averaging to 1D functions
    

"""

#############################################
# Background Theory
# -----------------
#
# Let us define a 1D mesh that contains 8 cells of arbitrary width.
# The mesh is illustrated in the figure below. The width of each cell is
# defined as :math:`h_i`. The location of each node is defined as :math:`x_i`.
# 
# .. figure:: ../../images/interpolation_1d.png
#     :align: center
#     :width: 600
#     :name: operators_interpolation_1d
# 
#     Tensor mesh in 1D.
# 
# Now let :math:`u(x)` be a function whose values are known at the nodes;
# i.e. :math:`u_i = u(x_i)`.
# The approximate value of the function at location :math:`x^*` 
# using linear interpolation is given by:
# 
# .. math::
#     u(x^*) \approx u_3 + \Bigg ( \frac{u_4 - u_3}{h_3} \Bigg ) (x^* - x_3)
#     :label: operators_averaging_interpolation_1d
# 
# 
# Suppose now that we organize the known values of :math:`u(x)` at the nodes
# into a vector of the form:
# 
# .. math::
#     \boldsymbol{u} = \begin{bmatrix} u_0 & u_1 & u_2 & u_3 & u_4 & u_5 & u_6 & u_7 & u_8 \end{bmatrix}^T
# 
# If we define a row:
# 
# .. math::
#     \boldsymbol{p_0} = \begin{bmatrix} 0 & 0 & 0 & a_3 & a_4 & 0 & 0 & 0 & 0 \end{bmatrix}
# 
# where
# 
# .. math::
#     a_3 = 1 - \frac{x^* - x_3}{h_3} \;\;\;\;\; \textrm{and} \;\;\;\;\; a_4 = \frac{x^* - x_3}{h_3}
# 
# then
# 
# .. math::
#     u(x^*) \approx \boldsymbol{p_0 \, u}
# 
# For a single location, we have just seen how a linear operator can be constructed to
# compute the interpolation using a matrix vector-product.
# 
# Now consider the case where you would like to interpolate the function from the nodes to
# an arbitrary number of locations within the boundaries of the mesh.
# For each location, we simply construct the corresponding row in the interpolation matrix.
# Where :math:`\boldsymbol{u^*}` is a vector containing the approximations of :math:`u(x)` at :math:`M`
# locations:
# 
# .. math::
#     \boldsymbol{u^*} \approx \boldsymbol{P\, u} \;\;\;\;\;\; \textrm{where} \;\;\;\;\;\;
#     \boldsymbol{P} = \begin{bmatrix} \cdots \;\; \boldsymbol{p_0} \;\; \cdots \\
#     \cdots \;\; \boldsymbol{p_1} \;\; \cdots \\ \vdots \\
#     \cdots \, \boldsymbol{p_{M-1}} \, \cdots \end{bmatrix}
#     :label: operators_averaging_interpolation_matrix
# 
# :math:`\boldsymbol{P}` is a sparse matrix whose rows contain a maximum of 2 non-zero elements.
# The size of :math:`\boldsymbol{P}` is the number of locations by the number of nodes.
# For seven locations (:math:`x^* = 3,1,9,2,5,2,10`) and our mesh (9 nodes),
# the non-zero elements of the interpolation matrix are illustrated below.
# 
# .. figure:: ../../images/interpolation_1d_sparse.png
#     :align: center
#     :width: 250
# 


###############################################
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
# Constructing and Applying a 1D Interpolation Matrix
# ---------------------------------------------------
#
# Here we discretize a continuous scalar function to live on cell nodes,
# then interpolate the values to a set of locations within the domain.
# Next, we compute the scalar function at the specified locations to
# validate out 1D interpolation operator.
#

# Create a uniform grid
h = 10 * np.ones(20)
mesh = TensorMesh([h], "C")

# Define locations
x_nodes = mesh.nodes_x

# Define a set of locations for the interpolation
np.random.seed(6)
x_interp = np.random.uniform(np.min(x_nodes), np.max(x_nodes), 20)

# Define a continuous function
def fun(x):
    return np.exp(-(x ** 2) / 50 ** 2)

# Compute function on nodes and at the location
v_nodes = fun(x_nodes)
v_true = fun(x_interp)

# Create interpolation matrix and apply. When creating the interpolation matrix,
# we must define where the discrete quantity lives and where it is being
# interpolated to.
P = mesh.get_interpolation_matrix(x_interp, 'N')
v_interp = P * v_nodes

# Compare
fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_axes([0.1, 0.05, 0.25, 0.8])
ax1.spy(P, markersize=5)
ax1.set_title("Spy plot for interpolation operator", pad=15)

k = np.argsort(x_interp)

ax2 = fig.add_axes([0.45, 0.1, 0.5, 0.8])
ax2.plot(
    x_nodes, v_nodes, 'k',
    x_interp[k], v_true[k], "b^",
    x_interp[k], v_interp[k], "gv",
    x_interp[k], np.c_[v_true[k] - v_interp[k]], "ro",
)
ax2.set_ylim([-0.1, 1.5])
ax2.set_title("Comparison plot")
ax2.legend(
    (
    "original function", "true value at locations",
    "interpolated from nodes", "error"
    ), loc="upper right"
)

fig.show()


#########################################################
# Constructing and Applying a 1D Averaging Matrix
# -----------------------------------------------
#
# Here we compute a scalar function on cell nodes and average to cell centers.
# We then compute the scalar function at cell centers to validate the
# averaging operator.
#

# Create a uniform grid
h = 10 * np.ones(20)
mesh = TensorMesh([h], "C")

# Get node and cell center locations
x_nodes = mesh.vectorNx
x_centers = mesh.vectorCCx

# Define a continuous function
def fun(x):
    return np.exp(-(x ** 2) / 50 ** 2)

# Compute function on nodes and cell centers
v_nodes = fun(x_nodes)
v_centers = fun(x_centers)

# Create operator and average from nodes to cell centers
A = mesh.average_node_to_cell
v_approx = A * v_nodes

# Compare
fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_axes([0.1, 0.05, 0.25, 0.8])
ax1.spy(A, markersize=5)
ax1.set_title("Sparse representation of A", pad=10)

ax2 = fig.add_axes([0.45, 0.1, 0.5, 0.8])
ax2.plot(
    x_centers, v_centers, "b-",
    x_centers, v_approx, "ko",
    x_centers, np.c_[v_centers - v_approx], "r-"
)
ax2.set_title("Comparison plot")
ax2.legend(("evaluated at centers", "averaged from nodes", "error"))

fig.show()

