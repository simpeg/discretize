r"""
Gradient Operator
=================

For geophysical problems, the relationship between two physical quantities may include the gradient:

.. math::
    \nabla \phi = \dfrac{\partial \phi}{\partial x}\hat{x} + \dfrac{\partial \phi}{\partial y}\hat{y} + \dfrac{\partial \phi}{\partial z}\hat{z}

For discretized quantities living on 1D, 2D or 3D meshes, sparse matricies can be used to
approximate the gradient operator. For each mesh type, the gradient
operator is a property that is only constructed when called.

This tutorial focusses on:

    - how to construct the gradient operator
    - applying the gradient operator to a discrete quantity
    - mapping and dimensions or the gradient operator

"""

#####################################################################
# Background Theory
# -----------------
#
# Let us define a continuous scalar function :math:`\phi` and a continuous vector function :math:`\vec{u}` such that:
# 
# .. math::
#     \vec{u} = \nabla \phi
# 
# And let :math:`\boldsymbol{\phi}` and :math:`\boldsymbol{u}` be the discrete representations of :math:`\phi` and :math:`\vec{u}`
# that live on the mesh. Provided we know the discrete values :math:`\boldsymbol{\phi}`,
# our goal is to use discrete differentiation to approximate the vector components of :math:`\boldsymbol{u}`.
# We begin by considering a single cell (2D or 3D). We let the indices :math:`i`, :math:`j` and :math:`k` 
# denote positions along the x, y and z axes, respectively.
# 
# .. figure:: ../../images/gradient_discretization.png
#     :align: center
#     :width: 600
# 
#     Discretization for approximating the gradient on the edges of a single 2D cell (left) and 3D cell (right).
# 
# As we will see, it makes the most sense for :math:`\boldsymbol{\phi}` to live at the cell nodes and
# for the components of :math:`\boldsymbol{u}` to live on corresponding edges. If :math:`\phi` lives on the nodes, then:
# 
#     - the partial derivative :math:`\dfrac{\partial \phi}{\partial x}\hat{x}` lives on x-edges,
#     - the partial derivative :math:`\dfrac{\partial \phi}{\partial y}\hat{y}` lives on y-edges, and
#     - the partial derivative :math:`\dfrac{\partial \phi}{\partial z}\hat{z}` lives on z-edges
# 
# Thus to approximate the gradient of :math:`\phi`, 
# we simply need to take discrete derivatives of :math:`\phi` with respect to :math:`x`, :math:`y` and :math:`z`,
# and organize the resulting vector components on the corresponding edges.
# Let :math:`h_x`, :math:`h_y` and :math:`h_z` represent the dimension of the cell along the x, y and
# z directions, respectively.
# 
# **In 2D**, the value of :math:`\phi` at 4 node locations is used to approximate the vector components of the
# gradient at 4 edges locations (2 x-edges and 2 y-edges) as follows:
# 
# .. math::
#     \begin{align}
#     u_x \Big ( i+\frac{1}{2},j \Big ) \approx \; & \frac{\phi (i+1,j) - \phi (i,j)}{h_x} \\
#     u_x \Big ( i+\frac{1}{2},j+1 \Big ) \approx \; & \frac{\phi (i+1,j+1) - \phi (i,j+1)}{h_x} \\
#     u_y \Big ( i,j+\frac{1}{2} \Big ) \approx \; & \frac{\phi (i,j+1) - \phi (i,j)}{h_y} \\
#     u_y \Big ( i+1,j+\frac{1}{2} \Big ) \approx \; & \frac{\phi (i+1,j+1) - \phi (i+1,j)}{h_y}
#     \end{align}
# 
# **In 3D**, the value of :math:`\phi` at 8 node locations is used to approximate the vector components of the
# gradient at 12 edges locations (4 x-edges, 4 y-edges and 4 z-edges). An example of the approximation
# for each vector component is given below:
# 
# .. math::
#     \begin{align}
#     u_x \Big ( i+\frac{1}{2},j,k \Big ) \approx \; & \frac{\phi (i+1,j,k) - \phi (i,j,k)}{h_x} \\
#     u_y \Big ( i,j+\frac{1}{2},k \Big ) \approx \; & \frac{\phi (i,j+1,k) - \phi (i,j,k)}{h_y} \\
#     u_z \Big ( i,j,k+\frac{1}{2} \Big ) \approx \; & \frac{\phi (i,j,k+1) - \phi (i,j,k)}{h_z}
#     \end{align}
# 
# 
# Ultimately we are trying to approximate the vector components of the gradient at all edges of a mesh.
# Adjacent cells share nodes. If :math:`\phi` is continuous at the nodes, then :math:`\boldsymbol{\phi}` and :math:`\boldsymbol{u}`
# can be related by a sparse matrix-vector product:
# 
# .. math::
#     \boldsymbol{u} = \boldsymbol{G \, \phi}
# 
# where :math:`\boldsymbol{G}` is the gradient matrix that maps from nodes to edges,
# :math:`\boldsymbol{\phi}` is a vector containing :math:`\phi` at all nodes,
# and :math:`\boldsymbol{u}` stores the components of :math:`\vec{u}` on cell edges as a vector of the form:
# 
# .. math::
#     \boldsymbol{u} = \begin{bmatrix} \boldsymbol{u_x} \\ \boldsymbol{u_y} \\ \boldsymbol{u_z} \end{bmatrix}
# 
# 

###############################################
# Import Packages
# ---------------
#

from discretize import TensorMesh, TreeMesh
from discretize.utils import mkvc, refine_tree_xyz
import matplotlib.pyplot as plt
import numpy as np

# sphinx_gallery_thumbnail_number = 2


#############################################
# 1D Example
# ----------
#
# For the 1D case, the gradient operator simply approximates the derivative
# of a function. Here we compute a scalar function on cell nodes and
# differentiate with respect to x. We then compute the analytic
# derivative of function to validate the numerical differentiation.
#

# Create a uniform grid
h = np.ones(20)
mesh = TensorMesh([h], "C")

# Get node and cell center locations
x_nodes = mesh.vectorNx
x_centers = mesh.vectorCCx

# Compute function on nodes and derivative at cell centers
v = np.exp(-(x_nodes ** 2) / 4 ** 2)
dvdx = -(2 * x_centers / 4 ** 2) * np.exp(-(x_centers ** 2) / 4 ** 2)

# Derivative in x (gradient in 1D) from nodes to cell centers
G = mesh.nodalGrad
dvdx_approx = G * v

# Compare
fig = plt.figure(figsize=(12, 4))
ax1 = fig.add_axes([0.1, 0.02, 0.25, 0.8])
ax1.spy(G, markersize=5)
ax1.set_title("Spy plot of gradient", pad=10)

ax2 = fig.add_axes([0.4, 0.06, 0.55, 0.85])
ax2.plot(x_nodes, v, "b-", x_centers, dvdx, "r-", x_centers, dvdx_approx, "ko")
ax2.set_title("Comparison plot")
ax2.legend(("function", "analytic derivative", "numeric derivative"))

fig.show()

#############################################
# 2D Example
# ----------
#
# Here we apply the gradient to a scalar function that lives
# on the nodes of a 2D tensor mesh. This produces a
# a discrete approximation of the gradient that lives
# on the edges of the mesh.
#

# Create a uniform grid
h = np.ones(20)
mesh = TensorMesh([h, h], "CC")

# Get the gradient
GRAD = mesh.nodal_gradient  # Gradient from nodes to edges

# Evaluate gradient of a scalar function
nodes = mesh.nodes
u = np.exp(-(nodes[:, 0] ** 2 + nodes[:, 1] ** 2) / 4 ** 2)
grad_u = GRAD * u

# Plot Gradient of u
fig = plt.figure(figsize=(11, 5))

ax1 = fig.add_subplot(121)
mesh.plotImage(u, ax=ax1, v_type="N")
ax1.set_title("u on cell nodes")

ax2 = fig.add_subplot(122)
mesh.plotImage(
    grad_u, ax=ax2, v_type="E", view="vec", stream_opts={"color": "w", "density": 1.0}
)
ax2.set_title("gradient of u on edges")

fig.show()


#############################################
# Mapping and Dimensions
# ----------------------
#
# When discretizing and solving differential equations, it is
# natural for:
#
#    - Scalar quantities on nodes or at cell centers
#    - Vector quantities on cell edges or on cell faces
#
# As a result, the gradient operator will map from one part of
# the mesh to another; either nodes to edges or cell centers to faces.
# 
# Here we construct the gradient operator for a tensor mesh and for
# a tree mesh. By plotting the operators on a spy plot, we gain
# an understanding of the dimensions
# of the gradient operator and its structure.
#

# Create a basic tensor mesh
h = np.ones(20)
tensor_mesh = TensorMesh([h, h], "CC")

# Create a basic tree mesh
h = np.ones(32)
tree_mesh = TreeMesh([h, h], origin="CC")
xp, yp = np.meshgrid([-8., 8.], [-8., 8.])
xy = np.c_[mkvc(xp), mkvc(yp)]
tree_mesh = refine_tree_xyz(tree_mesh, xy, octree_levels=[1, 1], method="box", finalize=False)
tree_mesh.finalize()

# Plot meshes
fig, ax = plt.subplots(1, 2, figsize=(11, 5))

tensor_mesh.plot_grid(ax=ax[0])
ax[0].set_title("Tensor Mesh")

tree_mesh.plot_grid(ax=ax[1])
ax[1].set_title("Tree Mesh")

# Construct gradient operators
tensor_gradient = tensor_mesh.cell_gradient  # 2D gradient from centers to faces
tree_gradient = tree_mesh.nodal_gradient  # 2D gradient from nodes to edges 

# Plot gradient operators
fig = plt.figure(figsize=(7, 5))

ax1 = fig.add_axes([0.15, 0.05, 0.30, 0.85])
ax1.spy(tensor_gradient, markersize=0.5)
ax1.set_title("Tensor Gradient")

ax2 = fig.add_axes([0.65, 0.05, 0.30, 0.85])
ax2.spy(tree_gradient, markersize=0.5)
ax2.set_title("Tree Gradient")

fig.show()

# Print some properties
print("\n Centers to Faces on Tensor Mesh:")
print("- Number of nodes:", str(tensor_mesh.nN))
print("- Number of edges:", str(tensor_mesh.nE))
print("- Dimensions of operator:", str(tensor_mesh.nE), "x", str(tensor_mesh.nN))
print("- Number of non-zero elements:", str(tensor_gradient.nnz), "\n")

print("\n Nodes to Edges on Tree Mesh:")
print("- Number of nodes:", str(tree_mesh.nN))
print("- Number of edges:", str(tree_mesh.nE))
print("- Dimensions of operator:", str(tree_mesh.nE), "x", str(tree_mesh.nN))
print("- Number of non-zero elements:", str(tree_gradient.nnz), "\n")
