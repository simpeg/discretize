r"""
Divergence Operator
===================

For geophysical problems, the relationship between two physical quantities may include the divergence:

.. math::
    \nabla \cdot \vec{u} = \dfrac{\partial u_x}{\partial x} + \dfrac{\partial u_y}{\partial y} + \dfrac{\partial u_y}{\partial y}

For discretized quantities living on 2D or 3D meshes, sparse matricies can be used to
approximate the divergence operator. For each mesh type, the divergence
operator is a property that is only constructed when called.

This tutorial focusses on:

    - how to construct the divergence operator
    - applying the divergence operator to a discrete quantity
    - mapping and dimensions

"""


#####################################################################
# Background Theory
# -----------------
# 
# Let us define a continuous scalar function :math:`\phi` and a continuous vector function :math:`\vec{u}` such that:
# 
# .. math::
#     \phi = \nabla \cdot \vec{u}
# 
# And let :math:`\boldsymbol{\phi}` and :math:`\boldsymbol{u}` be the discrete representations of :math:`\phi` and :math:`\vec{u}`
# that live on the mesh. Provided we know the discrete values :math:`\boldsymbol{u}`,
# our goal is to use discrete differentiation to approximate the values of :math:`\boldsymbol{\phi}`.
# We begin by considering a single cell (2D or 3D). We let the indices :math:`i`, :math:`j` and :math:`k` 
# denote positions along the x, y and z axes, respectively.
# 
# .. figure:: ../../images/divergence_discretization.png
#     :align: center
#     :width: 600
# 
#     Discretization for approximating the divergence at the center of a single 2D cell (left) and 3D cell (right).
# 
# As we will see, it makes the most sense for :math:`\boldsymbol{\phi}` to live at the cell centers and
# for the components of :math:`\boldsymbol{u}` to live on the faces. If :math:`u_x` lives on x-faces, then its discrete
# derivative with respect to :math:`x` lives at the cell center. And if :math:`u_y` lives on y-faces its discrete
# derivative with respect to :math:`y` lives at the cell center. Likewise for :math:`u_z`. Thus to approximate the
# divergence of :math:`\vec{u}` at the cell center, we simply need to sum the discrete derivatives of :math:`u_x`, :math:`u_y`
# and :math:`u_z` that are defined at the cell center. Where :math:`h_x`, :math:`h_y` and :math:`h_z` represent the dimension of the cell along the x, y and
# z directions, respectively:
# 
# .. math::
#     \begin{align}
#     \mathbf{In \; 2D:} \;\; \phi(i,j) \approx \; & \frac{u_x(i,j+\frac{1}{2}) - u_x(i,j-\frac{1}{2})}{h_x} \\
#     & + \frac{u_y(i+\frac{1}{2},j) - u_y(i-\frac{1}{2},j)}{h_y}
#     \end{align}
# 
# |
# 
# .. math::
#     \begin{align}
#     \mathbf{In \; 3D:} \;\; \phi(i,j,k) \approx \; & \frac{u_x(i+\frac{1}{2},j,k) - u_x(i-\frac{1}{2},j,k)}{h_x} \\
#     & + \frac{u_y(i,j+\frac{1}{2},k) - u_y(i,j-\frac{1}{2},k)}{h_y} \\
#     & + \frac{u_z(i,j,k+\frac{1}{2}) - u_z(i,j,k-\frac{1}{2})}{h_z}
#     \end{align}
# 
# 
# Ultimately we are trying to approximate the divergence at the center of every cell in a mesh.
# Adjacent cells share faces. If the components :math:`u_x`, :math:`u_y` and :math:`u_z` are
# continuous across their respective faces, then :math:`\boldsymbol{\phi}` and :math:`\boldsymbol{u}`
# can be related by a sparse matrix-vector product:
# 
# .. math::
#     \boldsymbol{\phi} = \boldsymbol{D \, u}
# 
# where :math:`\boldsymbol{D}` is the divergence matrix from faces to cell centers,
# :math:`\boldsymbol{\phi}` is a vector containing the discrete approximations of :math:`\phi` at all cell centers,
# and :math:`\boldsymbol{u}` stores the components of :math:`\vec{u}` on cell faces as a vector of the form:
# 
# .. math::
#     \boldsymbol{u} = \begin{bmatrix} \boldsymbol{u_x} \\ \boldsymbol{u_y} \\ \boldsymbol{u_z} \end{bmatrix}
# 

###############################################
#
# Import Packages
# ---------------
#
# Here we import the packages required for this tutorial.
#

from discretize import TensorMesh, TreeMesh
from discretize.utils import mkvc, refine_tree_xyz
import matplotlib.pyplot as plt
import numpy as np

# sphinx_gallery_thumbnail_number = 1

#############################################
# 2D Example
# ----------
#
# Here we apply the divergence operator to a vector
# that lives on the faces of a 2D tensor mesh.
# We then plot the results.
#

# Create a uniform grid
h = np.ones(20)
mesh = TensorMesh([h, h], "CC")

# Get divergence
DIV = mesh.face_divergence  # Divergence from faces to cell centers

# Evaluate divergence of a vector function in x and y
faces_x = mesh.gridFx
faces_y = mesh.gridFy

vx = (faces_x[:, 0] / np.sqrt(np.sum(faces_x ** 2, axis=1))) * np.exp(
    -(faces_x[:, 0] ** 2 + faces_x[:, 1] ** 2) / 6 ** 2
)

vy = (faces_y[:, 1] / np.sqrt(np.sum(faces_y ** 2, axis=1))) * np.exp(
    -(faces_y[:, 0] ** 2 + faces_y[:, 1] ** 2) / 6 ** 2
)

v = np.r_[vx, vy]
div_v = DIV * v

# Plot divergence of v
fig = plt.figure(figsize=(11, 5))

ax1 = fig.add_subplot(121)
mesh.plotImage(
    v, ax=ax1, v_type="F", view="vec", stream_opts={"color": "w", "density": 1.0}
)
ax1.set_title("v at cell faces")

ax2 = fig.add_subplot(122)
mesh.plotImage(div_v, ax=ax2)
ax2.set_title("divergence of v at cell centers")

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
# As a result, the divergence operator will map from one part of
# the mesh to another; either edges to nodes or faces to cell centers.
# 
# Here we construct the divergence operator for a tensor mesh and for
# a tree mesh. By plotting the operators on a spy plot, we gain
# an understanding of the dimensions
# of the divergence operator and its structure.
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

# Construct divergence operators
tensor_divergence = tensor_mesh.face_divergence  # 2D divergence from faces to centers
tree_divergence = tree_mesh.face_divergence  # 2D divergence from faces to centers

# Plot divergence operators
fig = plt.figure(figsize=(6, 5))

ax1 = fig.add_axes([0.15, 0.55, 0.8, 0.35])
ax1.spy(tensor_divergence, markersize=0.5)
ax1.set_title("2D Tensor Mesh Divergence")

ax2 = fig.add_axes([0.15, 0.05, 0.8, 0.35])
ax2.spy(tree_divergence, markersize=0.5)
ax2.set_title("2D Tree Mesh Divergence")

fig.show()

print("Faces to Centers on Tensor Mesh:")
print("- Number of faces:", str(tensor_mesh.nF))
print("- Number of cells:", str(tensor_mesh.nC))
print("- Dimensions of operator:", str(tensor_mesh.nC), "x", str(tensor_mesh.nF))
print("- Number of non-zero elements:", str(tensor_divergence.nnz), "\n")

print("Faces to Centers on Tree Mesh:")
print("- Number of faces:", str(tree_mesh.nF))
print("- Number of cells:", str(tree_mesh.nC))
print("- Dimensions of operator:", str(tree_mesh.nC), "x", str(tree_mesh.nF))
print("- Number of non-zero elements:", str(tree_divergence.nnz), "\n")


