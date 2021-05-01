r"""
Curl Opertor
============

For geophysical problems, the relationship between two physical quantities may include the curl:

.. math::
    \nabla \times \vec{u} = \Bigg ( \dfrac{\partial u_y}{\partial z} - \dfrac{\partial u_z}{\partial y} \Bigg )\hat{x} - \Bigg ( \dfrac{\partial u_x}{\partial z} - \dfrac{\partial u_z}{\partial x} \Bigg )\hat{y} + \Bigg ( \dfrac{\partial u_x}{\partial y} - \dfrac{\partial u_y}{\partial x} \Bigg )\hat{z}

For discretized quantities living on 2D or 3D meshes, sparse matricies can be used to
approximate the curl operator. For each mesh type, the curl
operator is a property that is only constructed when called.

This tutorial focusses on:

    - how to construct the curl operator
    - applying the curl operator to a discrete quantity
    - mapping and dimensions

"""

#####################################################################
# Background Theory
# -----------------
# 
# Let us define two continuous vector functions :math:`\vec{u}` and :math:`\vec{w}` such that:
# 
# .. math::
#     \vec{w} = \nabla \times \vec{u}
# 
# And let :math:`\boldsymbol{u}` and :math:`\boldsymbol{w}` be the discrete representations of :math:`\vec{u}` and :math:`\vec{w}`
# that live on the mesh. Provided we know the discrete values :math:`\boldsymbol{u}`,
# our goal is to use discrete differentiation to approximate the vector components of :math:`\boldsymbol{w}`.
# We begin by considering a single 3D cell. We let the indices :math:`i`, :math:`j` and :math:`k` 
# denote positions along the x, y and z axes, respectively.
# 
# .. figure:: ../../images/curl_discretization.png
#     :align: center
#     :width: 800
# 
#     Discretization for approximating the x, y and z components of the curl on the respective faces of a 3D cell.
# 
# 
# As we will see, it makes the most sense for the vector components of :math:`\boldsymbol{u}` to live on the edges
# for the vector components of :math:`\boldsymbol{w}` to live the faces. In this case, we need to approximate:
# 
# 
#     - the partial derivatives :math:`\dfrac{\partial u_y}{\partial z}` and :math:`\dfrac{\partial u_z}{\partial y}` to compute :math:`w_x`,
#     - the partial derivatives :math:`\dfrac{\partial u_x}{\partial z}` and :math:`\dfrac{\partial u_z}{\partial x}` to compute :math:`w_y`, and
#     - the partial derivatives :math:`\dfrac{\partial u_x}{\partial y}` and :math:`\dfrac{\partial u_y}{\partial x}` to compute :math:`w_z`
# 
# **In 3D**, discrete values at 12 edge locations (4 x-edges, 4 y-edges and 4 z-edges) are used to
# approximate the vector components of the curl at 6 face locations (2 x-faces, 2-faces and 2 z-faces).
# An example of the approximation for each vector component is given below:
# 
# .. math::
#     \begin{align}
#     w_x \Big ( i,j \! +\!\!\frac{1}{2},k \! +\!\!\frac{1}{2} \Big ) \!\approx\! \; &
#     \!\Bigg ( \! \frac{u_z (i,j \! +\!\!1,k \! +\!\!\frac{1}{2})  \! -\! u_z (i,j,k \! +\!\!\frac{1}{2})}{h_y} \Bigg) \!
#     \! -\! \!\Bigg ( \! \frac{u_y (i,j \! +\!\!\frac{1}{2},k \! +\!\!1)  \! -\! u_y (i,j \! +\!\!\frac{1}{2},k)}{h_z} \Bigg) \! \\
#     & \\
#     w_y \Big ( i \! +\!\!\frac{1}{2},j,k \! +\!\!\frac{1}{2} \Big ) \!\approx\! \; &
#     \!\Bigg ( \! \frac{u_x (i \! +\!\!\frac{1}{2},j,k \! +\!\!1)  \! -\! u_x (i \! +\!\!\frac{1}{2},j,k)}{h_z} \Bigg)
#     \! -\! \!\Bigg ( \! \frac{u_z (i \! +\!\!1,j,k \! +\!\!\frac{1}{2})  \! -\! u_z (i,j,k \! +\!\!\frac{1}{2})}{h_x} \Bigg) \! \\
#     & \\
#     w_z \Big ( i \! +\!\!\frac{1}{2},j \! +\!\!\frac{1}{2},k \Big ) \!\approx\! \; &
#     \!\Bigg ( \! \frac{u_y (i \! +\!\!1,j \! +\!\!\frac{1}{2},k)  \! -\! u_y (i,j \! +\!\!\frac{1}{2},k)}{h_x} \Bigg )
#     \! -\! \!\Bigg ( \! \frac{u_x (i \! +\!\!\frac{1}{2},j \! +\!\!1,k)  \! -\! u_x (i \! +\!\!\frac{1}{2},j,k)}{h_y} \Bigg) \!
#     \end{align}
# 
# 
# Ultimately we are trying to approximate the curl on all the faces within a mesh.
# Adjacent cells share edges. If the components :math:`u_x`, :math:`u_y` and :math:`u_z` are
# continuous across at the edges, then :math:`\boldsymbol{u}` and :math:`\boldsymbol{w}`
# can be related by a sparse matrix-vector product:
# 
# .. math::
#     \boldsymbol{w} = \boldsymbol{C \, u}
# 
# where :math:`\boldsymbol{C}` is the curl matrix from edges to faces,
# :math:`\boldsymbol{u}` is a vector that stores the components of :math:`\vec{u}` on cell edges,
# and :math:`\boldsymbol{w}` is a vector that stores the components of :math:`\vec{w}` on cell faces such that:
# 
# .. math::
#     \boldsymbol{u} = \begin{bmatrix} \boldsymbol{u_x} \\ \boldsymbol{u_y} \\ \boldsymbol{u_z} \end{bmatrix}
#     \;\;\;\; \textrm{and} \;\;\;\; \begin{bmatrix} \boldsymbol{w_x} \\ \boldsymbol{w_y} \\ \boldsymbol{w_z} \end{bmatrix}
# 
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
# Here we apply the curl operator to a vector that lives
# on the edges of a 2D tensor mesh. We then plot the results.
# The goal is to demonstrate the construction of the
# discrete curl operator. In practise, the curl is only present
# in 3D problems.
#

# Create a uniform grid
h = np.ones(20)
mesh = TensorMesh([h, h], "CC")

# Get the curl operator
CURL = mesh.edgeCurl  # Curl edges to cell centers (goes to faces in 3D)

# Evaluate curl of a vector function in x and y
edges_x = mesh.gridEx
edges_y = mesh.gridEy

wx = (-edges_x[:, 1] / np.sqrt(np.sum(edges_x ** 2, axis=1))) * np.exp(
    -(edges_x[:, 0] ** 2 + edges_x[:, 1] ** 2) / 6 ** 2
)

wy = (edges_y[:, 0] / np.sqrt(np.sum(edges_y ** 2, axis=1))) * np.exp(
    -(edges_y[:, 0] ** 2 + edges_y[:, 1] ** 2) / 6 ** 2
)

w = np.r_[wx, wy]
curl_w = CURL * w

# Plot curl of w
fig = plt.figure(figsize=(11, 5))

ax1 = fig.add_subplot(121)
mesh.plotImage(
    w, ax=ax1, v_type="E", view="vec", stream_opts={"color": "w", "density": 1.0}
)
ax1.set_title("w at cell edges")

ax2 = fig.add_subplot(122)
mesh.plotImage(curl_w, ax=ax2)
ax2.set_title("curl of w at cell centers")

fig.show()


#############################################
# Mapping and Dimensions
# ----------------------
#
# When discretizing and solving differential equations, it is
# natural for vector quantities on cell edges or on cell faces
# The curl operator is generally defined by map from edges to faces;
# although one could theoretically construct a curl operator which
# maps from faces to edges.
# 
# Here we construct the curl operator for a tensor mesh and for
# a tree mesh. By plotting the operators on a spy plot, we gain
# an understanding of the dimensions
# of the gradient operator and its structure.
#

# Create a basic tensor mesh
h = np.ones(20)
tensor_mesh = TensorMesh([h, h, h], "CCC")

# Create a basic tree mesh
h = np.ones(32)
tree_mesh = TreeMesh([h, h, h], origin="CCC")
xp, yp, zp = np.meshgrid([-8., 8.], [-8., 8.], [-8., 8.])
xyz = np.c_[mkvc(xp), mkvc(yp), mkvc(zp)]
tree_mesh = refine_tree_xyz(tree_mesh, xyz, octree_levels=[1, 1], method="box", finalize=False)
tree_mesh.finalize()

# Construct curl operators
tensor_curl = tensor_mesh.edge_curl  # 3D curl from edges to faces
tree_curl = tree_mesh.edge_curl  # 3D curl from edges to faces

fig = plt.figure(figsize=(9, 4))

ax1 = fig.add_axes([0.15, 0.05, 0.35, 0.85])
ax1.spy(tensor_curl, markersize=0.5)
ax1.set_title("Tensor Mesh Curl")

ax2 = fig.add_axes([0.65, 0.05, 0.30, 0.85])
ax2.spy(tree_curl, markersize=0.5)
ax2.set_title("Tree Mesh Curl")

fig.show()

# Print some properties
print("Curl on Tensor Mesh:")
print("- Number of faces:", str(tensor_mesh.nF))
print("- Number of edges:", str(tensor_mesh.nE))
print("- Dimensions of operator:", str(tensor_mesh.nE), "x", str(tensor_mesh.nF))
print("- Number of non-zero elements:", str(tensor_curl.nnz))

print("Curl on Tree Mesh:")
print("- Number of faces:", str(tree_mesh.nF))
print("- Number of edges:", str(tree_mesh.nE))
print("- Dimensions of operator:", str(tree_mesh.nE), "x", str(tree_mesh.nF))
print("- Number of non-zero elements:", str(tree_curl.nnz))


