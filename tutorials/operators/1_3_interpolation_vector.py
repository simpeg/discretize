r"""
Interpolating and Averaging Vector Quantities in 2D and 3D
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
    - how to apply the interpolation and averaging to vector functions
    

"""

#############################################
# Background Theory
# -----------------
#
# We will begin by presenting the theory for 2D interpolation, then extend the
# theory to cover 3D interpolation.
# The components of a vector quantity are discretized to live either on
# their respective mesh faces or edges. Thus different components of the
# vector are being interpolated from different locations.
# 
# .. figure:: ../../images/interpolation_2d_vectors.png
#     :align: center
#     :width: 600
# 
#     A tensor mesh in 2D denoting interpolation from faces (left) and edges (right).
# 
# Let :math:`\vec{u} (x,y)` be a 2D vector function that is known on the faces of the mesh;
# that is, :math:`u_x` lives on the x-faces and :math:`u_y` lives on the y-faces. 
# Note that in the above figure, the x-faces and y-faces both form tensor grids.
# If we want to approximate the components of the vector at a location :math:`(x^*,y^*)`,
# we simply need to treat each component as a scalar function and interpolate it separately.
# 
# Where :math:`u_{x,i}` represents the x-component of :math:`\vec{u} (x,y)` on a face :math:`i` being used for the interpolation,
# the approximation of the x-component at :math:`(x^*, y^*)` has the form:
# 
# .. math::
#     u_x(x^*, y^*) \approx a_0 u_{x,0} + a_1 u_{x,1} + a_2 u_{x,2} + a_3 u_{x,3}
#     :label: operators_interpolation_xvec_coef
# 
# For the the y-component, we have a similar representation:
# 
# .. math::
#     u_y(x^*, y^*) \approx b_0 u_{y,0} + b_1 u_{y,1} + b_2 u_{y,2} + b_3 u_{y,3}
# 
# Where :math:`\boldsymbol{u}` is a vector that organizes the discrete components of :math:`\vec{u} (x,y)` on cell faces,
# and :math:`\boldsymbol{u^*}` is a vector organizing the components of the approximations of :math:`\vec{u}(x,y)` at an arbitrary number of locations,
# the interpolation matrix :math:`\boldsymbol{P}` is defined by:
# 
# .. math::
#     \boldsymbol{u^*} \approx \boldsymbol{P \, u}
#     :label: operators_interpolation_2d_sys
# 
# where
# 
# .. math::
#     \boldsymbol{u} = \begin{bmatrix} \boldsymbol{u_x} \\ \boldsymbol{u_y} \end{bmatrix}
#     \;\;\textrm{,}\;\;\;\;
#     \boldsymbol{u^*} = \begin{bmatrix} \boldsymbol{u_x^*} \\ \boldsymbol{u_y^*} \end{bmatrix}
#     \;\;\;\;\textrm{and}\;\;\;\;
#     \boldsymbol{P} = \begin{bmatrix} \boldsymbol{P_x} & \boldsymbol{0} \\ \boldsymbol{0} & \boldsymbol{P_y} \end{bmatrix}
# 
# The interpolation matrix :math:`\boldsymbol{P}` is a sparse block-diagonal matrix.
# The size of the interpolation matrix is the number of locations by the number of faces in the mesh.
# 
# **What if we want to interpolate from edges?**
# 
# In this case, the derivation is effectively the same.
# However, the locations used for the interpolation are different and
# :math:`\boldsymbol{u}` is now a vector that organizes the discrete components of :math:`\vec{u} (x,y)` on cell edges.
# 
# 
# **What if we are interpolating a 3D vector?**
# 
# In this case, there are 8 face locations or 8 edge locations that are used to approximate
# :math:`\vec{u}(x,y,z)` at each location :math:`(x^*, y^*, z^*)`.
# Similar to expression :eq:`operators_interpolation_xvec_coef` we have:
# 
# .. math::
#     \begin{align}
#     u_x(x^*, y^*, z^*) & \approx \sum_{i=1}^7 a_i u_{x,i} \\
#     u_y(x^*, y^*, z^*) & \approx \sum_{i=1}^7 b_i u_{y,i} \\
#     u_z(x^*, y^*, z^*) & \approx \sum_{i=1}^7 c_i u_{z,i}
#     \end{align}
# 
# The interpolation can be expressed similar to that in equation :eq:`operators_interpolation_2d_sys`,
# however:
# 
# .. math::
#     \boldsymbol{u} = \begin{bmatrix} \boldsymbol{u_x} \\ \boldsymbol{u_y} \\ \boldsymbol{u_z} \end{bmatrix}
#     \;\;\textrm{,}\;\;\;\;
#     \boldsymbol{u^*} = \begin{bmatrix} \boldsymbol{u_x^*} \\ \boldsymbol{u_y^*} \\ \boldsymbol{u_z^*} \end{bmatrix}
#     \;\;\;\;\textrm{and}\;\;\;\;
#     \boldsymbol{P} = \begin{bmatrix} \boldsymbol{P_x} & \boldsymbol{0} & \boldsymbol{0} \\
#     \boldsymbol{0} & \boldsymbol{P_y} & \boldsymbol{0} \\
#     \boldsymbol{0} & \boldsymbol{0} & \boldsymbol{P_z} 
#     \end{bmatrix}
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
# Constructing and Applying a 2D/3D Interpolation Matrix
# ------------------------------------------------------
#
# Here we discretize a vector quantity to live on the edges of a 2D tensor mesh,
# where the x component lives on x-edges and the y component lives on y-edges.
# We then use interpolation matrices to approximate the vector components
# on the faces of the mesh. That is, we interpolate the x component from x-edges
# to x-faces, and we interpolate the y component from y-edges to y-faces.
# Since the x and y components of vectors are discretized at different locations
# on the mesh, separate interpolation matrices must be constructed for the
# x and y components.
#

# Create a tensor mesh
h = np.ones(75)
mesh = TensorMesh([h, h], "CC")

# Define the x and y components of the vector function.
# In this case, the vector function is a circular vortex.
def fun_x(xy):
    r = np.sqrt(np.sum(xy ** 2, axis=1))
    return 5. * (-xy[:, 1] / r) * (1 + np.tanh(0.15 * (28.0 - r)))

def fun_y(xy):
    r = np.sqrt(np.sum(xy ** 2, axis=1))
    return 5. * (xy[:, 0] / r) * (1 + np.tanh(0.15 * (28.0 - r)))

# Evaluate x and y components of the vector on x and y edges, respectively
edges_x = mesh.edges_x
edges_y = mesh.edges_y

ux_edges = fun_x(edges_x)
uy_edges = fun_y(edges_y)
u_edges = np.r_[ux_edges, uy_edges]

# Compute true x and y components of the vector on x and y faces, respectively
faces_x = mesh.faces_x
faces_y = mesh.faces_y

ux_faces = fun_x(faces_x)
uy_faces = fun_y(faces_y)
u_faces = np.r_[ux_faces, uy_faces]

# Generate the interpolation matricies and interpolate from edges to faces.
# Interpolation matrices from edges and faces assume all vector components
# are defined on their respective edges or faces. Thus an interpolation matrix
# from x-edges will extract the x component values then interpolate to locations.
Px = mesh.get_interpolation_matrix(faces_x, "Ex")
Py = mesh.get_interpolation_matrix(faces_y, "Ey")

ux_interp = Px*u_edges
uy_interp = Py*u_edges
u_interp = np.r_[ux_interp, uy_interp]

# Plotting
fig = plt.figure(figsize=(14, 4))

ax1 = fig.add_axes([0.05, 0.15, 0.22, 0.75])
mesh.plot_image(
    u_faces, ax=ax1, v_type="F", view="vec",
    stream_opts={"color": "w", "density": 1.0}, clim=[0.0, 10.0],
)
ax1.set_title("True Vector on Faces")

ax2 = fig.add_axes([0.35, 0.15, 0.22, 0.75])
mesh.plot_image(
    u_interp, ax=ax2, v_type="F", view="vec",
    stream_opts={"color": "w", "density": 1.0}, clim=[0.0, 10.0],
)
ax2.set_title("Interpolated from Edges to Faces")

ax3 = fig.add_axes([0.65, 0.15, 0.22, 0.75])
mesh.plot_image(
    u_faces-u_interp, ax=ax3, v_type="F", view="vec",
    stream_opts={"color": "w", "density": 1.0}, clim=[0.0, 10.0],
)
ax3.set_title("Error")

ax4 = fig.add_axes([0.92, 0.15, 0.025, 0.75])
norm = mpl.colors.Normalize(vmin=0., vmax=10.)
cbar = mpl.colorbar.ColorbarBase(
    ax4, norm=norm, orientation="vertical"
)


#########################################################
# Constructing and Applying a 2D Averaging Matrix
# -----------------------------------------------
#
# Here we compute a vector function that lives on the edges.
# We then create an averaging operator to approximate the components
# of the vector at cell centers.
#

# Create a tensor mesh
h = np.ones(75)
mesh = TensorMesh([h, h], "CC")

# Define the x and y components of the vector function.
# In this case, the vector function is a circular vortex.
def fun_x(xy):
    r = np.sqrt(np.sum(xy ** 2, axis=1))
    return 5. * (-xy[:, 1] / r) * (1 + np.tanh(0.15 * (28.0 - r)))

def fun_y(xy):
    r = np.sqrt(np.sum(xy ** 2, axis=1))
    return 5. * (xy[:, 0] / r) * (1 + np.tanh(0.15 * (28.0 - r)))

# Evaluate x and y components of the vector on x and y edges, respectively
edges_x = mesh.edges_x
edges_y = mesh.edges_y

ux_edges = fun_x(edges_x)
uy_edges = fun_y(edges_y)
u_edges = np.r_[ux_edges, uy_edges]

# Compute true x and y components of the vector at cell centers
centers = mesh.cell_centers

ux_centers = fun_x(centers)
uy_centers = fun_y(centers)
u_centers = np.r_[ux_centers, uy_centers]

# Create the averaging operator for a vector from edges to cell centers
A = mesh.average_edge_to_cell_vector

# Apply the averaging operator
u_average = A*u_edges

# Plotting
fig = plt.figure(figsize=(14, 4))

ax1 = fig.add_axes([0.05, 0.15, 0.22, 0.75])
mesh.plot_image(
    u_edges, ax=ax1, v_type="E", view="vec",
    stream_opts={"color": "w", "density": 1.0}, clim=[0.0, 10.0],
)
ax1.set_title("True Vector on Edges")

ax2 = fig.add_axes([0.35, 0.15, 0.22, 0.75])
mesh.plot_image(
    u_average, ax=ax2, v_type="CCv", view="vec",
    stream_opts={"color": "w", "density": 1.0}, clim=[0.0, 10.0],
)
ax2.set_title("Averaged to Cell Centers")

ax3 = fig.add_axes([0.65, 0.15, 0.22, 0.75])
mesh.plot_image(
    u_centers-u_average, ax=ax3, v_type="CCv", view="vec",
    stream_opts={"color": "w", "density": 1.0}, clim=[0.0, 10.0],
)
ax3.set_title("Error")

ax4 = fig.add_axes([0.92, 0.15, 0.025, 0.75])
norm = mpl.colors.Normalize(vmin=0., vmax=10.)
cbar = mpl.colorbar.ColorbarBase(
    ax4, norm=norm, orientation="vertical"
)
