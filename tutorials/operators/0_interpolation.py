"""
Interpolation Matricies
=======================

Interpolation is required when a discrete quantity is known on the mesh (centers, nodes, edges or faces),
but we would like to estimate its value at locations within the continuous domain.
Here, we demonstrate how a sparse matrix can be formed which interpolates
the discrete values to a set of locations in continuous space.
The focus of this tutorial is as follows:

    - How to construct and apply interpolation matrices in 1D, 2D and 3D
    - Interpolation on different mesh types
    - Interpolation of scalars and vectors

:ref:`See our theory section on interpolation operators <operators_interpolation>`

"""

###############################################
#
# Import Packages
# ---------------
#
# Here we import the packages required for this tutorial.
#

from discretize import TensorMesh, TreeMesh
from discretize.utils import refine_tree_xyz
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
mpl.rcParams.update({'font.size':14})

# sphinx_gallery_thumbnail_number = 2


#############################################
# 1D Example
# ----------
#
# Here discretize a scalar function to live on cell nodes and
# interpolate the values to a set of locations within the domain.
# We then compute the scalar function at these locations to
# validate the interpolation operator.
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

#############################################
# Interpolate a Scalar Quantity in 2D
# -----------------------------------
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

#############################################
# Interpolate a Vector Quantity in 2D
# -----------------------------------
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
mesh.plotImage(
    u_faces, ax=ax1, v_type="F", view="vec",
    stream_opts={"color": "w", "density": 1.0}, clim=[0.0, 10.0],
)
ax1.set_title("True Vector on Faces")

ax2 = fig.add_axes([0.35, 0.15, 0.22, 0.75])
mesh.plotImage(
    u_interp, ax=ax2, v_type="F", view="vec",
    stream_opts={"color": "w", "density": 1.0}, clim=[0.0, 10.0],
)
ax2.set_title("Interpolated from Edges to Faces")

ax3 = fig.add_axes([0.65, 0.15, 0.22, 0.75])
mesh.plotImage(
    u_faces-u_interp, ax=ax3, v_type="F", view="vec",
    stream_opts={"color": "w", "density": 1.0}, clim=[0.0, 10.0],
)
ax3.set_title("Error")

ax4 = fig.add_axes([0.92, 0.15, 0.025, 0.75])
norm = mpl.colors.Normalize(vmin=0., vmax=10.)
cbar = mpl.colorbar.ColorbarBase(
    ax4, norm=norm, orientation="vertical"
)
