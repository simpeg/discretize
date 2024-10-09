r"""
Basic Inner Products
====================

Inner products between two scalar or vector quantities represents the most
basic class of inner products. For this class of inner products, we demonstrate:

    - How to construct the inner product matrix
    - How to use inner product matricies to approximate the inner product
    - How to construct the inverse of the inner product matrix.

For scalar quantities :math:`\psi` and :math:`\phi`, the
inner product is given by:

.. math::
    (\psi , \phi ) = \int_\Omega \psi \, \phi \, dv


And for vector quantities :math:`\vec{u}` and :math:`\vec{v}`, the
inner product is given by:

.. math::
    (\vec{u}, \vec{v}) = \int_\Omega \vec{u} \cdot \vec{v} \, dv


In discretized form, we can approximate the aforementioned inner-products as:

.. math::
    (\psi , \phi) \approx \mathbf{\psi^T \, M \, \phi}


and

.. math::
    (\vec{u}, \vec{v}) \approx \mathbf{u^T \, M \, v}


where :math:`\mathbf{M}` in either equation represents an
*inner-product matrix*. :math:`\mathbf{\psi}`, :math:`\mathbf{\phi}`,
:math:`\mathbf{u}` and :math:`\mathbf{v}` are discrete variables that live
on the mesh. It is important to note a few things about the
inner-product matrix in this case:

    1. It depends on the dimensions and discretization of the mesh
    2. It depends on where the discrete variables live; e.g. edges, faces, nodes, centers

For this simple class of inner products, the inner product matricies for
discrete quantities living on various parts of the mesh have the form:

.. math::
    \textrm{Centers:} \; \mathbf{M_c} &= \textrm{diag} (\mathbf{v} ) \\
    \textrm{Nodes:} \; \mathbf{M_n} &= \frac{1}{2^{2k}} \mathbf{P_n^T } \textrm{diag} (\mathbf{v} ) \mathbf{P_n} \\
    \textrm{Faces:} \; \mathbf{M_f} &= \frac{1}{4} \mathbf{P_f^T } \textrm{diag} (\mathbf{I_k \otimes v} ) \mathbf{P_f} \\
    \textrm{Edges:} \; \mathbf{M_e} &= \frac{1}{4^{k-1}} \mathbf{P_e^T } \textrm{diag} (\mathbf{I_k \otimes v}) \mathbf{P_e}

where :math:`k = 1,2,3`, :math:`\mathbf{I_k}` is the identity matrix and
:math:`\otimes` is the kronecker product. :math:`\mathbf{P}` are projection
matricies that map quantities from one part of the cell (nodes, faces, edges)
to cell centers.


"""

####################################################
#
# Import Packages
# ---------------
#
# Here we import the packages required for this tutorial
#

from discretize.utils import sdiag
from discretize import TensorMesh
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(8572)

# sphinx_gallery_thumbnail_number = 2


#####################################################
# Scalars
# -------
#
# It is natural for scalar quantities to live at cell centers or nodes. Here
# we will define a scalar function (a Gaussian distribution in this case):
#
# .. math::
#     \phi(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \, e^{- \frac{(x- \mu )^2}{2 \sigma^2}}
#
#
# We will then evaluate the following inner product:
#
# .. math::
#     (\phi , \phi) = \int_\Omega \phi^2 \, dx = \frac{1}{2\sigma \sqrt{\pi}}
#
#
# according to the mid-point rule using inner-product matricies. Next we
# compare the numerical approximation of the inner product with the analytic
# solution. *Note that the method for evaluating inner products here can be
# extended to variables in 2D and 3D*.
#


# Define the Gaussian function
def fcn_gaussian(x, mu, sig):
    return (1 / np.sqrt(2 * np.pi * sig**2)) * np.exp(-0.5 * (x - mu) ** 2 / sig**2)


# Create a tensor mesh that is sufficiently large
h = 0.1 * np.ones(100)
mesh = TensorMesh([h], "C")

# Define center point and standard deviation
mu = 0
sig = 1.5

# Evaluate at cell centers and nodes
phi_c = fcn_gaussian(mesh.gridCC, mu, sig)
phi_n = fcn_gaussian(mesh.gridN, mu, sig)

# Define inner-product matricies
Mc = sdiag(mesh.cell_volumes)  # cell-centered
# Mn = mesh.getNodalInnerProduct()  # on nodes (*functionality pending*)

# Compute the inner product
ipt = 1 / (2 * sig * np.sqrt(np.pi))  # true value of (f, f)
ipc = np.dot(phi_c, (Mc * phi_c))
# ipn = np.dot(phi_n, (Mn*phi_n)) (*functionality pending*)

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
ax.plot(mesh.gridCC, phi_c)
ax.set_title("phi at cell centers")

# Verify accuracy
print("ACCURACY")
print("Analytic solution:    ", ipt)
print("Cell-centered approx.:", ipc)
# print('Nodal approx.:        ', ipn)


#####################################################
# Vectors
# -------
#
# To preserve the natural boundary conditions for each cell, it is standard
# practice to define fields on cell edges and fluxes on cell faces. Here we
# will define a 2D vector quantity:
#
# .. math::
#     \vec{v}(x,y) = \Bigg [ \frac{-y}{r} \hat{x} + \frac{x}{r} \hat{y} \Bigg ]
#     \, e^{-\frac{x^2+y^2}{2\sigma^2}}
#
# We will then evaluate the following inner product:
#
# .. math::
#     (\vec{v}, \vec{v}) = \int_\Omega \vec{v} \cdot \vec{v} \, da
#     = 2 \pi \sigma^2
#
# using inner-product matricies. Next we compare the numerical evaluation
# of the inner products with the analytic solution. *Note that the method for
# evaluating inner products here can be extended to variables in 3D*.
#


# Define components of the function
def fcn_x(xy, sig):
    return (-xy[:, 1] / np.sqrt(np.sum(xy**2, axis=1))) * np.exp(
        -0.5 * np.sum(xy**2, axis=1) / sig**2
    )


def fcn_y(xy, sig):
    return (xy[:, 0] / np.sqrt(np.sum(xy**2, axis=1))) * np.exp(
        -0.5 * np.sum(xy**2, axis=1) / sig**2
    )


# Create a tensor mesh that is sufficiently large
h = 0.1 * np.ones(100)
mesh = TensorMesh([h, h], "CC")

# Define center point and standard deviation
sig = 1.5

# Evaluate inner-product using edge-defined discrete variables
vx = fcn_x(mesh.gridEx, sig)
vy = fcn_y(mesh.gridEy, sig)
v = np.r_[vx, vy]

Me = mesh.get_edge_inner_product()  # Edge inner product matrix

ipe = np.dot(v, Me * v)

# Evaluate inner-product using face-defined discrete variables
vx = fcn_x(mesh.gridFx, sig)
vy = fcn_y(mesh.gridFy, sig)
v = np.r_[vx, vy]

Mf = mesh.get_face_inner_product()  # Edge inner product matrix

ipf = np.dot(v, Mf * v)

# The analytic solution of (v, v)
ipt = np.pi * sig**2

# Plot the vector function
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
mesh.plot_image(
    v, ax=ax, v_type="F", view="vec", stream_opts={"color": "w", "density": 1.0}
)
ax.set_title("v at cell faces")

fig.show()

# Verify accuracy
print("ACCURACY")
print("Analytic solution:    ", ipt)
print("Edge variable approx.:", ipe)
print("Face variable approx.:", ipf)

##############################################
# Inverse of Inner Product Matricies
# ----------------------------------
#
# The final discretized system using the finite volume method may contain
# the inverse of an inner-product matrix. Here we show how the inverse of
# the inner product matrix can be explicitly constructed. We validate its
# accuracy for cell-centers, nodes, edges and faces by computing the folling
# L2-norm for each:
#
# .. math::
#     \| \mathbf{v - M^{-1} M v} \|^2
#
#


# Create a tensor mesh
h = 0.1 * np.ones(100)
mesh = TensorMesh([h, h], "CC")

# Cell centered for scalar quantities
Mc = sdiag(mesh.cell_volumes)
Mc_inv = sdiag(1 / mesh.cell_volumes)

# Nodes for scalar quantities  (*functionality pending*)
# Mn = mesh.getNodalInnerProduct()
# Mn_inv = mesh.getNodalInnerProduct(invert_matrix=True)

# Edges for vector quantities
Me = mesh.get_edge_inner_product()
Me_inv = mesh.get_edge_inner_product(invert_matrix=True)

# Faces for vector quantities
Mf = mesh.get_face_inner_product()
Mf_inv = mesh.get_face_inner_product(invert_matrix=True)

# Generate some random vectors
phi_c = rng.random(mesh.nC)
# phi_n = rng.random(mesh.nN)
vec_e = rng.random(mesh.nE)
vec_f = rng.random(mesh.nF)

# Generate some random vectors
norm_c = np.linalg.norm(phi_c - Mc_inv.dot(Mc.dot(phi_c)))
# norm_n = np.linalg.norm(phi_n - Mn_inv*Mn*phi_n)
norm_e = np.linalg.norm(vec_e - Me_inv * Me * vec_e)
norm_f = np.linalg.norm(vec_f - Mf_inv * Mf * vec_f)

# Verify accuracy
print("ACCURACY")
print("Norm for centers:", norm_c)
# print('Norm for nodes:  ', norm_n)
print("Norm for edges:  ", norm_e)
print("Norm for faces:  ", norm_f)
