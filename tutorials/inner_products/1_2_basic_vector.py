r"""
Basic Vector Inner Products
===========================

The inner product between two vector quantities represents one of the most
basic classes of inner products. For this class of inner products, we demonstrate:

    - How to construct the inner product matrix
    - How to use inner product matricies to approximate the inner product
    - How to construct the inverse of the inner product matrix.

"""

############################################################################
# Background Theory
# -----------------
# 
# For vector quantities :math:`\vec{u}` and :math:`\vec{w}`, the
# inner product is given by:
# 
# .. math::
#     (\vec{u}, \vec{w}) = \int_\Omega \vec{u} \cdot \vec{w} \, dv
# 
# In discretized form, we can approximate the aforementioned inner-products as:
# 
# .. math::
#     (\vec{u}, \vec{w}) \approx \boldsymbol{u^T M \, w}
# 
# where :math:`\mathbf{M}` represents the *inner-product matrix*.
# :math:`\boldsymbol{u}` and :math:`\boldsymbol{w}`
# are discrete variables that live on the mesh (face or edges).
# 
# It is important to note a few things about the inner-product matrix:
# 
#     1. It depends on the dimensions and discretization of the mesh
#     2. It depends on whether the discrete scalar quantities live on faces or edges
# 
# For this simple class of inner products, the corresponding form of the inner product matrices for
# discrete vector quantities living on faces and edges are shown below:
# 
# .. math::
#     \textrm{Vectors on faces:} \; \boldsymbol{M_f} &= \frac{1}{4} \boldsymbol{P_f^T } \textrm{diag} (\boldsymbol{e_k \otimes v} ) \boldsymbol{P_f} \\
#     \textrm{Vectors on edges:} \; \boldsymbol{M_e} &= \frac{1}{4^{k-1}} \boldsymbol{P_e^T } \textrm{diag} (\boldsymbol{e_k \otimes v}) \boldsymbol{P_e}
# 
# where
# 
#     - :math:`\boldsymbol{v}` is a vector that stores all of the volumes of the cells
#     - :math:`k = 1,2,3` represent the dimension (1D, 2D or 3D)
#     - :math:`\boldsymbol{e_k}` is a vector of 1s of length :math:`k`
#     - :math:`\otimes` is the kronecker product
#     - :math:`\boldsymbol{P_f}` is a projection matrix that maps quantities from faces to cell centers
#     - :math:`\boldsymbol{P_e}` is a projection matrix that maps quantities from faces to cell centers
#
#


####################################################
#
# Import Packages
# ---------------
#

from discretize.utils import sdiag
from discretize import TensorMesh
import matplotlib.pyplot as plt
import numpy as np

# sphinx_gallery_thumbnail_number = 1


#####################################################
# Approximating the Inner Product for 2D Vectors
# ----------------------------------------------
#
# To preserve the natural boundary conditions for each cell, it is standard
# practice to define fields on cell edges and fluxes on cell faces. Here we
# will define a 2D vector quantity:
#
# .. math::
#     \vec{u}(x,y) = \Bigg [ \frac{-y}{r} \hat{x} + \frac{x}{r} \hat{y} \Bigg ]
#     \, e^{-\frac{x^2+y^2}{2\sigma^2}}
#
# We will then evaluate the following inner product:
#
# .. math::
#     (\vec{u}, \vec{u}) = \int_\Omega \vec{u} \cdot \vec{u} \, da
#     = 2 \pi \sigma^2
#
# using inner-product matricies. Next we compare the numerical evaluation
# of the inner products with the analytic solution. *Note that the method for
# evaluating inner products here can be extended to variables in 3D*.
#

# Define vector components of the function
def fcn_x(xy, sig):
    return (-xy[:, 1] / np.sqrt(np.sum(xy ** 2, axis=1))) * np.exp(
        -0.5 * np.sum(xy ** 2, axis=1) / sig ** 2
    )

def fcn_y(xy, sig):
    return (xy[:, 0] / np.sqrt(np.sum(xy ** 2, axis=1))) * np.exp(
        -0.5 * np.sum(xy ** 2, axis=1) / sig ** 2
    )

# The analytic solution of (u, u)
sig = 1.5
ipt = np.pi * sig ** 2

# Create a tensor mesh that is sufficiently large
h = 0.1 * np.ones(100)
mesh = TensorMesh([h, h], "CC")

# Evaluate inner-product using edge-defined discrete variables
ux = fcn_x(mesh.edges_x, sig)
uy = fcn_y(mesh.edges_y, sig)
u = np.r_[ux, uy]

Me = mesh.get_edge_inner_product()  # Edge inner product matrix

ipe = np.dot(u, Me * u)

# Evaluate inner-product using face-defined discrete variables
ux = fcn_x(mesh.faces_x, sig)
uy = fcn_y(mesh.faces_y, sig)
u = np.r_[ux, uy]

Mf = mesh.get_face_inner_product()  # Edge inner product matrix

ipf = np.dot(u, Mf * u)


# Plot the vector function
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
mesh.plot_image(
    u, ax=ax, v_type="F", view="vec", stream_opts={"color": "w", "density": 1.0}
)
ax.set_title("u at cell faces")

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
# accuracy for edges and faces by computing the folling
# L2-norm for each:
#
# .. math::
#     \| \mathbf{u - M^{-1} M u} \|^2
#
# which we expect to be small
#


# Create a tensor mesh
h = 0.1 * np.ones(100)
mesh = TensorMesh([h, h], "CC")

# Cell centered for scalar quantities
Mc = sdiag(mesh.vol)
Mc_inv = sdiag(1 / mesh.vol)

# Inner product for edges
Me = mesh.get_edge_inner_product()
Me_inv = mesh.get_edge_inner_product(invert_matrix=True)

# Inner product for faces
Mf = mesh.get_face_inner_product()
Mf_inv = mesh.get_face_inner_product(invert_matrix=True)

# Generate some random vectors
vec_e = np.random.rand(mesh.nE)
vec_f = np.random.rand(mesh.nF)

# Compute norms
norm_e = np.linalg.norm(vec_e - Me_inv * Me * vec_e)
norm_f = np.linalg.norm(vec_f - Mf_inv * Mf * vec_f)

# Verify accuracy
print("ACCURACY")
print("Norm for edges:  ", norm_e)
print("Norm for faces:  ", norm_f)
