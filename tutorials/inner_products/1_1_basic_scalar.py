r"""
Basic Scalar Inner Products
===========================

The inner product between two scalar quantities represents the most
basic class of inner products. For this class of inner products, we demonstrate:

    - How to construct the inner product matrix for scalar on nodes or at cell centers
    - How to use inner product matricies to approximate the inner product
    - How to construct the inverse of the inner product matrix

"""

############################################################################
# Background Theory
# -----------------
# 
# For scalar quantities :math:`\psi` and :math:`\phi`, the
# inner product is defined as:
# 
# .. math::
#     (\psi , \phi ) = \int_\Omega \psi \, \phi \, dv
# 
# In discretized form, we can approximate the aforementioned inner-products as:
# 
# .. math::
#     (\psi , \phi) \approx \mathbf{\psi^T \, M \, \phi}
# 
# where :math:`\mathbf{M}` represents the *inner-product matrix*.
# :math:`\mathbf{\psi}` and :math:`\mathbf{\phi}`
# are discrete variables that live on the mesh (nodes or cell centers).
# 
# It is important to note a few things about the inner-product matrix:
# 
#     1. It depends on the dimensions and discretization of the mesh
#     2. It depends on whether the discrete scalar quantities live at cell centers or on nodes
# 
# For this simple class of inner products, the inner product matricies for
# discrete scalar quantities living on various parts of the mesh have the form:
# 
# .. math::
#     \textrm{Centers:} \; \mathbf{M_c} &= \textrm{diag} (\mathbf{v} ) \\
#     \textrm{Nodes:} \; \mathbf{M_n} &= \frac{1}{2^{2k}} \mathbf{P_n^T } \textrm{diag} (\mathbf{v} ) \mathbf{P_n}
# where
# 
#     - :math:`\mathbf{v}` is a vector that contains the cell volumes
#     - :math:`k = 1,2,3` is the dimension (1D, 2D or 3D)
#     - :math:`\mathbf{P_n}` is a projection matrix that maps from nodes to cell centers
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
# Inner Product Matrices in 1D
# ----------------------------
#
# Here we define a scalar function (a Gaussian distribution):
#
# .. math::
#     \phi(x) = \frac{1}{\sqrt{2 \pi \sigma^2}} \, e^{- \frac{(x- \mu )^2}{2 \sigma^2}}
#
#
# We then use the inner product matrx to approximate the following inner product
# numerically:
#
# .. math::
#     (\phi , \phi) = \int_\Omega \phi^2 \, dx = \frac{1}{2\sigma \sqrt{\pi}}
#
#
# To evaluate our approximation, we compare agains the analytic solution.
# *Note that the method for evaluating inner products here can be
# extended to variables in 2D and 3D*.
#

# Define the Gaussian function
def fcn_gaussian(x, mu, sig):
    return (1 / np.sqrt(2 * np.pi * sig ** 2)) * np.exp(-0.5 * (x - mu) ** 2 / sig ** 2)


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
Mc = sdiag(mesh.vol)  # cell-centered
# Mn = mesh.getNodalInnerProduct()  # on nodes (*functionality pending*)

# Compute the inner product
ipt = 1 / (2 * sig * np.sqrt(np.pi))  # true value of (phi, phi)
ipc = np.dot(phi_c, (Mc * phi_c))  # inner product for cell centers
# ipn = np.dot(phi_n, (Mn*phi_n)) (*functionality pending*)

fig = plt.figure(figsize=(3, 3))
ax = fig.add_subplot(111)
ax.plot(mesh.cell_centers, phi_c)
ax.set_title("phi at cell centers")

# Verify accuracy
print("ACCURACY")
print("Analytic solution:    ", ipt)
print("Cell-centered approx.:", ipc)
# print('Nodal approx.:        ', ipn)



##############################################
# Inverse of Inner Product Matricies
# ----------------------------------
#
# The final discretized system using the finite volume method may contain
# the inverse of an inner-product matrix. Here we show how the inverse of
# the inner product matrix can be explicitly constructed. We then validate its
# accuracy for cell-centers and nodes by computing the following
# L2-norm for each:
#
# .. math::
#     \| \mathbf{u - M^{-1} M u} \|^2
#
# which we expect to be very small.
#


# Create a tensor mesh
h = 0.1 * np.ones(100)
mesh = TensorMesh([h, h], "CC")

# Inner product and inverse for cell centered scalar quantities
Mc = sdiag(mesh.vol)
Mc_inv = sdiag(1 / mesh.vol)

# Inner product and inverse for nodal scalar quantities
# Mn = mesh.get_nodal_inner_product()
# Mn_inv = mesh.get_nodal_inner_product(invert_matrix=True)

# Generate a random vector
phi_c = np.random.rand(mesh.nC)
phi_n = np.random.rand(mesh.nN)

# Compute the norm
norm_c = np.linalg.norm(phi_c - Mc_inv.dot(Mc.dot(phi_c)))
# norm_n = np.linalg.norm(phi_n - Mn_inv.dot(Mn.dot(phi_n)))

# Verify accuracy
print("ACCURACY")
print("Norm for centers:", norm_c)
print("Norm for nodes:", norm_c)
