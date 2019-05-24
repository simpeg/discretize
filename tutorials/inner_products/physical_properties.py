"""
Constitutive Relations 
======================

When solving PDEs using the finite volume approach, inner products may
contain constitutive relations; examples include Ohm's law and Hooke's law.
Here we demonstrate how to construct inner-product matricies, their inverses
and their derivatives with respect to constitutive parameters for this class
of inner-products.

Let :math:`\\vec{J}` and :math:`\\vec{E}` be two physically related
quantities. If their relationship is linear-isotropic (defined by a constant
:math:`\\sigma`), then the constitutive relation is given by:

.. math::
    \\vec{J} = \\sigma \\vec{E}

The inner product between a vector :math:`\\vec{v}` and the right-hand side
of this expression is given by:

.. math::
    (\\vec{v}, \\sigma \\vec{E} ) = \\int_\\Omega \\vec{v} \\cdot \\sigma \\vec{E} \\, dv

Just like in the previous tutorial, we would like to approximate this inner
product numerically using an *inner-product matrix* such that:

.. math::
    (\\vec{v}, \\sigma \\vec{E} ) \\approx \\mathbf{v^T M_\\sigma e}

where the inner product matrix now depends on:

    1. the dimension and discretization of the mesh
    2. where :math:`\\mathbf{v}` and :math:`\\mathbf{e}` live
    3. the property :math:`\\sigma`

Constitutive relations may also be defined by a tensor (:math:`\\Sigma`). In
this case, the constitutive relation is of the form:

.. math::
    \\vec{J} = \\Sigma \\vec{E}

where

.. math::
    \\Sigma = \\begin{bmatrix} \\sigma_{11} & \\sigma_{12} & \\sigma_{13} \n
    \\sigma_{21} & \\sigma_{22} & \\sigma_{23} \n
    \\sigma_{31} & \\sigma_{32} & \\sigma_{33} \\end{bmatrix}

The inner product between a vector :math:`\\vec{v}` and the right-hand side of
this expression is given by:

.. math::
    (\\vec{v}, \\Sigma \\vec{E} ) = \\int_\\Omega \\vec{v} \\cdot \\Sigma \\vec{E}  \\, dv

Once again we would like to approximate this inner
product numerically using an *inner-product matrix* such that:

.. math::
    (\\vec{v}, \\Sigma \\vec{E} ) \\approx \\mathbf{v^T M_\\Sigma e}
    


"""

####################################################
#
# Import Packages
# ---------------
#
# Here we import the packages required for this tutorial
#

from discretize.utils.matutils import sdiag
from discretize import TensorMesh
import numpy as np
import matplotlib.pyplot as plt

# sphinx_gallery_thumbnail_number = 1

#####################################################
# Inner Product for a Single Cell
# -------------------------------
#
# Here we compare the inner product matricies for a cubic single cell when the
# constitutive relationship is isotropic, linear and anisotropic. For
# natural cases, the tensor is symmetric and is thus defined by 6 parameters:
# 
# .. math::
#     \Sigma = \begin{bmatrix} \sigma_{1} & \sigma_{4} & \sigma_{5} \\
#     \sigma_{4} & \sigma_{2} & \sigma_{6} \\
#     \sigma_{5} & \sigma_{6} & \sigma_{3} \end{bmatrix}
#
# The relationship is linear when :math:`\sigma_4 = \sigma_5 = \sigma_6 = 0`.
# And the relationship is linear isotropic when :math:`\sigma_1=\sigma_2=\sigma_3`
# and :math:`\sigma_4 = \sigma_5 = \sigma_6 = 0`; e.g.
# :math:`\vec{J} = \sigma \vec{E}`.
# 
# When approximating the inner product according to the finite volume approach,
# the constitutive parameters are defined at cell centers; even if the
# fields/fluxes live at cell edges/faces. As we will see, the inner-product
# matrix is diagonal in the linear case, but contains a significant number
# of non-diagonal entries in the anisotropic case.
# 

# Create a single 3D cell
h = np.ones(1)
mesh = TensorMesh([h, h, h])

# Define 6 constitutive parameters for the cell
sig1, sig2, sig3, sig4, sig5, sig6 = 6, 5, 4, 3, 2, 1

# Linear isotropic case: sig is a (nC, 1) numpy array
sig = sig1*np.ones((1, 1))
Me1 = mesh.getEdgeInnerProduct(sig)  # Edges inner product matrix
Mf1 = mesh.getFaceInnerProduct(sig)  # Faces inner product matrix

# Linear case: sig is a (nC, dim) numpy array
sig = np.c_[sig1, sig2, sig3]
Me2 = mesh.getEdgeInnerProduct(sig)
Mf2 = mesh.getFaceInnerProduct(sig)

# Anisotropic case: sig is a (nC, 3) numpy array for 2D and a (nC, 6) for 3D
sig = np.c_[sig1, sig2, sig3, sig4, sig5, sig6]
Me3 = mesh.getEdgeInnerProduct(sig)
Mf3 = mesh.getFaceInnerProduct(sig)

# Plotting matrix entries
fig = plt.figure(figsize=(14,9))

Ax1 = fig.add_subplot(231)
Ax1.imshow(Mf1.todense())
Ax1.set_title('Linear Isotropic (Faces)')

Ax2 = fig.add_subplot(232)
Ax2.imshow(Mf2.todense())
Ax2.set_title('Linear (Faces)')

Ax3 = fig.add_subplot(233)
Ax3.imshow(Mf3.todense())
Ax3.set_title('Anisotropic (Faces)')

Ax4 = fig.add_subplot(234)
Ax4.imshow(Me1.todense())
Ax4.set_title('Linear Isotropic (Edges)')

Ax5 = fig.add_subplot(235)
Ax5.imshow(Me2.todense())
Ax5.set_title('Linear (Edges)')

Ax6 = fig.add_subplot(236)
Ax6.imshow(Me3.todense())
Ax6.set_title('Anisotropic (Edges)')


#############################################################
# Whole Mesh, the Inverse and Reciprocal Properties
# -------------------------------------------------
# 
# Here we demonstrate general methods to do the following:
# 
#     - Construct the inner-product matrix for the isotropic, linear and anisotropic cases 
#     - Construct the inverse of the inner product matrix using *invMat*
#     - Define the inner-product matrix for the reciprocal of property using *invProp* 
#
# The latter is used when we must take the inner product of a constitutive
# relation of the form:
#
# .. math::
#     \frac{1}{\sigma} \vec{J} = \vec{E}
# 
# The example is done on a 2D mesh for a constitutive relation that does not
# vary spacially. However, this approach is generalized to work in 3D or in
# the even that :math:`\sigma` or :math:`\Sigma` vary spatially.
#

# Create a mesh
h = np.ones(10)
mesh = TensorMesh([h, h])

# Define 3 constitutive parameters for the cell
sig1, sig2, sig3 = 3, 2, 1

# Linear isotropic case: sig is a (nC, 1) numpy array
sig = sig1*np.ones((mesh.nC, 1))
Mf1 = mesh.getFaceInnerProduct(sig)  # Faces inner product matrix
Mf1_inv = mesh.getFaceInnerProduct(sig, invMat=True)  # Inverse
Mf1_recip = mesh.getFaceInnerProduct(sig, invProp=True)  # Faces inner product matrix for 1/sig

# Linear case: sig is a (nC, dim) numpy array
sig = np.kron(np.c_[sig1, sig2], np.ones((mesh.nC, 1)))
Mf2 = mesh.getFaceInnerProduct(sig)
Mf2_inv = mesh.getFaceInnerProduct(sig, invMat=True)  # Inverse
Mf2_recip = mesh.getFaceInnerProduct(sig, invProp=True)  # Faces inner product matrix for 1/sig

# Anisotropic case: sig is a (nC, 3) numpy array for 2D and a (nC, 6) for 3D
sig = np.kron(np.c_[sig1, sig2, sig3], np.ones((mesh.nC, 1)))
Mf3 = mesh.getFaceInnerProduct(sig)
# This inverse cannot be formed explicitly
Mf3_recip = mesh.getFaceInnerProduct(sig, invProp=True)  # Faces inner product matrix for 1/sig

# Plotting matrix entries
fig = plt.figure(figsize=(14,4))

Ax1 = fig.add_subplot(131)
Ax1.spy(Mf1, markersize=1)
Ax1.set_title('Linear Isotropic (Faces)')

Ax2 = fig.add_subplot(132)
Ax2.spy(Mf2, markersize=1)
Ax2.set_title('Linear (Faces)')

Ax3 = fig.add_subplot(133)
Ax3.spy(Mf3, markersize=1)
Ax3.set_title('Anisotropic (Faces)')







































