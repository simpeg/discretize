"""
Constitutive Relations
======================

When solving PDEs using the finite volume approach, inner products may
contain constitutive relations; examples include Ohm's law and Hooke's law.
For this class of inner products, you will learn how to:

    - Construct the inner-product matrix in the case of isotropic, linear and anisotropic constitutive relations
    - Construct the inverse of the inner-product matrix
    - Work with constitutive relations defined by the reciprocal of a parameter

Let :math:`\\vec{J}` and :math:`\\vec{E}` be two physically related
quantities. If their relationship is isotropic (defined by a constant
:math:`\\sigma`), then the constitutive relation is given by:

.. math::
    \\vec{J} = \\sigma \\vec{E}

The inner product between a vector :math:`\\vec{v}` and the right-hand side
of this expression is given by:

.. math::
    (\\vec{v}, \\sigma \\vec{E} ) = \\int_\\Omega \\vec{v} \\cdot \\sigma \\vec{E} \\, dv

Just like in the previous tutorial, we would like to approximate the inner
product numerically using an *inner-product matrix* such that:

.. math::
    (\\vec{v}, \\sigma \\vec{E} ) \\approx \\mathbf{v^T M_\\sigma e}

where the inner product matrix :math:`\\mathbf{M_\\sigma}` now depends on:

    1. the dimensions and discretization of the mesh
    2. where :math:`\\mathbf{v}` and :math:`\\mathbf{e}` live
    3. the spatial distribution of the property :math:`\\sigma`

Constitutive relations may also be defined by a tensor (:math:`\\Sigma`). In
this case, the constitutive relation is of the form:

.. math::
    \\vec{J} = \\Sigma \\vec{E}

where

.. math::
    \\Sigma = \\begin{bmatrix} \\sigma_{1} & \\sigma_{4} & \\sigma_{5} \n
    \\sigma_{4} & \\sigma_{2} & \\sigma_{6} \n
    \\sigma_{5} & \\sigma_{6} & \\sigma_{3} \\end{bmatrix}

Is symmetric and defined by 6 independent parameters. The inner product between
a vector :math:`\\vec{v}` and the right-hand side of this expression is given
by:

.. math::
    (\\vec{v}, \\Sigma \\vec{E} ) = \\int_\\Omega \\vec{v} \\cdot \\Sigma \\vec{E}  \\, dv

Once again we would like to approximate the inner product numerically using an
*inner-product matrix* :math:`\\mathbf{M_\\Sigma}` such that:

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
# Here we compare the inner product matricies for a single cell when the
# constitutive relationship is:
# 
#     - **isotropic:** :math:`\sigma_1 = \sigma_2 = \sigma_3 = \sigma` and :math:`\sigma_4 = \sigma_5 = \sigma_6 = 0`; e.g. :math:`\vec{J} = \sigma \vec{E}`
#     - **linear:** independent parameters :math:`\sigma_1, \sigma_2, \sigma_3` and :math:`\sigma_4 = \sigma_5 = \sigma_6 = 0`
#     - **anisotropic:** independent parameters :math:`\sigma_1, \sigma_2, \sigma_3, \sigma_4, \sigma_5, \sigma_6`
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

# Isotropic case
sig = sig1*np.ones((1, 1))
sig_tensor_1 = np.diag(sig1*np.ones(3))
Me1 = mesh.getEdgeInnerProduct(sig)  # Edges inner product matrix
Mf1 = mesh.getFaceInnerProduct(sig)  # Faces inner product matrix

# Linear case
sig = np.c_[sig1, sig2, sig3]
sig_tensor_2 = np.diag(np.array([sig1, sig2, sig3]))
Me2 = mesh.getEdgeInnerProduct(sig)
Mf2 = mesh.getFaceInnerProduct(sig)

# Anisotropic case
sig = np.c_[sig1, sig2, sig3, sig4, sig5, sig6]
sig_tensor_3 = np.diag(np.array([sig1, sig2, sig3]))
sig_tensor_3[(0, 1), (1, 0)] = sig4
sig_tensor_3[(0, 2), (2, 0)] = sig5
sig_tensor_3[(1, 2), (2, 1)] = sig6
Me3 = mesh.getEdgeInnerProduct(sig)
Mf3 = mesh.getFaceInnerProduct(sig)

# Plotting matrix entries
fig = plt.figure(figsize=(12, 12))

Ax1 = fig.add_subplot(331)
Ax1.imshow(sig_tensor_1)
Ax1.set_title('Property Tensor (isotropic)')

Ax2 = fig.add_subplot(332)
Ax2.imshow(sig_tensor_2)
Ax2.set_title('Property Tensor (linear)')

Ax3 = fig.add_subplot(333)
Ax3.imshow(sig_tensor_3)
Ax3.set_title('Property Tensor (anisotropic)')

Ax4 = fig.add_subplot(334)
Ax4.imshow(Mf1.todense())
Ax4.set_title('M-faces Matrix (isotropic)')

Ax5 = fig.add_subplot(335)
Ax5.imshow(Mf2.todense())
Ax5.set_title('M-faces Matrix (linear)')

Ax6 = fig.add_subplot(336)
Ax6.imshow(Mf3.todense())
Ax6.set_title('M-faces Matrix (anisotropic)')

Ax7 = fig.add_subplot(337)
Ax7.imshow(Me1.todense())
Ax7.set_title('M-edges Matrix (isotropic)')

Ax8 = fig.add_subplot(338)
Ax8.imshow(Me2.todense())
Ax8.set_title('M-edges Matrix (linear)')

Ax9 = fig.add_subplot(339)
Ax9.imshow(Me3.todense())
Ax9.set_title('M-edges Matrix (anisotropic)')


#############################################################
# Spatially Variant Parameters
# ----------------------------
#
# In practice, the parameter :math:`\sigma` or tensor :math:`\Sigma` will
# vary spatially. In this case, we define the parameter
# :math:`\sigma` (or parameters :math:`\Sigma`) for each cell. When
# creating the inner product matrix, we enter these parameters as
# a numpy array. This is demonstrated below. Properties of the resulting
# inner product matricies are discussed.
#

# Create a small 3D mesh
h = np.ones(5)
mesh = TensorMesh([h, h, h])

# Isotropic case: (nC, ) numpy array
sig = np.random.rand(mesh.nC)        # sig for each cell
Me1 = mesh.getEdgeInnerProduct(sig)  # Edges inner product matrix
Mf1 = mesh.getFaceInnerProduct(sig)  # Faces inner product matrix

# Linear case: (nC, dim) numpy array
sig = np.random.rand(mesh.nC, mesh.dim)
Me2 = mesh.getEdgeInnerProduct(sig)
Mf2 = mesh.getFaceInnerProduct(sig)

# Anisotropic case: (nC, 3) for 2D and (nC, 6) for 3D
sig = np.random.rand(mesh.nC, 6)
Me3 = mesh.getEdgeInnerProduct(sig)
Mf3 = mesh.getFaceInnerProduct(sig)

# Properties of inner product matricies
print('\n FACE INNER PRODUCT MATRIX')
print('- Number of faces              :', mesh.nF)
print('- Dimensions of operator       :', str(mesh.nF), 'x', str(mesh.nF))
print('- Number non-zero (isotropic)  :', str(Mf1.nnz))
print('- Number non-zero (linear)     :', str(Mf2.nnz))
print('- Number non-zero (anisotropic):', str(Mf3.nnz), '\n')

print('\n EDGE INNER PRODUCT MATRIX')
print('- Number of faces              :', mesh.nE)
print('- Dimensions of operator       :', str(mesh.nE), 'x', str(mesh.nE))
print('- Number non-zero (isotropic)  :', str(Me1.nnz))
print('- Number non-zero (linear)     :', str(Me2.nnz))
print('- Number non-zero (anisotropic):', str(Me3.nnz), '\n')


#############################################################
# Inverse
# -------
#
# The final discretized system using the finite volume method may contain
# the inverse of the inner-product matrix. Here we show how to call this
# using the *invMat* keyword argument.
#
# For the isotropic and linear cases, the inner product matrix is diagonal.
# As a result, its inverse can be easily formed. For the anisotropic case
# however, we cannot expicitly form the inverse because the inner product
# matrix contains a significant number of off-diagonal elements.
#
# For the isotropic case, we form M_inv and apply it to a vector using
# the :math:`*` operator. For the anisotropic case, we must form the
# inner product matrix and do a numerical solve.
#

# Create a small 3D mesh
h = np.ones(5)
mesh = TensorMesh([h, h, h])

# Isotropic case: (nC, ) numpy array
sig = np.random.rand(mesh.nC)
Me1_inv = mesh.getEdgeInnerProduct(sig, invMat=True)
Mf1_inv = mesh.getFaceInnerProduct(sig, invMat=True)

# Linear case: (nC, dim) numpy array
sig = np.random.rand(mesh.nC, mesh.dim)
Me2_inv = mesh.getEdgeInnerProduct(sig, invMat=True)
Mf2_inv = mesh.getFaceInnerProduct(sig, invMat=True)

# Anisotropic case: (nC, 3) for 2D and (nC, 6) for 3D
sig = np.random.rand(mesh.nC, 6)
Me3 = mesh.getEdgeInnerProduct(sig)
Mf3 = mesh.getFaceInnerProduct(sig)


###########################################################################
# Reciprocal Properties
# ---------------------
#
# At times, the constitutive relation may be defined by the reciprocal of
# a parameter (:math:`\rho`). Here we demonstrate how inner product matricies
# can be formed using the keyword argument *invProp*. We will do this for a
# single cell and plot the matrix elements. We can easily extend this to
# a mesh comprised of many cells.
#
# In this case, the constitutive relation is given by:
#
# .. math::
#     \vec{J} = \frac{1}{\rho} \vec{E}
#
# The inner product between a vector :math:`\\vec{v}` and the right-hand side
# of the expression is given by:
#
# .. math::
#     (\vec{v}, \rho^{-1} \vec{E} ) = \int_\Omega \vec{v} \cdot \rho^{-1} \vec{E} \, dv
#
# where the inner product is approximated using an inner product matrix
# :math:`\mathbf{M_\rho}` as follows:
#
# .. math::
#     (\vec{v}, \rho^{-1} \vec{E} ) \approx \mathbf{v^T M_\rho e}
#
# In the case that the constitutive relation is defined by the inverse of a
# tensor :math:`\Gamma`, e.g.:
#
# .. math::
#     \vec{J} = \Gamma^{-1} \vec{E}
#
# where
#
# .. math::
#     \Gamma = \begin{bmatrix} \rho_{1} & \rho_{4} & \rho_{5} \\
#     \rho_{4} & \rho_{2} & \rho_{6} \\
#     \rho_{5} & \rho_{6} & \rho_{3} \end{bmatrix}
#
# The inner product between a vector :math:`\vec{v}` and the right-hand side of
# this expression is given by:
#
# .. math::
#     (\vec{v}, \Gamma^{-1} \vec{E} ) = \int_\Omega \vec{v} \cdot \Gamma^{-1} \vec{E}  \\, dv
#
# Once again we would like to approximate the inner product numerically using an
# *inner-product matrix* :math:`\\mathbf{M_\Gamma}` such that:
#
# .. math::
#     (\vec{v}, \Gamma{-1} \vec{E} ) \approx \mathbf{v^T M_\Gamma e}
#
# Here we demonstrate how to form the inner-product matricies
# :math:`\mathbf{M_\rho}` and :math:`\mathbf{M_\Gamma}`.
#

# Create a small 3D mesh
h = np.ones(1)
mesh = TensorMesh([h, h, h])

# Define 6 constitutive parameters for the cell
rho1, rho2, rho3, rho4, rho5, rho6 = 1./6., 1./5., 1./4., 1./3., 1./2., 1

# Isotropic case
rho = rho1*np.ones((1, 1))
Me1 = mesh.getEdgeInnerProduct(rho, invProp=True)  # Edges inner product matrix
Mf1 = mesh.getFaceInnerProduct(rho, invProp=True)  # Faces inner product matrix

# Linear case
rho = np.c_[rho1, rho2, rho3]
Me2 = mesh.getEdgeInnerProduct(rho, invProp=True)
Mf2 = mesh.getFaceInnerProduct(rho, invProp=True)

# Anisotropic case
rho = np.c_[rho1, rho2, rho3, rho4, rho5, rho6]
Me3 = mesh.getEdgeInnerProduct(rho, invProp=True)
Mf3 = mesh.getFaceInnerProduct(rho, invProp=True)

# Plotting matrix entries
fig = plt.figure(figsize=(14, 9))

Ax1 = fig.add_subplot(231)
Ax1.imshow(Mf1.todense())
Ax1.set_title('Isotropic (Faces)')

Ax2 = fig.add_subplot(232)
Ax2.imshow(Mf2.todense())
Ax2.set_title('Linear (Faces)')

Ax3 = fig.add_subplot(233)
Ax3.imshow(Mf3.todense())
Ax3.set_title('Anisotropic (Faces)')

Ax4 = fig.add_subplot(234)
Ax4.imshow(Me1.todense())
Ax4.set_title('Isotropic (Edges)')

Ax5 = fig.add_subplot(235)
Ax5.imshow(Me2.todense())
Ax5.set_title('Linear (Edges)')

Ax6 = fig.add_subplot(236)
Ax6.imshow(Me3.todense())
Ax6.set_title('Anisotropic (Edges)')
