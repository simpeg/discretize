r"""
Gradient Operator
=================

When solving PDEs using the finite volume approach, inner products may
contain the gradient operator. Where :math:`\phi` is a scalar quantity
and :math:`\vec{u}` is a vector quantity, we may need to derive a
discrete approximation to the following inner product:

.. math::
    (\vec{u} , \nabla \phi) = \int_\Omega \, \vec{u} \cdot \nabla \phi \, dv

In this section, we demonstrate how to go from the inner product to the
discrete approximation. In doing so, we must construct
discrete differential operators, inner product matricies and consider
boundary conditions.


"""

####################################################
# Background Theory
# -----------------
#
# For the inner product between a vector :math:`\vec{u}` and
# the gradient of a scalar :math:`\phi`,
# there are two options for where the variables should live.
# 
# **For** :math:`\boldsymbol{\phi}` **on the nodes and** :math:`\boldsymbol{u}` **on cell edges:**
# 
# .. math::
#     \int_\Omega \vec{u} \cdot \nabla \phi \, dv \approx \boldsymbol{u^T M_e G \, \phi}
# 
# where
# 
#     - :math:`\boldsymbol{M_e}` is the basic inner product matrix for vectors at edges
#     - :math:`\boldsymbol{G}` is the discrete gradient operator which maps from nodes to edges
# 
# **For** :math:`\boldsymbol{\phi}` **at cell centers and** :math:`\boldsymbol{u}` **on cell faces**,
# the gradient operator would have to map from cell centers to faces. This would require knowledge
# of :math:`\phi` outside the domain for boundary faces. In this case, we use the identity
# :math:`\vec{u} \cdot \nabla \phi = \nabla \cdot \phi\vec{u} - \phi \nabla \cdot \vec{u}`
# and apply the divergence theorem such that:
# 
# .. math::
#     \int_\Omega \vec{u} \cdot \nabla \phi \, dv =
#     - \int_\Omega \phi \nabla \cdot \vec{u} \, dv + \oint_{\partial \Omega} \phi \hat{n} \cdot \vec{u} \, da
#     \approx - \boldsymbol{u^T D^T M_c \, \phi} + \boldsymbol{u^T B \, \phi}
#     = \boldsymbol{u^T \tilde{G} \, \phi}
# 
# where
# 
#     - :math:`\boldsymbol{D}` is the discrete divergence operator from faces to cell centers
#     - :math:`\boldsymbol{M_c}` is the basic inner product matrix for scalars at cell centers
#     - :math:`\boldsymbol{B}` is a sparse matrix that imposes the boundary conditions on :math:`\phi`
#     - :math:`\boldsymbol{\tilde{G}} = \boldsymbol{-D^T M_c + B}` acts as a modified gradient operator with boundary conditions imposed
# 
# Note that when :math:`\phi = 0` on the boundary, the term containing :math:`\boldsymbol{B}` is zero.
# 


####################################################
# Import Packages
# ---------------
#

from discretize.utils import sdiag
from discretize import TensorMesh
import numpy as np
import matplotlib.pyplot as plt


#####################################################
# Gradient
# --------
#

# Make basic mesh
h = np.ones(10)
mesh = TensorMesh([h, h, h])

# Items required to perform u.T*(Me*Gn*phi)
Me = mesh.getEdgeInnerProduct()  # Basic inner product matrix (edges)
Gn = mesh.nodalGrad  # Nodes to edges gradient

# Items required to perform u.T*(Mf*Gc*phi)
Mf = mesh.getFaceInnerProduct()  # Basic inner product matrix (faces)
mesh.setCellGradBC(["neumann", "dirichlet", "neumann"])  # Set boundary conditions
Gc = mesh.cellGrad  # Cells to faces gradient

# Plot Sparse Representation
fig = plt.figure(figsize=(5, 6))

ax1 = fig.add_subplot(121)
ax1.spy(Me * Gn, markersize=0.5)
ax1.set_title("Me*Gn")

ax2 = fig.add_subplot(122)
ax2.spy(Mf * Gc, markersize=0.5)
ax2.set_title("Mf*Gc")


