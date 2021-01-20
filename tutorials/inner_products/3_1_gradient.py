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
# The first option is to define :math:`\boldsymbol{\phi}` on the nodes
# and :math:`\boldsymbol{u}` on cell edges:
# 
# .. math::
#     \int_\Omega \vec{u} \cdot \nabla \phi \, dv \approx \boldsymbol{u^T M_e G \, \phi}
# 
# The second option is to define :math:`\boldsymbol{\phi}` at cell centers
# and :math:`\boldsymbol{u}` on cell faces. In this case, 
# we use the identity :math:`\vec{u} \cdot \nabla \phi = \nabla \cdot \phi\vec{u} - \phi \nabla \cdot \vec{u}`
# and apply the divergence theorem such that:
# 
# .. math::
#     \int_\Omega \vec{u} \cdot \nabla \phi \, dv = - \int_\Omega \phi \nabla \cdot \vec{u} \, dv + \oint_{\partial \Omega} \phi \hat{n} \cdot \vec{u} \, da \approx - \boldsymbol{u^T D^T M_c \, \phi} + B.\! C.
# 
# where
# 
#     - :math:`\boldsymbol{G}` is a :ref:`discrete gradient operator <operators_differential_gradient>`
#     - :math:`\boldsymbol{D}` is a :ref:`discrete divergence operator <operators_differential_divergence>`
#     - :math:`\boldsymbol{M_c}` is the :ref:`basic inner product matrix for vectors at cell centers <inner_products_basic>`
#     - :math:`\boldsymbol{M_e}` is the :ref:`basic inner product matrix for vectors at edges <inner_products_basic>`
#     - :math:`B.\! C.` represents an additional term that must be constructed to impose boundary conditions correctly on :math:`\phi`. It is zero when :math:`\phi = 0` on the boundary.
# 
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


