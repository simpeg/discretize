r"""
Divergence Operator
===================

When solving PDEs using the finite volume approach, inner products may
contain the divergence operator. Where :math:`\psi` is a scalar quantity
and :math:`\vec{w}` is a vector quantity, we may need to derive a
discrete approximation to the following inner product:

.. math::
    (\psi , \nabla \cdot \vec{w}) = \int_\Omega \, \psi \, \nabla \cdot \vec{w} \, dv

In this section, we demonstrate how to go from the inner product to the
discrete approximation. In doing so, we must construct
discrete differential operators, inner product matricies and consider
boundary conditions.


"""

####################################################
# Import Packages
# ---------------
#

from discretize.utils import sdiag
from discretize import TensorMesh
import numpy as np
import matplotlib.pyplot as plt

####################################################
# Background Theory
# -----------------
#
# For the inner product between a scalar (:math:`\psi`) and
# the divergence of a vector (:math:`\vec{w}`),
# there are two options for where the variables should live.
# The first option is to define :math:`\boldsymbol{\psi}` at cell centers
# and define :math:`\boldsymbol{w}` on cell faces:
# 
# .. math::
#     \int_\Omega \psi \; (\nabla \cdot \vec{w}) \, dv \approx \boldsymbol{\psi^T M_c D \, w}
# 
# The second option is to define :math:`\boldsymbol{\psi}` on
# the nodes and to define :math:`\boldsymbol{w}` defined on cell edges.
# To evaluate the inner product we use the identity
# :math:`\psi \nabla \cdot \vec{w} = \nabla \cdot \psi\vec{w} - \vec{w} \cdot \nabla \psi`
# and apply the divergence theorem such that:
# 
# .. math::
#     \int_\Omega \psi \; (\nabla \cdot \vec{w}) \, dv = - \int_\Omega \vec{w} \cdot \nabla \psi \, dv + \oint_{\partial \Omega} \psi (\hat{n} \cdot \vec{w}) \, da \approx - \boldsymbol{\psi^T G^T M_e \, w } + B.\! C.
# 
# where
# 
#     - :math:`\boldsymbol{G}` is a :ref:`discrete gradient operator <operators_differential_gradient>`
#     - :math:`\boldsymbol{D}` is a :ref:`discrete divergence operator <operators_differential_divergence>`
#     - :math:`\boldsymbol{M_c}` is the :ref:`basic inner product matrix for vectors at cell centers <inner_products_basic>`
#     - :math:`\boldsymbol{M_e}` is the :ref:`basic inner product matrix for vectors at edges <inner_products_basic>`
#     - :math:`B.\! C.` represents an additional term that must be constructed to impose boundary conditions correctly on :math:`\vec{w}`. It is zero when :math:`\hat{n} \cdot \vec{w} = 0` on the boundary.
# 
# 


#####################################################
# Divergence
# ----------
#

# Make basic mesh
h = np.ones(10)
mesh = TensorMesh([h, h, h])

# Items required to perform psi.T*(Mc*D*v)
Mc = sdiag(mesh.vol)  # Basic inner product matrix (centers)
D = mesh.faceDiv  # Faces to centers divergence

# Plot sparse representation
fig = plt.figure(figsize=(8, 5))

ax1 = fig.add_subplot(111)
ax1.spy(Mc * D, markersize=0.5)
ax1.set_title("Mc*D", pad=20)
