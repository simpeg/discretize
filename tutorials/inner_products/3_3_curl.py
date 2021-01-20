r"""
Curl Operator
=============

When solving PDEs using the finite volume approach, inner products may
contain the curl operator. Where :math:`\vec{u}` and :math:`\vec{w}` are vector
quantities, we may need to derive a discrete approximation for the following
inner product:

.. math::
    (\vec{u} , \nabla \times \vec{w}) = \int_\Omega \, \vec{u} \cdot \vec{w} \, dv

In this section, we demonstrate how to go from the inner product to the
discrete approximation. In doing so, we must construct
discrete differential operators, inner product matricies and consider
boundary conditions.


"""

####################################################
# Background Theory
# -----------------
#
# For the inner product between a vector (:math:`\vec{u}`) and the
# curl of another vector (:math:`\vec{w}`),
# there are two options for where the variables should live. The first
# option is to define :math:`\boldsymbol{u}` on the faces
# and to define :math:`\boldsymbol{w}` on cell edges:
# 
# .. math::
#     \int_\Omega \vec{u} \cdot (\nabla \times \vec{w} ) \, dv \approx \boldsymbol{u^T M_f C \, w}
# 
# The second option is to define :math:`\boldsymbol{u}` on the edges
# and to define :math:`\boldsymbol{w}` on cell faces.
# In this case, we use the identity
# :math:`\vec{u} \cdot (\nabla \times \vec{w}) = \vec{w} \cdot (\nabla \times \vec{u}) - \nabla \cdot (\vec{u} \times \vec{w})`
# and apply the divergence theorem such that:
# 
# .. math::
#     \int_\Omega \vec{u} \cdot (\nabla \times \vec{w} ) \, dv = \int_\Omega \vec{w} \cdot (\nabla \times \vec{u} ) \, dv - \oint_{\partial \Omega} (\vec{u} \times \vec{w}) \cdot d\vec{a} \approx \boldsymbol{u^T C^T \! M_f \, w } + B.\! C.
# 
# where
# 
#     - :math:`\boldsymbol{C}` is a :ref:`discrete curl operator <operators_differential_curl>`
#     - :math:`\boldsymbol{M_f}` is the :ref:`basic inner product matrix for vectors on cell faces <inner_products_basic>`
#     - :math:`B.\! C.` represents an additional term that must be constructed to impose boundary conditions correctly on :math:`\vec{w}`. It is zero when :math:`\hat{n} \times \vec{w} = 0` on the boundary.
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
# Curl
# ----
#


# Make basic mesh
h = np.ones(10)
mesh = TensorMesh([h, h, h])

# Items required to perform u.T*(Mf*Ce*v)
Mf = mesh.getFaceInnerProduct()  # Basic inner product matrix (faces)
Ce = mesh.edgeCurl  # Edges to faces curl

# Items required to perform u.T*(Me*Cf*v)
Me = mesh.getEdgeInnerProduct()  # Basic inner product matrix (edges)
Cf = mesh.edgeCurl.T  # Faces to edges curl (assumes Dirichlet)

# Plot Sparse Representation
fig = plt.figure(figsize=(9, 5))

ax1 = fig.add_subplot(121)
ax1.spy(Mf * Ce, markersize=0.5)
ax1.set_title("Mf*Ce", pad=10)

ax2 = fig.add_subplot(122)
ax2.spy(Me * Cf, markersize=0.5)
ax2.set_title("Me*Cf", pad=10)

