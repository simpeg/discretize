# -*- coding: utf-8 -*-
"""
Differential Operators
======================

When solving PDEs using the finite volume approach, inner products may
contain differential operators. Where :math:`\\psi` and :math:`\\phi` are
scalar quantities, and :math:`\\vec{u}` and :math:`\\vec{v}` are vector
quantities, we may need to derive a discrete approximation for the following
inner products:

    1. :math:`(\\vec{u} , \\nabla \\phi)`
    2. :math:`(\\psi , \\nabla \\cdot \\vec{v})`
    3. :math:`(\\vec{u} , \\nabla \\times \\vec{v})`
    4. :math:`(\\psi, \\Delta^2 \\phi)`

In this section, we demonstrate how to go from the inner product to the
discrete approximation for each case. In doing so, we must construct
discrete differential operators, inner product matricies and consider
boundary conditions.
    


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
import numpy as np
import matplotlib.pyplot as plt


#####################################################
# Gradient
# --------
#
# Where :math:`\phi` is a scalar quantity and :math:`\vec{u}` is a vector
# quantity, we would like to evaluate the following inner product:
#
# .. math::
#     (\vec{u} , \nabla \phi) = \int_\Omega \vec{u} \cdot \nabla \phi \, dv
#
# **Inner Product at edges:**
#
# In the case that :math:`\vec{u}` represents a field, it is natural for it to
# be discretized to live on cell edges. By defining :math:`\phi` to live at
# the nodes, we can use the nodal gradient operator (:math:`\mathbf{G_n}`) to
# map from nodes to edges. The inner product is therefore computed using an
# inner product matrix (:math:`\mathbf{M_e}`) for
# quantities living on cell edges, e.g.:
#
# .. math::
#     (\vec{u} , \nabla \phi) \approx \mathbf{u^T M_e G_n \phi}
#
# **Inner Product at faces:**
#
# In the case that :math:`\vec{u}` represents a flux, it is natural for it to
# be discretized to live on cell faces. By defining :math:`\phi` to live at
# cell centers, we can use the cell gradient operator (:math:`\mathbf{G_c}`) to
# map from centers to faces. In this case, we must impose boundary conditions
# on the discrete gradient operator because it cannot use locations outside
# the mesh to evaluate the gradient on the boundary. If done correctly, the
# inner product is computed using an inner product matrix (:math:`\mathbf{M_f}`)
# for quantities living on cell faces, e.g.:
#
# .. math::
#    (\vec{u} , \nabla \phi) \approx \mathbf{u^T M_f G_c \phi}
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

#####################################################
# Divergence
# ----------
#
# Where :math:`\psi` is a scalar quantity and :math:`\vec{v}` is a vector
# quantity, we would like to evaluate the following inner product:
#
# .. math::
#     (\psi , \nabla \cdot \vec{v}) = \int_\Omega \psi \nabla \cdot \vec{v} \, dv
#
# The divergence defines a measure of the flux leaving/entering a volume. As a
# result, it is natural for :math:`\vec{v}` to be a flux defined on cell faces.
# The face divergence operator (:math:`\mathbf{D}`) maps from cell faces to
# cell centers, therefore # we should define :math:`\psi` at cell centers. The
# inner product is ultimately computed using an inner product matrix
# (:math:`\mathbf{M_f}`) for quantities living on cell faces, e.g.:
#
# .. math::
#    (\psi , \nabla \cdot \vec{v}) \approx \mathbf{\psi^T} \textrm{diag} (\mathbf{vol} ) \mathbf{D v}
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

#####################################################
# Curl
# ----
#
# Where :math:`\vec{u}` and :math:`\vec{v}` are vector quantities, we would
# like to evaluate the following inner product:
#
# .. math::
#     (\vec{u} , \nabla \times \vec{v}) = \int_\Omega \vec{u} \nabla \times \vec{v} \, dv
#
# **Inner Product at Faces:**
#
# Let :math:`\vec{u}` denote a flux and let :math:`\vec{v}` denote a field.
# In this case, it is natural for the flux :math:`\vec{u}` to live on cell
# faces and for the field :math:`\vec{v}` to live on cell edges. The discrete
# curl operator (:math:`\mathbf{C_e}`) in this case naturally maps from cell
# edges to cell faces without the need to define boundary conditions. The
# inner product can be approxiated using an inner product matrix
# (:math:`\mathbf{M_f}`) for quantities living on cell faces, e.g.:
#
# .. math::
#     (\vec{u} , \nabla \times \vec{v}) \approx \mathbf{u^T M_f C_e v}
#
# **Inner Product at Edges:**
#
# Now let :math:`\vec{u}` denote a field and let :math:`\vec{v}` denote a flux.
# Now it is natural for the :math:`\vec{u}` to live on cell edges
# and for :math:`\vec{v}` to live on cell faces. We would like to compute the
# inner product using an inner product matrix (:math:`\mathbf{M_e}`) for
# quantities living on cell edges. However, this requires a discrete curl
# operator (:math:`\mathbf{C_f}`) that maps from cell faces
# to cell edges; which requires to impose boundary conditions on the operator.
# If done successfully:
#
# .. math::
#     (\vec{u} , \nabla \times \vec{v}) \approx \mathbf{u^T M_e C_f v}
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


###########################################################
# Scalar Laplacian
# ----------------
#
# Where :math:`\psi` and :math:`\phi` are scalar quantities, and the scalar
# Laplacian :math:`\Delta^2 = \nabla \cdot \nabla`, we would like to
# approximate the following inner product:
#
# .. math::
#     (\psi , \nabla \cdot \nabla \phi) = \int_\Omega \psi (\nabla \cdot \nabla \phi) \, dv
#
# Using :math:`p \nabla \cdot \mathbf{q} = \nabla \cdot (p \mathbf{q}) - \mathbf{q} \cdot (\nabla p )`
# and the Divergence theorem we obtain:
#
# .. math::
#     \int_{\partial \Omega} \mathbf{n} \cdot ( \psi \nabla \phi ) \, da
#     - \int_\Omega (\nabla \psi ) \cdot (\nabla \phi ) \, dv
#
# In this case, the surface integral can be eliminated if we can assume a
# Neumann condition of :math:`\partial \phi/\partial n = 0` on the boundary.
#
# **Inner Prodcut at Edges:**
#
# Let :math:`\psi` and :math:`\phi` be discretized to the nodes. In this case,
# the discrete gradient operator (:math:`\mathbf{G_n}`) must map from nodes
# to edges. Ultimately we evaluate the inner product using an inner product
# matrix (:math:`\mathbf{M_e}` for quantities living on cell edges, e.g.:
#
# .. math::
#     (\psi , \nabla \cdot \nabla \phi) \approx \mathbf{\psi G_n^T M_e G_n \phi}
#
# **Inner Product at Faces:**
#
# Let :math:`\psi` and :math:`\phi` be discretized to cell centers. In this
# case, the discrete gradient operator (:math:`\mathbf{G_c}`) must map from
# centers to faces; and requires the user to set Neumann conditions in the
# operator. Ultimately we evaluate the inner product using an inner product
# matrix (:math:`\mathbf{M_f}`) for quantities living on cell faces, e.g.:
#
# .. math::
#     (\psi , \nabla \cdot \nabla \phi) \approx \mathbf{\psi G_c^T M_f G_c \phi}
#
#

# Make basic mesh
h = np.ones(10)
mesh = TensorMesh([h, h, h])

# Items required to perform psi.T*(Gn.T*Me*Gn*phi)
Me = mesh.getEdgeInnerProduct()  # Basic inner product matrix (edges)
Gn = mesh.nodalGrad  # Nodes to edges gradient

# Items required to perform psi.T*(Gc.T*Mf*Gc*phi)
Mf = mesh.getFaceInnerProduct()  # Basic inner product matrix (faces)
mesh.setCellGradBC(["dirichlet", "dirichlet", "dirichlet"])
Gc = mesh.cellGrad  # Centers to faces gradient

# Plot Sparse Representation
fig = plt.figure(figsize=(9, 4))

ax1 = fig.add_subplot(121)
ax1.spy(Gn.T * Me * Gn, markersize=0.5)
ax1.set_title("Gn.T*Me*Gn", pad=5)

ax2 = fig.add_subplot(122)
ax2.spy(Gc.T * Mf * Gc, markersize=0.5)
ax2.set_title("Gc.T*Mf*Gc", pad=5)
