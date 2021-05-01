r"""
2D Poisson Equation with Zero Neumann Condition (Nodal Formulation)
===================================================================

Here we use the *discretize* package to approximate the solution to a 2D
Poisson equation with zero Neumann boundary conditions.
For our tutorial, we consider the free-space electrostatic problem
for 2 electric point charges of opposite sign.
This tutorial focusses on:

    - approximating basic types of inner products
    - discretization which imposes the boundary conditions naturally
    - basic disretization of point sources

"""

####################################################
# 1. Formulating the Problem
# --------------------------
# 
# For this tutorial, we would like to compute the electric potential (:math:`\phi`)
# and electric fields (:math:`\mathbf{e}`) in 2D that result from
# a distribution of point charges in a vacuum. Given the electric permittiviy
# is uniform within the domain and equal to the permittivity of free space, the physics
# are defined by a 2D Poisson equation:
#
# .. math::
#     \nabla^2 \phi = - \frac{\rho}{\epsilon_0}
# 
# where :math:`\rho` is the charge density and :math:`\epsilon_0` is
# the permittivity of free space. A more practical
# formulation for applying the finite volume method is to 
# define the problem as two fist-order differential equations.
# Starting with Gauss's law and Faraday's law in the case of
# electrostatics (:cite:`griffiths1999`):
#     
# .. math::
#     &\nabla \cdot \vec{e} = \frac{\rho}{\epsilon_0} \\
#     &\nabla \times \vec{e} = \boldsymbol{0} \;\;\; \Rightarrow \;\;\; \vec{e} = -\nabla \phi \\
#     &\textrm{s.t.} \;\;\; \frac{\partial \phi}{\partial n} \, \Big |_{\partial \Omega} = - \hat{n} \cdot \vec{e} \, \Big |_{\partial \Omega} = 0
# 
# where the Neumann boundary condition on :math:`\phi` implies no electric flux leaves the system.
# For 2 point charges of equal and opposite sign, the charge density is given by:
# 
# .. math::
#     \rho = \rho_0 \big [ \delta ( \boldsymbol{r_+}) - \delta (\boldsymbol{r_-} ) \big ]
#     :label: tutorial_eq_1_1_2
#
# where :math:`\rho_0` is a constant.
# We chose this charge distribution so that we would not violate the boundary conditions.
# Since the total electric charge is zero, the net flux leaving the system is zero.
#

####################################################
# 2. Taking Inner Products
# ------------------------
# 
# To solve this problem numerically, we take the inner product
# of each differential equation with an appropriate test function.
# Where :math:`\psi` is a scalar test function and :math:`\vec{u}` is a
# vector test function:
# 
# .. math::
#     \int_\Omega \psi (\nabla \cdot \vec{e}) \, dv = \frac{1}{\epsilon_0} \int_\Omega \psi \rho \, dv
#     :label: tutorial_eq_1_1_3
# 
# and
# 
# .. math::
#     \int_\Omega \vec{u} \cdot \vec{e} \, dv = - \int_\Omega \vec{u} \cdot (\nabla \phi ) \, dv
#     :label: tutorial_eq_1_1_4
# 
# In the case of Gauss' law, we have a volume integral containing the Dirac delta function.
# Thus expression :eq:`tutorial_eq_1_1_3` becomes:
# 
# .. math::
#     \int_\Omega \psi (\nabla \cdot \vec{e}) \, dv = \frac{1}{\epsilon_0} \psi \, q
#     :label: tutorial_eq_1_1_5
# 
# where :math:`q` represents an integrated charge density.
# 

####################################################
# 3. Discretizing the Inner Products
# ----------------------------------
# 
# Here, we let :math:`\boldsymbol{\phi}` be the discrete representation
# of the electic potential :math:`\phi` on the nodes. Since :math:`\vec{e} = -\nabla \phi`,
# it is natural for the discrete representation of the electric fields :math:`\boldsymbol{e}`
# to live on the edges; allowing the discrete gradient operator to map from nodes to edges.
# 
# In tutorials for inner products, we showed how to approximate most classes
# of inner products. Since the electric field is discretized at edges, so much the
# discrete representation of the test function :math:`\boldsymbol{u}`.
# Approximating the inner products in expression :eq:`tutorial_eq_1_1_4`
# according to the finite volume method, we obtain:
# 
# .. math::
#     \boldsymbol{u^T M_e \, e} = - \boldsymbol{u^T M_e G \, \phi}
#     :label: tutorial_eq_1_1_6
# 
# where
# 
#     - :math:`\boldsymbol{M_e}` is the inner product matrix at edges
#     - :math:`\boldsymbol{G}` is the discrete gradient operator from nodes to edges
# 
# Now we approximate the inner products in expression :eq:`tutorial_eq_1_1_5`.
# Since :math:`\boldsymbol{e}` lives on the edges, it would be natural for the
# discrete divergence operator to map from edges to nodes. Unfortunately for nodes
# at the boundary, this would require knowledge of the electric field at edges
# outside our domain.
#
# For the left-hand side of expression :eq:`tutorial_eq_1_1_5`, we must use the
# identity :math:`\psi \nabla \cdot \vec{e} = \nabla \cdot \psi\vec{e} - \vec{e} \cdot \nabla \psi`
# and apply the divergence theorem such that expression :eq:`tutorial_eq_1_1_5` becomes:
# 
# .. math::
#     - \int_\Omega \vec{e} \cdot \nabla \psi \, dv + \oint_{\partial \Omega} \psi (\hat{n} \cdot \vec{e}) \, da = \frac{1}{\epsilon_0} \psi \, q
#     :label: tutorial_eq_1_1_7
# 
# Since :math:`\hat{n} \cdot \vec{e}` is zero on the boundary, the surface integral is equal to zero.
# We have chosen a discretization where boundary conditions are imposed naturally!
# 
# Test function :math:`\psi` and the integrated charge density :math:`q` are defined such that their
# discrete representations :math:`\boldsymbol{\psi}` and :math:`\boldsymbol{q}` must live on the nodes.
# The discrete approximation to expression :eq:`tutorial_eq_1_1_7` is given by:
# 
# .. math::
#     - \boldsymbol{\psi^T G^T M_e \, e} = \frac{1}{} \boldsymbol{\psi^T q}
#     :label: tutorial_eq_1_1_8
# 
# The easiest way to discretize the source is to let :math:`\boldsymbol{q_i}=\rho_0` at the nearest node to the positive charge and
# let :math:`\boldsymbol{q_i}=-\rho_0` at the nearest node to the negative charge.
# The value is zero for all other nodes.
# 

####################################################
# 4. Solvable Linear System
# -------------------------
# 
# By combining the discrete representations from expressions
# :eq:`tutorial_eq_1_1_6` and :eq:`tutorial_eq_1_1_8` and factoring like-terms
# we obtain:
# 
# .. math::
#     \boldsymbol{G^T M_e G \, \phi} = \frac{1}{\epsilon_0} \boldsymbol{q}
#     :label: tutorial_eq_1_1_9
# 
# Let :math:`\boldsymbol{A} = \boldsymbol{G^T M_e G}`.
# The linear system :math:`\boldsymbol{A}` has a single null vector and is not invertible.
# To remedy this, we define a reference potential on the boundary
# by setting :math:`A_{0,0} = 1` and by setting all other values in that row to 0.
# We can now solve for the electric potential on the nodes.
#
# Once the electric potential at nodes has been computed, the electric field on
# the edges can be computed using expression :eq:`tutorial_eq_1_1_6`:
# 
# .. math::
#     \boldsymbol{e} = - \boldsymbol{G \, \phi}
# 

###############################################
# 5. Implement Discretize
# -----------------------

#%%
# Import the necessary packages for the tutorial.

from discretize import TensorMesh
from pymatsolver import SolverLU
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from discretize.utils import sdiag

mpl.rcParams.update({'font.size':14})

#%%
# Construct a mesh.

h = np.ones(100)
mesh = TensorMesh([h, h], "CC")

#%%
# Construct the required discrete operators and inner product matrices.

G = mesh.nodal_gradient               # gradient operator (nodes to edges)
Me = mesh.get_edge_inner_product()    # edge inner product matrix

#%%
# Form the linear system and remove the null-space.

A = G.T * Me * G
A[0,0] = 1.
A[0, 1:] = 0

#%%
# Construct the right-hand side.

xyn = mesh.nodes
kneg = (xyn[:, 0] == -10) & (xyn[:, 1] == 0)   # -ve charge at (-10, 0)
kpos = (xyn[:, 0] == 10) & (xyn[:, 1] == 0)    # +ve charge at (10, 0)

rho = np.zeros(mesh.n_nodes)
rho[kneg] = -1
rho[kpos] = 1

#%%
# Solve for the electric potential on the nodes and the electric field
# on the edges.

AinvM = SolverLU(A)  # Define the inverse of A using direct solver
phi = AinvM * rho    # Compute electric potential on nodes
E = - G * phi        # Compute electric field on edges

#%%
# Plot the source term, electric potential and electric field.

fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(131)
mesh.plotImage(rho, v_type="N", ax=ax1)
ax1.set_title("Charge Density")

ax2 = fig.add_subplot(132)
mesh.plotImage(phi, v_type="N", ax=ax2)
ax2.set_title("Electric Potential")

ax3 = fig.add_subplot(133)
mesh.plotImage(
    E, ax=ax3, v_type="E", view="vec", stream_opts={"color": "w", "density": 1.0}
)
ax3.set_title("Electric Fields")

plt.tight_layout()
