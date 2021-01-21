r"""
2D Poisson Equation with Zero Neumann Condition (Cell Centered Formulation)
===========================================================================

Here we use the *discretize* package to approximate the solution to a 2D
Poisson equation with zero Neumann boundary conditions.
For our tutorial, we consider the free-space electrostatic problem
for 2 electric point charges of opposite sign.
This tutorial focusses on:

    - approximating basic types of inner products
    - imposing the boundary condition by approximating a surface integral
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
#     :label: tutorial_eq_1_2_1
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
#     :label: tutorial_eq_1_2_3
# 
# and
# 
# .. math::
#     \int_\Omega \vec{u} \cdot \vec{e} \, dv = - \int_\Omega \vec{u} \cdot (\nabla \phi ) \, dv
#     :label: tutorial_eq_1_2_4
# 
# In the case of Gauss' law, we have a volume integral containing the Dirac delta function.
# Thus expression :eq:`tutorial_eq_1_2_3` becomes:
# 
# .. math::
#     \int_\Omega \psi (\nabla \cdot \vec{e}) \, dv = \frac{1}{\epsilon_0} \psi \, q
#     :label: tutorial_eq_1_2_5
# 
# where :math:`q` represents an integrated charge density.
# 

####################################################
# 3. Discretizing the Inner Products
# ----------------------------------
# 
# Here, we let :math:`\boldsymbol{\phi}` be the discrete representation
# of the electic potential :math:`\phi` at cell centers. Since :math:`\vec{e} = -\nabla \phi`,
# it is natural for discrete representation of the electric field :math:`\boldsymbol{e}`
# to live on the faces; implying the discrete divergence operator maps from faces to cell centers.
# 
# In tutorials for inner products, we showed how to approximate most classes
# of inner products. Since the numerical divergence of :math:`\boldsymbol{e}` is
# mapped to cell centers, the discrete representation of the test function :math:`\boldsymbol{\psi}`
# and the integrated charge density :math:`\boldsymbol{q}` must live at cell centers.
#
# Approximating the inner products in expression :eq:`tutorial_eq_1_2_5`
# according to the finite volume method, we obtain:
# 
# .. math::
#     \boldsymbol{\psi^T M_c D e} = \frac{1}{\epsilon_0} \boldsymbol{\psi^T q}
#     :label: tutorial_eq_1_2_10
# 
# where
# 
#     - :math:`\boldsymbol{M_c}` is the inner product matrix at cell centers,
#     - :math:`\boldsymbol{D}` is the discrete divergence operator
#     - :math:`\boldsymbol{q}` is a discrete representation for the integrated charge density for each cell
# 
# The easiest way to discretize the source is to let :math:`\boldsymbol{q_i}=\rho_0` at the nearest
# cell center to the positive charge and to let :math:`\boldsymbol{q_i}=-\rho_0` at
# the nearest cell center to the negative charge. The value is zero for all other cells.
# 
# Now we approximate the inner product in expression :eq:`tutorial_eq_1_2_4`.
# Since :math:`\boldsymbol{\phi}` lives at cell centers, it would be natural for the
# discrete gradient operator to map from centers to faces. Unfortunately for 
# boundary faces, this would require knowledge of the electric potential at cell
# centers outside our domain.
# For the right-hand side, we must use the identity
# :math:`\vec{u} \cdot \nabla \phi = \nabla \cdot \phi\vec{u} - \phi \nabla \cdot \vec{u}`
# and apply the divergence theorem such that expression :eq:`tutorial_eq_1_2_4` becomes:
# 
# .. math::
#     \int_\Omega \vec{u} \cdot \vec{e} \, dv = \int_\Omega \phi \nabla \cdot \vec{u} \, dv - \oint_{\partial \Omega} \phi \hat{n} \cdot \vec{u} \, da
#     :label: tutorial_eq_1_2_11
# 
# According to expression :eq:`tutorial_eq_1_2_1`,
# :math:`- \frac{\partial \phi}{\partial n} = 0` on the boundaries.
# To accurately compute the electric potentials at cell centers,
# we must implement the boundary conditions by approximating the surface integral.
# In this case, expression :eq:`tutorial_eq_1_2_11` is approximated by:
# 
# .. math::
#     \boldsymbol{u^T M_f \, e} = \boldsymbol{u^T D^T M_c \, \phi} - \boldsymbol{u^T B \, \phi} = - \boldsymbol{\tilde{G} \, \phi}
#     :label: tutorial_eq_1_2_12
# 
# where
# 
#     - :math:`\boldsymbol{M_c}` is the inner product matrix at cell centers
#     - :math:`\boldsymbol{M_f}` is the inner product matrix at faces
#     - :math:`\boldsymbol{D}` is the discrete divergence operator
#     - :math:`\boldsymbol{B}` is a sparse matrix that imposes the Neumann boundary condition
#     - :math:`\boldsymbol{\tilde{G}} = - \boldsymbol{D^T M_c} + \boldsymbol{B}` acts as a modified gradient operator with boundary conditions included
# 

####################################################
# 4. Solvable Linear System
# -------------------------
# 
# By combining the discrete representations from expressions
# :eq:`tutorial_eq_1_2_10` and :eq:`tutorial_eq_1_2_12` and factoring like-terms
# we obtain:
# 
# .. math::
#     - \boldsymbol{M_c D M_f^{-1} \tilde{G} \, \phi} = \frac{1}{\epsilon_0} \boldsymbol{q}
#     :label: tutorial_eq_1_2_13
# 
# Once the electric potential at cell centers has been computed, the electric field on
# the faces can be computed using expression :eq:`tutorial_eq_1_2_12`:
# 
# .. math::
#     \boldsymbol{e} = - \boldsymbol{M_f^{-1} \tilde{G} \, \phi}
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

h = 2*np.ones(51)
mesh = TensorMesh([h, h], "CC")

#%%
# Construct the necessary discrete operators and inner product matrices.

DIV = mesh.faceDiv                                        # discrete divergence operator
Mc = sdiag(mesh.vol)                                      # cell center inner product matrix
Mf_inv = mesh.get_face_inner_product(invert_matrix=True)  # inverse of face inner product matrix

mesh.set_cell_gradient_BC(['neumann','neumann'])  # Set zero Neumann condition on gradient
G = mesh.cell_gradient                            # Modified gradient operator G = -D^T * Mc + B

#%%
# Form the linear system of equations to be solved.

A = - Mc * DIV * Mf_inv * G

#%%
# Construct the right-hand side.

xycc = mesh.gridCC
kneg = (xycc[:, 0] == -10) & (xycc[:, 1] == 0)      # -ve charge at (-10, 0)
kpos = (xycc[:, 0] == 10) & (xycc[:, 1] == 0)       # +ve charge at (10, 0)

rho = np.zeros(mesh.nC)
rho[kneg] = -1
rho[kpos] = 1

#%%
# Solve for the electric potential at cell centers and the electric field
# on the faces.

AinvM = SolverLU(A)     # Define the inverse of A using direct solver
phi = AinvM * rho       # Compute electric potential at cell centers
E = - Mf_inv * G * phi  # Compute electric field on faces

#%%
# Plot the source term, electric potential and electric field.

fig = plt.figure(figsize=(12, 4))

ax1 = fig.add_subplot(131)
mesh.plotImage(rho, v_type="CC", ax=ax1)
ax1.set_title("Charge Density")

ax2 = fig.add_subplot(132)
mesh.plotImage(phi, v_type="CC", ax=ax2)
ax2.set_title("Electric Potential")

ax3 = fig.add_subplot(133)
mesh.plotImage(
    E, ax=ax3, v_type="F", view="vec", stream_opts={"color": "w", "density": 1.0}
)
ax3.set_title("Electric Fields")

plt.tight_layout()

