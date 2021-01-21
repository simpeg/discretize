r"""
2D Advection-Diffusion Equation with Zero Neumann Condition
===========================================================

Here we use the *discretize package* to approximate the solution to
the 2D advection-diffusion equation with zero Neumann boundary conditions.
We assume the fluid is incompressible.
This tutorial focusses on:

    - strategies for applying the finite volume method to higher order PDEs
    - discretizing and solving time-dependent PDEs
    - including constitutive relationships defined by the reciprocal of a parameter
    

"""

####################################################
# 1. Formulating the Problem
# --------------------------
# 
# If we assume the fluid is incompressible (i.e. :math:`\nabla \cdot \vec{u} = 0`),
# the advection-diffusion equation with zero Neumann boundary conditions is given by:
# 
# .. math::
#     \begin{align}
#     & p_t = \nabla \cdot \alpha \nabla p - \vec{u} \cdot \nabla p + s \\
#     & \textrm{s.t.} \;\;\; \frac{\partial p}{\partial n} \Bigg |_{\partial \Omega} = 0 \\
#     & \textrm{and} \;\;\; p(t=0) = 0
#     \end{align}
#     :label: tutorial_eq_2_1_1
# 
# where 
# 
#     - :math:`p` is the unknown variable
#     - :math:`p_t` is its time-derivative
#     - :math:`\alpha` defines the diffusivity within the domain
#     - :math:`\vec{u}` is the velocity field
#     - :math:`s` is the source term
# 
# We will consider the case where there is a single point source within our domain.
# Where :math:`s_0` is a constant:
# 
# .. math::
#     s = s_0 \delta ( \vec{r} )
# 
# Direct implementation of the finite volume method is more challenging for higher order PDEs.
# One strategy is to redefine the problem as a set of first order PDEs:
# 
# .. math::
#     \begin{align}
#     p_t = \nabla \cdot \vec{j} - \vec{u} \cdot \vec{w} + s \;\;\;\; &(1)\\
#     \vec{w} = \nabla p \;\;\;\; &(2)\\
#     \alpha^{-1} \vec{j} = \vec{w} \;\;\;\; &(3)
#     \end{align}
#     :label: tutorial_eq_2_1_2
# 
# We can now take the inner products between each expression in equation
# :eq:`tutorial_eq_2_1_2` and an appropriate test function.
#

####################################################
# 2. Taking Inner Products
# ------------------------
# 
# The inner product between a scalar test function :math:`\psi` and the first equation
# in :eq:`tutorial_eq_2_1_2` is given by:
# 
# .. math::
#     \int_\Omega \psi p_t \, dv = \int_\Omega \psi \nabla \cdot \vec{j} \, dv + \int_\Omega \psi (\vec{u} \cdot \vec{w}) \, dv + \int_\Omega \psi s \, dv
#     :label: tutorial_eq_2_1_3
# 
# The inner product between a vector test function :math:`\vec{f}` and the second equation
# in :eq:`tutorial_eq_2_1_2` is given by:
# 
# .. math::
#     \int_\Omega \vec{f} \cdot \vec{w} = \int_\Omega \vec{f} \cdot \nabla p \, dv
#     :label: tutorial_eq_2_1_4
# 
# And the inner product between a vector test function :math:`\vec{f}` and the third equation
# in :eq:`tutorial_eq_2_1_2` is given by:
# 
# .. math::
#     \int_\Omega \vec{f} \cdot \alpha^{\! -1} \vec{j} \, dv = \int_\Omega \vec{f} \cdot \vec{w} \, dv
#     :label: tutorial_eq_2_1_5
# 

####################################################
# 3. Discretizing the Inner Products
# ----------------------------------
#
# Because this is a time-dependent problem, we must consider discretization in both space and time.
# We generally begin by discretizing in space, then we discretize in time.
# Here we let :math:`\boldsymbol{p}` be the discrete representation of the unkown variable :math:`p` and its time-derivative :math:`p_t`
# at cell centers. Examining expressions :eq:`tutorial_eq_2_1_3`,
# :eq:`tutorial_eq_2_1_4` and :eq:`tutorial_eq_2_1_5`:
# 
#     - The scalar test function :math:`\psi` must be discretized to cell centers 
#     - The source term :math:`s` must also be discretized to cell centers
#     - For this discretization, the divergence operator maps naturally from faces to cell centers and :math:`\vec{j}` must be discretized to faces
#     - For this discretization, the gradient operator maps naturally from cell centers to faces and :math:`\vec{w}` must be discretized to faces
#     - Since :math:`\vec{w}` is discretized to the faces, so must the vector field :math:`\vec{u}` and the vector test function :math:`\vec{f}`
# 
# **Inner Product #1:**
#
# Since the source term in expression :eq:`tutorial_eq_2_1_3` contains a Dirac delta function, 
# we rewrite the expression as: 
# 
# .. math::
#     \int_\Omega \psi p_t \, dv = \int_\Omega \psi \nabla \cdot \vec{j} \, dv + \int_\Omega \psi (\vec{u} \cdot \vec{w}) \, dv + \psi q
#     :label: tutorial_eq_2_1_6
#
# where :math:`q` is an integrated source term. Here, discrete representations of scalars (:math:`\boldsymbol{\psi}`, :math:`\boldsymbol{p_t}` and :math:`\boldsymbol{q}`)
# will live at cell centers, while discrete representations of vectors (:math:`\boldsymbol{j}`, :math:`\boldsymbol{u}` and :math:`\boldsymbol{w}`)
# will live on the faces.
#
# In the tutorials for inner products, we showed how to approximate most classes of inner products.
# However the third term in expression :eq:`tutorial_eq_2_1_6` is challenging.
# The inner product for this term is approximated using discrete scalar quantities at cell centers,
# but discrete vectors :math:`\boldsymbol{u}` and :math:`\boldsymbol{w}` live on the faces.
# To remedy this, we multiply the Cartesian components of :math:`\boldsymbol{u}` and :math:`\boldsymbol{w}`
# independently on their respective faces, then project the quantity to the cell centers
# before carrying out the dot product. Our final discretized representation of
# expression :eq:`tutorial_eq_2_1_6` is as follows:
#
# .. math::
#     \boldsymbol{\psi^T M_c \, p_t} = \boldsymbol{\psi^T M_c D \, j} -
#     c\, \boldsymbol{\psi^T M_c A_{fc}} \, diag(\boldsymbol{u}) \, \boldsymbol{w} + \boldsymbol{\psi^T q}
#     :label: tutorial_eq_2_1_7
# 
# where
# 
#     - :math:`\boldsymbol{M_c}` is the inner product matrix at cell centers
#     - :math:`\boldsymbol{D}` is the discrete divergence operator from faces to cell centers
#     - :math:`\boldsymbol{A_{fc}}` is a projection matrix from faces to cell centers
#     - :math:`c=1,2,3` is a constant equal to the dimension of the problem (*c* = 2 in this case)
#
# **Inner Product #2:**
#
# Consider expression :eq:`tutorial_eq_2_1_4`.
# The discrete gradient operator maps naturally from cell centers to faces.
# Unfortunately for boundary faces, this would require we know values of :math:`p` outside the domain.
# By using the vector identity :math:`\vec{f} \cdot \nabla p = \nabla \cdot p\vec{f} - p \nabla \cdot \vec{f}`
# and applying the divergence theorem, expression :eq:`tutorial_eq_2_1_4` becomes:
# 
# .. math::
#     \int_\Omega \vec{f} \cdot \vec{w} = - \int_\Omega p \nabla \cdot \vec{f} \, dv + \oint_{\partial \Omega} p \hat{n} \cdot \vec{f} \, da
#     :label: tutorial_eq_2_1_8
# 
# The discrete approximation is therefore given by:
# 
# .. math::
#     \boldsymbol{f^T M_f \, w} = - \boldsymbol{f^T D^T M_c \, p + f^T B \, p} = \boldsymbol{f^T \tilde{G} \, p}
#     :label: tutorial_eq_2_1_9
# 
# where
# 
#     - :math:`\boldsymbol{f}` is the discrete repressentation of :math:`\vec{f}` on cell faces
#     - :math:`\boldsymbol{M_f}` is the inner product matrix on cell faces
#     - :math:`\boldsymbol{B}` is a sparse matrix that imposes boundary conditions correctly on :math:`p`
#     - :math:`\boldsymbol{\tilde{G}} = \boldsymbol{-D^T M_c + B}` acts as a modified gradient operator with boundary conditions implemented
# 
# **Inner Product #3:**
#
# In expression :eq:`tutorial_eq_2_1_5`, we must take the inner product
# where the constitutive relation is defined by the reciprocal of a parameter.
# This was covered in the inner products section for constitutive relationships.
# The resulting approximation to expression :eq:`tutorial_eq_2_1_5`
# is given by:
#
# .. math::
#     \boldsymbol{f^T M_\alpha \, j} = \boldsymbol{f^T M_f w}
#     :label: tutorial_eq_2_1_10
# 
# where :math:`\boldsymbol{M_\alpha}` is the inner product matrix at faces
# for the reciprocal of the diffusivity.
#

####################################################
# 4. Solvable Linear System
# -------------------------
# 
# Before we can solve the problem numerically, we must amalgamate our
# discrete expressions and discretize in time.
# We begin by substituting the discrete representations of the inner products from expressions
# :eq:`tutorial_eq_2_1_8` and :eq:`tutorial_eq_2_1_10`
# into expression :eq:`tutorial_eq_2_1_6` and factoring like-terms.
# The resulting system of equations discretized in space is given by:
# 
# .. math::
#     \boldsymbol{p_t} = \boldsymbol{\big [ D \, M_\alpha^{-1} \tilde{G}} -
#     c\, \boldsymbol{A_{fc}} diag(\boldsymbol{u}) \, \boldsymbol{M_f^{-1} \tilde{G} \big ] \, p} + \boldsymbol{M_c^{-1} \, q}
#     :label: tutorial_eq_2_1_11
# 
# To discretize in time, let us re-express equations :eq:`tutorial_eq_2_1_11` as:
# 
# .. math::
#     \boldsymbol{p_t} = \boldsymbol{- M \, p + s}
#     :label: tutorial_eq_2_1_12
# 
# where
# 
# .. math::
#     \boldsymbol{M} = - \boldsymbol{D \, M_\alpha^{-1} \tilde{G}} +  
#     c\, \boldsymbol{A_{fc}} diag(\boldsymbol{u}) \, \boldsymbol{M_f^{-1} \tilde{G}}
#     :label: tutorial_eq_2_1_13
# 
# and
# 
# .. math::
#     \boldsymbol{s} = \boldsymbol{M_c^{-1} \, q}
#     :label: tutorial_eq_2_1_14
# 
# There are a multitude of ways in which discretization in time can be implemented.
# A stable and easy method to implement is the backward Euler.
# By implementing the backward Euler, we must solve the following linear system
# at each time step :math:`k`:
# 
# .. math::
#     \big [ \boldsymbol{I} + \Delta t \, \boldsymbol{M} \big ] \, \boldsymbol{p}^{k+1} = \boldsymbol{p}^k + \Delta t \, \boldsymbol{s}
#     :label: tutorial_eq_2_1_15
# 
# where :math:`\boldsymbol{I}` is the identity matrix and :math:`\Delta t` is the step length.
# 
# 


###############################################
# 5. Solving the System with Discretize
# -------------------------------------
#

#%%
# Import the necessary packages for the tutorial.

from discretize import TensorMesh
from pymatsolver import SolverLU
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from discretize.utils import sdiag, mkvc
mpl.rcParams.update({'font.size':14})
# sphinx_gallery_thumbnail_number = 2

#%%
# Construct a mesh.

h = np.ones(75)
mesh = TensorMesh([h, h], "CC")

#%%
# Define a divergence free vector field on the faces and plot.

faces_x = mesh.faces_x
faces_y = mesh.faces_y

r_x = np.sqrt(np.sum(faces_x ** 2, axis=1))
r_y = np.sqrt(np.sum(faces_y ** 2, axis=1))

ux = 0.5 * (-faces_x[:, 1] / r_x) * (1 + np.tanh(0.15 * (28.0 - r_x)))
uy = 0.5 * (faces_y[:, 0] / r_y) * (1 + np.tanh(0.15 * (28.0 - r_y)))

u_max = 10.0                # Maximum velocity is 10 m/s
u = u_max * np.r_[ux, uy]

fig = plt.figure(figsize=(8, 6))
ax = 2 * [None]

ax[0] = fig.add_axes([0.15, 0.1, 0.6, 0.8])
mesh.plotImage(
    u,
    ax=ax[0],
    v_type="F",
    view="vec",
    stream_opts={"color": "w", "density": 1.0},
    clim=[0.0, 10.0],
)
ax[0].set_title("Divergence free vector field")

ax[1] = fig.add_axes([0.8, 0.1, 0.05, 0.8])
ax[1].set_aspect(10, anchor="W")
norm = mpl.colors.Normalize(vmin=0, vmax=u_max)
cbar = mpl.colorbar.ColorbarBase(ax[1], norm=norm, orientation="vertical")
cbar.set_label("Velocity (m/s)", rotation=270, labelpad=15)

#%%
# Construct the source term. Her we define a discrete vector q where
# qi=1 at the nearest cell center and zero for all other cells.

xycc = mesh.cell_centers
k = (xycc[:, 0] == 0) & (xycc[:, 1] == -15)  # source at (0, -15)
q = np.zeros(mesh.nC)
q[k] = 1

#%%
# Define the diffusivity for all cells. In this case the diffusivity
# is the same for all cells. However, a more complex distribution
# of diffusivities could be created here.

a = mkvc(8.0 * np.ones(mesh.nC))

#%%
# Define any discrete operators and inner product matrices require to
# solve the problem.

Afc = mesh.average_face_to_cell                        # average face to cell matrix
Mf_inv = mesh.getFaceInnerProduct(invert_matrix=True)  # inverse of inner product matrix at faces
Mc = sdiag(mesh.vol)                                   # inner product matrix at centers
Mc_inv = sdiag(1 / mesh.vol)                           # inverse of inner product matrix at centers
Mf_alpha_inv = mesh.getFaceInnerProduct(
    a, invert_model=True, invert_matrix=True
)                       # Inverse of the inner product matrix for the reciprocal of the diffusivity

D = mesh.face_divergence                           # divergence operator

mesh.set_cell_gradient_BC(["neumann", "neumann"])  # Set zero Neumann BC
G = mesh.cell_gradient                             # modified gradient operator with BC implemented

#%%
# Construct the linear system that is solved at each time step.

M = -D * Mf_alpha_inv * G * Mc + mesh.dim * Afc * sdiag(u) * Mf_inv * G * Mc

dt = 0.02                          # Step width
p = np.zeros(mesh.nC)              # Initial conditions p(t=0)=0

I = sdiag(np.ones(mesh.nC))        # Identity matrix
B = I + dt * M                     # Linear system solved at each time step
s = Mc_inv * q                     # RHS

Binv = SolverLU(B)                 # Define inverse of B using solver

#%%
# Perform backward Euler at each time step and plot at specified times.

fig = plt.figure(figsize=(15, 10))
ax = 6 * [None]
n = 0

for ii in range(300):

    p = Binv * (p + s)

    if ii + 1 in (1, 25, 50, 100, 200, 300):
        ax[n] = fig.add_subplot(2, 3, n+1)
        mesh.plotImage(p, v_type="CC", ax=ax[n], pcolor_opts={"cmap": "gist_heat_r"})
        title_str = "p at t = " + str((ii + 1) * dt) + " s"
        ax[n].set_title(title_str)
        n = n + 1

plt.tight_layout()