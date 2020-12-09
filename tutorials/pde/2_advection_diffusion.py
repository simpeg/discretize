"""
Advection-Diffusion Equation
============================

Here we use the discretize package to solve the 2D advection-diffusion equation.
We implement a zero Neumann boundary condition on the unknown variable :math:`p`
and assume the fluid is incompressible. In the theory section of the discretie website we provided a
:ref:`derivation for the final numerical solution <derivation_examples_advection_diffusion>`.

If we assume the fluid is incompressible (i.e. :math:`\\nabla \\cdot \\vec{u} = 0`),
the advection-diffusion equation with zero Neumann boundary conditions is given by:

.. math::
    & p_t = \\nabla \\cdot \\alpha \\nabla p - \\vec{u} \\cdot \\nabla p + s \\\\
    & \\textrm{s.t.} \\;\\;\\; \\frac{\\partial p}{\\partial n} \\Bigg |_{\\partial \\Omega} = 0 \\\\
    & \\textrm{and} \\;\\;\\; p(t=0) = 0

where 

    - :math:`p` is the unknown variable
    - :math:`p_t` is its time derivative
    - :math:`\\alpha` defines the diffusivity within the domain
    - :math:`\\vec{u}` is the velocity field
    - :math:`s` is the source term

We will consider the case where there is a single point source within our domain.
Where :math:`s_0` is a constant, the source term is given by:

.. math::
    s = s_0 \\delta ( \\vec{r} )


The numerical solution was obtained by discretizing the unknown variable
to live at cell centers (:math:`\\boldsymbol{p}`) and using backward Euler to
discretize in time. Where :math:`\\Delta t` is the step length, the system which
must be solved at each time step :math:`k` is given by:
    
.. math::
    \\big [ \\boldsymbol{I} + \\Delta t \\, \\boldsymbol{M} \\big ] \\, \\boldsymbol{p}^{k+1} = \\boldsymbol{p}^k + \\Delta t \\, \\boldsymbol{s}

where

.. math::
    \\boldsymbol{M} = - \\boldsymbol{D \\, M_\\alpha^{-1} \\tilde{G}} +  
    c\\, \\boldsymbol{A_{fc}} diag(\\boldsymbol{u}) \\, \\boldsymbol{M_f^{-1} \\tilde{G}}

and

.. math::
    \\boldsymbol{s} = \\boldsymbol{M_c^{-1} \\, q}

Discrete operators are defined as follows:
    
    - :math:`\\boldsymbol{I}` is the identity matrix
    - :math:`\\boldsymbol{M_c}` is the inner product matrix at cell centers
    - :math:`\\boldsymbol{M_f}` is the inner product matrix on faces
    - :math:`\\boldsymbol{M_\\alpha}` is the inner product matrix at faces for the inverse of the diffusivity
    - :math:`\\boldsymbol{A_{fc}}` is averaging matrix from faces to cell centers
    - :math:`\\boldsymbol{D}` is the discrete divergence operator
    - :math:`\\boldsymbol{\\tilde{G}}` acts as a modified gradient operator which also implements the boundary conditions


"""

###################################################
# Import Packages
# ---------------
#
# Here we import the packages required for this tutorial.
#

from discretize import TensorMesh
from pymatsolver import SolverLU
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from discretize.utils import sdiag, mkvc
mpl.rcParams.update({'font.size':14})

###############################################
# Create a tensor mesh
# --------------------
#

h = np.ones(75)
mesh = TensorMesh([h, h], "CC")

#####################################################
# Define a divergence free vector field on faces
# ----------------------------------------------
#

faces_x = mesh.faces_x
faces_y = mesh.faces_y

r_x = np.sqrt(np.sum(faces_x ** 2, axis=1))
r_y = np.sqrt(np.sum(faces_y ** 2, axis=1))

ux = 0.5 * (-faces_x[:, 1] / r_x) * (1 + np.tanh(0.15 * (28.0 - r_x)))
uy = 0.5 * (faces_y[:, 0] / r_y) * (1 + np.tanh(0.15 * (28.0 - r_y)))

u = 10.0 * np.r_[ux, uy]  # Maximum velocity is 10 m/s

#####################################################
# Define the source term
# ----------------------
#

# Define vector q where qi=1 at the nearest cell center
xycc = mesh.cell_centers
k = (xycc[:, 0] == 0) & (xycc[:, 1] == -15)  # source at (0, -15)

q = np.zeros(mesh.nC)
q[k] = 1

#####################################################
# Define discrete operators and diffusivity
# -----------------------------------------
#

# Define diffusivity for all cells
a = mkvc(8.0 * np.ones(mesh.nC))

# Define discrete operators
Afc = mesh.average_face_to_cell                   # average face to cell matrix
Mf_inv = mesh.getFaceInnerProduct(invMat=True)    # inverse of inner product matrix at faces
Mc = sdiag(mesh.vol)                              # inner product matrix at centers
Mc_inv = sdiag(1 / mesh.vol)                      # inverse of inner product matrix at centers

# Inverse of the inner product matrix for the reciprocal of the diffusivity
Mf_alpha_inv = mesh.getFaceInnerProduct(a, invProp=True, invMat=True)  

D = mesh.face_divergence                          # divergence operator

mesh.set_cell_gradient_BC(["neumann", "neumann"]) # Set zero Neumann BC
G = mesh.cell_gradient                            # modified gradient operator with BC implemented

# Construct matrix M
M = -D * Mf_alpha_inv * G * Mc + mesh.dim * Afc * sdiag(u) * Mf_inv * G * Mc

#####################################################
# Define time discretization using backward Euler
# -----------------------------------------------
#

dt = 0.02                          # Step width
p = np.zeros(mesh.nC)              # Initial conditions p(t=0)=0

I = sdiag(np.ones(mesh.nC))        # Identity matrix
B = I + dt * M                     # Linear system solved at each time step
s = Mc_inv * q                     # RHS

Binv = SolverLU(B)                 # Define inverse using solver

#####################################################
# Carry out time stepping and plot progress
# -----------------------------------------
#

# Plot the vector field
fig = plt.figure(figsize=(15, 15))
ax = 9 * [None]

ax[0] = fig.add_subplot(332)
mesh.plotImage(
    u,
    ax=ax[0],
    v_type="F",
    view="vec",
    stream_opts={"color": "w", "density": 1.0},
    clim=[0.0, 10.0],
)
ax[0].set_title("Divergence free vector field")

ax[1] = fig.add_subplot(333)
ax[1].set_aspect(10, anchor="W")
cbar = mpl.colorbar.ColorbarBase(ax[1], orientation="vertical")
cbar.set_label("Velocity (m/s)", rotation=270, labelpad=15)

# Perform backward Euler and plot at specified times
n = 3

for ii in range(300):

    p = Binv * (p + s)

    if ii + 1 in (1, 25, 50, 100, 200, 300):
        ax[n] = fig.add_subplot(3, 3, n + 1)
        mesh.plotImage(p, v_type="CC", ax=ax[n], pcolor_opts={"cmap": "gist_heat_r"})
        title_str = "p at t = " + str((ii + 1) * dt) + " s"
        ax[n].set_title(title_str)
        n = n + 1

plt.tight_layout()