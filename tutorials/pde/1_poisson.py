"""
Poisson Equation with Zero Neumann Boundary Condition
=====================================================

Here we use the discretize package to solve for the electric potential
(:math:`\phi`) and electric fields (:math:`\mathbf{e}`) in 2D that result from
a static charge distribution. Provided the electric permittiviy is uniform
within the domain, the physics are represented by a Poisson equation.
The solution can easily be adapted to solve the same problem in 3D.
In the theory section of the discretie website we provided a
:ref:`derivation for the final numerical solution <derivation_examples_poisson>`.

Starting with Gauss' law and Faraday's law:
    
.. math::
    &\\nabla \\cdot \mathbf{E} = \\frac{\\rho}{\\epsilon_0} \n
    &\\nabla \\times \mathbf{E} = \\mathbf{0} \;\;\; \Rightarrow \;\;\; \\mathbf{E} = -\\nabla \\phi \n
    &\\textrm{s.t.} \;\;\; \\hat{n} \\cdot \\vec{e} \Big |_{\partial \Omega} =
    -\\frac{\\partial \\phi}{\\partial n} \Big |_{\partial \Omega} = 0
    
where :math:`\\rho` is the charge density and :math:`\\epsilon_0` is the
permittivity of free space. We will consider the case where there is both a
positive and a negative charge of equal magnitude within our domain. Thus:

.. math::
    \\rho = \\rho_0 \\big [ \\delta ( \\mathbf{r_+}) - \\delta (\\mathbf{r_-} ) \\big ]


For :math:`\\phi` defined on the nodes, the numerical solution is obtained by
solving the following linear system:
    
.. math::
    \\boldsymbol{G^T M_e G \\, \\phi} = \\frac{1}{\\epsilon_0} \\boldsymbol{q}

And for :math:`\\phi` discretized at cell centers, the numerical solution is
obtained by solving:

.. math::
    - \\boldsymbol{M_c D M_f^{-1} \\tilde{G} \\, \\phi} = \\frac{1}{\\epsilon_0} \\boldsymbol{q}

where

    - :math:`\\boldsymbol{M_c}` is the inner product matrix for cell centered quantities
    - :math:`\\boldsymbol{M_e}` is the inner product matrix for edge quantities
    - :math:`\\boldsymbol{M_f}` is the inner product matrix for face quantities
    - :math:`\\boldsymbol{G}` is the discrete gradient operator
    - :math:`\\boldsymbol{D}` is the discrete divergence operator
    - :math:`\\boldsymbol{B}` is a sparse matrix that implements the boundary condition
    - :math:`\\boldsymbol{\\tilde{G}}=\\boldsymbol{-D^T M_c + B}` is the modified gradient operator
    - :math:`\\boldsymbol{q}` is a discrete representation of the source term


"""

###############################################
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
from discretize.utils import sdiag

mpl.rcParams.update({'font.size':14})

###############################################
# Electric Potential Defined on the Nodes
# ---------------------------------------
#
# Here, we solve the problem for the nodal discretization
# of the electric potential.
#

# Create a tensor mesh
h = np.ones(100)
mesh = TensorMesh([h, h], "CC")

# Define discrete operators
G = mesh.nodal_gradient                        # gradient operator
Me = mesh.get_edge_inner_product()             # edge inner product matrix

# Define linear system and remove null space
A = G.T * Me * G
A[0,0] = 1.
A[0, 1:] = 0

# Define RHS (total charge on each node)
xyn = mesh.nodes
kneg = (xyn[:, 0] == -10) & (xyn[:, 1] == 0)   # -ve charge at (-10, 0)
kpos = (xyn[:, 0] == 10) & (xyn[:, 1] == 0)    # +ve charge at (10, 0)

rho = np.zeros(mesh.n_nodes)
rho[kneg] = -1
rho[kpos] = 1

# LU factorization and solve
AinvM = SolverLU(A)
phi = AinvM * rho

# Compute electric fields
E = - G * phi

# Plotting
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

###############################################
# Electric Potential at Cell Centers
# ----------------------------------
# Here, we solve the problem for the cell centered discretization
# of the electric potential.
#

# Create a tensor mesh
h = 2*np.ones(51)
mesh = TensorMesh([h, h], "CC")

# Define discrete operators
DIV = mesh.faceDiv                                  # discrete divergence operator
Mc = sdiag(mesh.vol)                                # cell center inner product matrix
Mf_inv = mesh.get_face_inner_product(invMat=True)   # inverse of face inner product matrix

mesh.set_cell_gradient_BC(['neumann','neumann'])    # Set zero Neumann condition on gradient
G = mesh.cell_gradient                              # Modified gradient operator G = -D^T Mc + B

# Define the linear system of equations
A = - Mc * DIV * Mf_inv * G

# Define RHS (total charge projected to nearest cell center)
xycc = mesh.gridCC
kneg = (xycc[:, 0] == -10) & (xycc[:, 1] == 0)      # -ve charge at (-10, 0)
kpos = (xycc[:, 0] == 10) & (xycc[:, 1] == 0)       # +ve charge at (10, 0)

rho = np.zeros(mesh.nC)
rho[kneg] = -1
rho[kpos] = 1

# LU factorization and solve
AinvM = SolverLU(A)
phi = AinvM * rho

# Compute electric fields
E = - Mf_inv * G * phi

# Plotting
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


