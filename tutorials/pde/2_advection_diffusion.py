"""
Advection-Diffusion Equation
============================

Here we use the discretize package to model the advection-diffusion
equation. The goal of this tutorial is to demonstrate:

    - How to solve time-dependent PDEs
    - How to apply Neumann boundary conditions


Derivation
----------

If we assume the fluid is incompressible and that diffusivity is spatially,
the advection-diffusion equation with Neumann boundary conditions is given by:

.. math::
    p_t = \\alpha \\nabla \\cdot \\nabla p
    - \\mathbf{u} \\cdot \\nabla p + s \n
    \\textrm{s.t.} \;\;\; \\frac{\\partial p}{\\partial n} \Bigg |_{\partial \Omega} = 0

where :math:`p` is an unknown variable, :math:`\\alpha` is the diffusivity
constant, :math:`\\mathbf{u}` the velocity field and :math:`s` is the source
term. We will consider the case where there are two point sources within our
domain. Thus:

.. math::
    s = s_0 \\big [ \\delta ( \\mathbf{r_1}) + \\delta (\\mathbf{r_2} ) \\big ]

To solve this problem numerically, we use the weak formulation; that is, we
take the inner product of the equation with an appropriate test function
(:math:`\\psi`):

.. math::
    \int_\\Omega \\psi \\, p_t \\, dv =
    \\alpha \int_\\Omega \\psi (\\nabla \\cdot \\nabla p) \\, dv
    - \int_\\Omega \\psi \\, \mathbf{u} \\cdot \\nabla p \\, dv
    + \int_\\Omega \\psi \\, s \\, dv

The source term is a volume integral containing the Dirac delta function, thus:

.. math::
    \int_\\Omega \\psi \\, s \\, dv = \\psi \\, q

where :math:`q=s_0` at the location of either point source and zero everywhere
else. In the weak formulation, we generally remove divergence terms. If we use
the identity
:math:`\\phi \\nabla \\cdot \\mathbf{a} = \\nabla \\cdot (\\phi \\mathbf{a}) - \\mathbf{a} \\cdot (\\nabla \\phi )`
on the diffusion term and apply the divergence theorem we obtain:

.. math::
    \int_\\Omega \\psi \\, p_t \\, dv =
    \\alpha \int_{\\partial \\Omega} \\mathbf{n} \\cdot ( \\psi \\nabla p ) \\, da
    - \\alpha \int_\\Omega (\\nabla \\psi ) \\cdot (\\nabla p ) \\, dv
    - \int_\\Omega (\\psi \\, \mathbf{u} ) \\cdot \\nabla p dv
    + \\psi \\, q

Since the flux on the faces is zero at the boundaries, we can eliminate
the first term on the right-hand side. By evaluating the inner products
according to the finite volume approach we obtain:

.. math::
    \\mathbf{\\psi^T M_c p_t} = - \\alpha \\mathbf{\\psi^T G^T M_f G \\, p}
    - \\mathbf{\\psi^T M_c \\tilde{A}_{fc}} \\textrm{diag} ( \\mathbf{u} ) \\mathbf{G \\, p}
    + \\mathbf{\\psi^T M_c s}

where :math:`\\mathbf{\\psi}`, :math:`\\mathbf{p}` and :math:`\\mathbf{p_t}`
live at cell centers and :math:`\\mathbf{u}` lives on faces. :math:`\\mathbf{G}`
is the discrete gradient operators. :math:`\\mathbf{M_c}` and
:math:`\\mathbf{M_f}` are the cell center and face inner product matricies,
respectively. :math:`\\mathbf{\\tilde{A}_{cf}}` averages the x, y and z
contributions of :math:`\\mathbf{u} \\cdot \\nabla p` to cell centers and sums them.

By eliminating :math:`\\psi^T` and multiplying both sides by
:math:`\\mathbf{M_c^{-1}}` we obtain:

.. math::
    \\mathbf{p_t} = - \\mathbf{M} \\mathbf{p} + \\mathbf{s}

where

.. math::
    \\mathbf{M} = \\alpha \\mathbf{M_c^{-1} G^T M_f G}
    + \\mathbf{A_{fc}} \\textrm{diag}(\\mathbf{u}) \\mathbf{G}

For the example, we will discretize in time using backward Euler. This results
in the following system which must be solve at every time step :math:`k`.
Where :math:`\\Delta t` is the step size:

.. math::
    \\big [ \\mathbf{I} + \\Delta t \\, \\mathbf{M} \\big ] \\mathbf{p}^{k+1} =
    \\mathbf{p}^k + \\Delta t \\, \\mathbf{s}
    


"""

###################################################
#
# Import Packages
# ---------------
#
# Here we import the packages required for this tutorial.
#

from discretize import TensorMesh
from pymatsolver import SolverLU
import matplotlib.pyplot as plt
import numpy as np
from discretize.utils.matutils import sdiag

###############################################
#
# Solving the Problem
# -------------------
#

# Create a tensor mesh
h = np.ones(75)
mesh = TensorMesh([h, h], 'CC')

# Define diffusivity constant
a = 25

# Define velocity vector u = [ux, uy] = [1, -1] on faces
ux = 10*np.ones((mesh.nFx))
uy = - 10*np.ones((mesh.nFy))
u = np.r_[ux, uy]

# Define source term diag(v)*s
xycc = mesh.gridCC
k1 = (xycc[:, 0]==-25) & (xycc[:, 1]==25)   # source at (-25, 25)
k2 = (xycc[:, 0]==25) & (xycc[:, 1]==-25)   # sink at (25, -25)

s = np.zeros(mesh.nC)
s[k1] = 10
s[k2] = 10

# Define system M
Afc = mesh.dim*mesh.aveF2CC
Mc_inv = sdiag(1/mesh.vol)
Mf = mesh.getFaceInnerProduct()

mesh.setCellGradBC(['neumann', 'neumann'])  # Set Neumann BC
G = mesh.cellGrad

M = a*Mc_inv*G.T*Mf*G + Afc*sdiag(u)*G

# Set time stepping, initial conditions and final matricies
dt = 0.02              # Step width
p = np.zeros(mesh.nC)  # Initial conditions p(t=0)=0

I = sdiag(np.ones(mesh.nC))  # Identity matrix
B = I + dt*M
s = dt*(sdiag(mesh.vol)*s)

Binv = SolverLU(B)

# Perform backward Euler and plot
fig = plt.figure(figsize=(14, 4))
ax = 3*[None]
n = 0

for ii in range(250):

    p = Binv*(p + s)

    if ii+1 in (1, 25, 250):
        ax[n] = fig.add_subplot(1, 3, n+1)
        mesh.plotImage(p, vType='CC', ax=ax[n])
        title_str = 'p at t = ' + str((ii+1)*dt) + ' s'
        ax[n].set_title(title_str)
        n = n+1
