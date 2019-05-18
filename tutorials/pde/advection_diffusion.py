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

where :math:`p` is an unknown variable, :math:`\\alpha` is the
diffusivity constant, :math:`\\mathbf{u}` the velocity
field and :math:`s` is the source term.

To solve this problem numerically, we use the weak formulation; that is, we
take the inner product of the equation with an appropriate test function
(:math:`\\psi`):

.. math::
    \int_\\Omega \\psi \\, p_t \\, dv =
    \\alpha \int_\\Omega \\psi (\\nabla \\cdot \\nabla p) \\, dv
    - \int_\\Omega (\\psi \\, \mathbf{u} ) \\cdot \\nabla p dv
    + \int_\\Omega \\psi \\, s \\, dv

In the weak formulation, we generally remove divergence terms. If we use the identity
:math:`\\phi \\nabla \\cdot \\mathbf{a} = \\nabla \\cdot (\\phi \\mathbf{a}) - \\mathbf{a} \\cdot (\\nabla \\phi )`
on the diffusion term and apply the divergence theorem we obtain:

.. math::
    \int_\\Omega \\psi \\, p_t \\, dv =
    \\alpha \int_{\\partial \\Omega} \\mathbf{n} \\cdot ( \\psi \\nabla p ) \\, da
    - \\alpha \int_\\Omega (\\nabla \\psi ) \\cdot (\\nabla p ) \\, dv
    - \int_\\Omega (\\psi \\, \mathbf{u} ) \\cdot \\nabla p dv
    + \int_\\Omega \\psi \\, s \\, dv

Since the flux on the faces is zero at the boundaries, we can eliminate
the first term on the right-hand side. By evaluating the inner products
according to the finite volume approach we obtain:

.. math::
    \\mathbf{\\psi^T p_t} = - \\alpha \\mathbf{\\psi^T G^T} \\textrm{diag} (\\mathbf{A_{cf}v}) \\mathbf{G \\, p}
    - \\mathbf{\\psi^T A_{cf}^T} \\textrm{diag} ( \\mathbf{u} \\odot (\\mathbf{A_{cf} v})) \\mathbf{G \\, p}
    + \\mathbf{\\psi^T} \\textrm{diag} (\\mathbf{v}) \\mathbf{s}

where :math:`\\mathbf{\\psi}`, :math:`\\mathbf{p}`, :math:`\\mathbf{p_t}` and
:math:`\\mathbf{v}` live at cell centers and :math:`\\mathbf{u}` lives on faces.
:math:`\\mathbf{D}` and :math:`\\mathbf{G}` are the discrete
divergence and gradient operators, respectively. And :math:`\\mathbf{A_{cf}}`
averages from cell centers to faces.

By eliminating :math:`\\psi^T` and amalgamating terms we obtain:

.. math::
    \\mathbf{p_t} = - \\mathbf{M} \\mathbf{p} + \\textrm{diag} (\\mathbf{v}) \\mathbf{s}

where

.. math::
    \\mathbf{M} = \\alpha \\mathbf{G^T} \\textrm{diag} (\\mathbf{A_{cf}v}) \\mathbf{G}
    + \\mathbf{A_{cf}^T} \\textrm{diag} ( \\mathbf{u} \\odot (\\mathbf{A_{cf} v})) \\mathbf{G}


For the example, we will discretize in time using backward Euler. This results
in the following system which must be solve at every time step :math:`k`. 
Where :math:`\\Delta t` is the step size:

.. math::
    \\big [ \\mathbf{I} + \\Delta t \\, \\mathbf{M} \\big ] \\mathbf{p}^{k+1} =
    \\mathbf{p}^k + \\Delta t \\, \\textrm{diag} ( \\mathbf{v} ) \\, \\mathbf{s}


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


##################################################
#
# Solving Problem in 2D
# ---------------------
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

# Define source term
xycc = mesh.gridCC
k1 = (xycc[:, 0]==-20) & (xycc[:, 1]==20)   # source at (-10, 10)
k2 = (xycc[:, 0]==25) & (xycc[:, 1]==-25)   # source at (10, -10)

s = np.zeros(mesh.nC)
s[k1] = 10
s[k2] = 10

# Define system M
v = mesh.vol  # Cell volumes (area in 2d)

Acf = mesh.aveCC2F  # Average centers to faces

mesh.setCellGradBC(['neumann','neumann'])  # Set Neumann BC
G = mesh.cellGrad  # Cell centers to faces gradient

M = a*G.T*(sdiag(Acf*v)*G) + Acf.T*(sdiag(u*(Acf*v))*G)

# Set time stepping, initial conditions and final matricies
dt = 0.02              # Step width
p = np.zeros(mesh.nC)  # Initial conditions p(t=0)=0

I = sdiag(np.ones(mesh.nC))  # Identity matrix
B = I + dt*M
s = dt*(sdiag(mesh.vol)*s)

Binv = SolverLU(B)

# Perform backward Euler and plot
fig = plt.figure(figsize=(14, 4))
Ax = 3*[None]
n = 0

for ii in range(250):

    p = Binv*(p + s)

    if ii+1 in (1, 25, 250):
        Ax[n] = fig.add_subplot(1, 3, n+1)
        mesh.plotImage(p, vType='CC', ax=Ax[n])
        title_str = 'p at t = ' + str((ii+1)*dt) + ' s'
        Ax[n].set_title(title_str)
        n = n+1
