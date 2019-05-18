"""
Gauss' Law of Electrostatics
============================

Derivation
----------

Here we use the discretize package to solve for the electric potential
(:math:`\phi`) and electric fields (:math:`\mathbf{E}`) in 2D that result from
a static charge distribution. Starting with Gauss' law and Faraday's law:

.. math::
    &\\nabla \\cdot \\mathbf{E} = \\frac{\\rho}{\\epsilon_0} \n
    &\\nabla \\times \\mathbf{E} = \\mathbf{0} \\;\\;\\; \\Rightarrow \;\;\; \\mathbf{E} = -\\nabla \phi \n
    &\\textrm{s.t.} \;\;\; \\phi \\Big |_{\\partial \\Omega} = 0

where :math:`\\rho` is the charge density and :math:`\\epsilon_0` is the
permittivity of free space.

To solve this problem numerically, we use the weak formulation; that is, we
take the inner product of each equation with an appropriate test function:

.. math::
    &\\int_\\Omega \\psi (\\nabla \\cdot \\mathbf{E}) dV = \\frac{1}{\\epsilon_0} \\int_\\Omega \\psi \\rho dV \n
    &\\int_\\Omega \\mathbf{f \\cdot e} \\; dV = - \\int_\\Omega \\mathbf{f} \\cdot (\\nabla \\phi ) \\, dV


where :math:`\\psi` is a scalar test function and :math:`\\mathbf{f}` is a
vector test function. We then evaluate the inner products according to the
finite volume approach:

.. math::
    &\\mathbf{\\psi^T} \\textrm{diag} (\\mathbf{v}) \\mathbf{D e} = \\frac{1}{\\epsilon_0} \\mathbf{\\psi^T} \\textrm{diag} (\\mathbf{v}) \\mathbf{\\rho} \n
    &\\mathbf{f^T} \\textrm{diag} (\\mathbf{A_{cf} v}) \\mathbf{e} = - \\mathbf{f^T} \\textrm{diag} (\\mathbf{A_{cf} v}) \\mathbf{G} \\mathbf{\\phi}


where :math:`\\mathbf{\\psi}`, :math:`\\mathbf{\\rho}` and :math:`\\mathbf{v}`
live at cell centers, and :math:`\\mathbf{f}` and :math:`\\mathbf{e}` live on
cell faces. :math:`\\mathbf{D}` and :math:`\\mathbf{G}` are the discrete
divergence and gradient operators, respectively. And :math:`\\mathbf{A_{cf}}`
averages from cell centers to faces.

By cancelling terms and combining the set of discrete equations we obtain:

.. math::
    \\mathbf{D \\, G \\, \\phi} = -\\frac{1}{\\epsilon_0} \\mathbf{\\rho}

from which we can solve for :math:`\\mathbf{\\phi}`. The electric field can be
obtained by computing:

.. math::
    \\mathbf{e = - G \\, \\phi}



"""

########################################################################
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


########################################################################
#
# Solving the Problem
# -------------------
#

# Create a tensor mesh
h = np.ones(75)
mesh = TensorMesh([h, h], 'CC')

# Define charge distributions at cell centers
xycc = mesh.gridCC
kneg = (xycc[:, 0]==-10) & (xycc[:, 1]==0)  # -ve charge distr. at (-10, 0)
kpos = (xycc[:, 0]==10) & (xycc[:, 1]==0)   # +ve charge distr. at (10, 0)

rho = np.zeros(mesh.nC)
rho[kneg] = -1
rho[kpos] = 1

# Create system
DIV = mesh.faceDiv  # Faces to cell centers divergence
mesh.setCellGradBC(['dirichlet','dirichlet'])  # Set Dirichlet BC
GRAD = mesh.cellGrad  # Cell centers to faces gradient
A = DIV*GRAD

# Define RHS
RHS = -rho

# LU factorization and solve
AinvM = SolverLU(A)
phi = AinvM*RHS

# Compute electric fields
E = - mesh.cellGrad*phi

# Plotting
fig = plt.figure(figsize=(14, 4))

Ax1 = fig.add_subplot(131)
mesh.plotImage(rho, vType='CC', ax=Ax1)
Ax1.set_title('Charge Density')

Ax2 = fig.add_subplot(132)
mesh.plotImage(phi, vType='CC', ax=Ax2)
Ax2.set_title('Electric Potential')

Ax3 = fig.add_subplot(133)
mesh.plotImage(E, ax=Ax3, vType='F', view='vec',
               streamOpts={'color': 'w', 'density': 1.0})
Ax3.set_title('Electric Fields')

