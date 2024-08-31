r"""
Nodal Dirichlet Poisson solution
================================
In this example, we demonstrate how to solve a Poisson equation where the
solution lives on nodes, and we want to impose Dirichlet conditions on the
boundary.

Solve a nodal Poisson's equation with Dirichlet boundary conditions
-------------------------------------------------------------------
The PDE we want to solve is Poisson's equation with a variable
conductivity, where we know the value of the solution on the boundary.

.. math::
    - \nabla \cdot \sigma \nabla u = f \\
    u = q \textrm{ on } \partial\Omega

We express this in a weak form, where we multiply by an arbitrary test function
and integrate over the volume:

.. math::
    \left<w,\nabla \cdot -\sigma \nabla u \right> = \left<w,f \right>

The left hand side can be changed to avoid taking the divergence of
:math:`-\sigma \nabla u` using a higher dimension integration by parts, instead
transfering the differentiability to :math:`w`.

.. math::
    \left<\nabla w,\sigma \nabla u \right>
    - \int_{\partial\Omega}w \left(\sigma \nabla u \right) \cdot \hat{n} dA
    = \left<w,f\right>

We are interested in solving this where :math:`u`, :math:`f`, and :math:`w` are
defined on cell nodes, :math:`\sigma` is defined on cell centers. Also of note
is that in the finite volume formulation we approximate the solution as
integrals of averages over each cell and the test functions and basis functions
are piecewise constant. Thus our discrete form goes to:

.. math::
    w^T G^T M_{e,\sigma} G u - w_b^T b_c = w^T M_n f

with :math:`b_c` representing the boundary condition on
:math:`\left(\sigma \nabla u \right) \cdot \hat{n}`. Where :math:`G` is a nodal
gradient operator that estimates the gradient at the edge centers,
:math:`M_{e, \sigma}` represents the inner product of vectors that live on mesh
edges taken with respect to :math:`\sigma`, and :math:`M_{n}` is
correspondingly the inner product operator for scalars that live on mesh nodes.

Taking care of the boundaries
-----------------------------

Set :math:`u_b`, and :math:`w_b` as the values of :math:`u` and :math:`w` on
the boundary, and :math:`u_f` and :math:`w_f` the values of :math:`u` and
:math:`w` on the interior. We have that

.. math::
    w = P_f w_f + P_b w_b \\
    u = P_f u_f + P_b u_b

Where :math:`P_b` and :math:`P_f` are the matrices which project the values
on the boundary and free interior nodes, respectively, to all of the nodes.

There is much freedom in what we choose to be our test functions, and to avoid
the integral on the boundary without assuming a value for
:math:`\left(\sigma \nabla u \right) \cdot \hat{n}`, we set that our test
functions are zero on the boundary, :math:`w_b = 0`

Thus the full system is

.. math::
    w_f^T P_f^T G^T M_{e,\sigma} G (P_f u_f + P_b u_b) = w_f^T P_f^T M_n f

Our final system of equations
-----------------------------

The above scalar equality must be true for all vectors :math:`w`, thus the
vectors are also equal:

.. math::
    P_f^T G^T M_{e,\sigma} G (P_f u_f + P_b u_b) = P_f^T M_n f

We then collect the unknowns :math:`u_f` on the left hand side, and move
everything else to the right.

.. math::
    P_f^T G^T M_{e,\sigma} G P_f u_f = P_f^T M_n f
    - P_f^T G^T M_{e,\sigma} G P_b u_b

Now lets actually code this solution up using these matrices!
"""

import numpy as np
import discretize
import matplotlib.pyplot as plt
import scipy.sparse as sp
from discretize import tests

import sympy
import sympy.vector as spv
from sympy.utilities.lambdify import lambdify

# %%
# Our PDE
# -------
# We are going to setup a simple two-dimensional problem which we will solve on
# a `TensorMesh`. We want to test the accuracy of our discretization scheme for
# the problem (it should be second order accurate). Thus we define a known
# solution :math:`u` and property model :math:`\sigma`. Then we use sympy to do
# all the derivatives for us to determine the correct forcing function.
#
# .. math::
#     u = \sin(x) \cos(y) e^{x + y} \\
#     \sigma = \cos(x) \sin(y)
#

C = spv.CoordSys3D("u")

u = sympy.sin(C.x) * sympy.cos(C.y) * sympy.exp(C.x + C.y)
sig = sympy.cos(C.x) * sympy.sin(C.y)

f = -spv.divergence(sig * spv.gradient(u))

# lambdify the functions so we can evalute them at any requested location
u_func = lambdify([C.x, C.y], u)
sig_func = lambdify([C.x, C.y], sig)
f_func = lambdify([C.x, C.y], f)
print(f)

# %%
# Next we are going to solve our PDE! We encapsulate the code into a
# function that returns the error and discretization size, so we can use
# `discretize`'s testing utilities to perform an order test for us.


def get_error(n_cells, plot_it=False):
    # Create a mesh with a certain number of cells on the [0, 1] square
    mesh = discretize.TensorMesh([n_cells, n_cells])
    h = mesh.nodes_x[1] - mesh.nodes_x[0]

    # evaluate our boundary conditions, the sigma model, and
    # the forcing function.
    u_b = u_func(*mesh.boundary_nodes.T)
    sig_cc = sig_func(*mesh.cell_centers.T)
    f_n = f_func(*mesh.nodes.T)

    # Get the finite volume operators for the mesh
    G = mesh.nodal_gradient
    M_e_sigma = mesh.get_edge_inner_product(sig_cc)
    M_n = sp.diags(mesh.average_node_to_cell.T @ mesh.cell_volumes)

    # Determin which mesh nodes are on the boundary, and which are interior
    is_boundary = np.zeros(mesh.n_nodes, dtype=bool)
    is_boundary[mesh.project_node_to_boundary_node.indices] = True

    # construct the projection matrices
    P_b = sp.eye(mesh.n_nodes, format="csc")[:, is_boundary]
    P_f = sp.eye(mesh.n_nodes, format="csc")[:, ~is_boundary]

    # Assemble the solution matrix and rhs.
    A = P_f.T @ G.T @ M_e_sigma @ G @ P_f
    rhs = P_f.T @ M_n @ f_n - P_f.T @ G.T @ M_e_sigma @ G @ P_b @ u_b

    # Solve for the value of u on the free nodes, then project
    # with the boundary conditions to obtain the solution on
    # all nodes.
    u_f = sp.linalg.spsolve(A, rhs)
    u_sol = P_f @ u_f + P_b @ u_b

    # Since we know the true solution we can check the error
    # which we expect to be second order accurate.
    u_true = u_func(*mesh.nodes.T)
    err = np.linalg.norm(u_sol - u_true, ord=np.inf)

    # Add some codes for us to use this function to plot the solution.
    if plot_it:
        ax = plt.subplot(121)
        (im,) = mesh.plot_image(u_sol, v_type="N", ax=ax)
        ax.set_title("Numerical Solution")
        ax.set_aspect("equal")
        plt.colorbar(im, shrink=0.3)

        ax = plt.subplot(122)
        (im,) = mesh.plot_image(u_sol - u_true, v_type="N", ax=ax)
        ax.set_title("Error")
        ax.set_aspect("equal")
        plt.colorbar(im, shrink=0.3)
        plt.tight_layout()
        plt.show()
    else:
        return err, h


# %%
# Lets look at the solution itself!
get_error(20, plot_it=True)

# %%
# Order Testing
# -------------
# To verify our discretization strategy we will use the above function to perform a
# convergence test. We make use of the `assert_expected_order` function to print out
# a convergence table. This will raise an error if our convergence is not what we
# expect.

tests.assert_expected_order(get_error, [10, 20, 30, 40, 50])
