import numpy as np
from scipy import sparse as sp
import warnings
from discretize.utils import (
    sdiag,
    speye,
    kron3,
    spzeros,
    ddx,
    av,
    av_extrap,
    make_boundary_bool,
)
from discretize.utils.code_utils import deprecate_method, deprecate_property


def _validate_BC(bc):
    """Checks if boundary condition 'bc' is valid.

    Each bc must be either 'dirichlet' or 'neumann'

    """
    if isinstance(bc, str):
        bc = [bc, bc]
    if not isinstance(bc, list):
        raise TypeError("bc must be a single string or list of strings")
    if not len(bc) == 2:
        raise TypeError("bc list must have two elements, one for each side")

    for bc_i in bc:
        if not isinstance(bc_i, str):
            raise TypeError("each bc must be a string")
        if bc_i not in ["dirichlet", "neumann"]:
            raise ValueError("each bc must be either, 'dirichlet' or 'neumann'")
    return bc


def _ddxCellGrad(n, bc):
    """
    Create 1D derivative operator from cell-centers to nodes this means we
    go from n to n+1

    For Cell-Centered **Dirichlet**, use a ghost point::

        (u_1 - u_g)/hf = grad

            u_g       u_1      u_2
             *    |    *   |    *     ...
                  ^
                  0

        u_g = - u_1
        grad = 2*u1/dx
        negitive on the other side.

    For Cell-Centered **Neumann**, use a ghost point::

        (u_1 - u_g)/hf = 0

            u_g       u_1      u_2
             *    |    *   |    *     ...

        u_g = u_1
        grad = 0;  put a zero in.

    """
    bc = _validate_BC(bc)

    D = sp.spdiags((np.ones((n + 1, 1)) * [-1, 1]).T, [-1, 0], n + 1, n, format="csr")
    # Set the first side
    if bc[0] == "dirichlet":
        D[0, 0] = 2
    elif bc[0] == "neumann":
        D[0, 0] = 0
    # Set the second side
    if bc[1] == "dirichlet":
        D[-1, -1] = -2
    elif bc[1] == "neumann":
        D[-1, -1] = 0
    return D


def _ddxCellGradBC(n, bc):
    """
    Create 1D derivative operator from cell-centers to nodes this means we
    go from n to n+1

    For Cell-Centered **Dirichlet**, use a ghost point::

        (u_1 - u_g)/hf = grad

         u_g       u_1      u_2
          *    |    *   |    *     ...
               ^
              u_b

    We know the value at the boundary (u_b)::

        (u_g+u_1)/2 = u_b               (the average)
        u_g = 2*u_b - u_1

        So plug in to gradient:

        (u_1 - (2*u_b - u_1))/hf = grad
        2*(u_1-u_b)/hf = grad

    Separate, because BC are known (and can move to RHS later)::

        ( 2/hf )*u_1 + ( -2/hf )*u_b = grad

                       (   ^   ) JUST RETURN THIS


    """
    bc = _validate_BC(bc)

    ij = (np.array([0, n]), np.array([0, 1]))
    vals = np.zeros(2)

    # Set the first side
    if bc[0] == "dirichlet":
        vals[0] = -2
    elif bc[0] == "neumann":
        vals[0] = 0
    # Set the second side
    if bc[1] == "dirichlet":
        vals[1] = 2
    elif bc[1] == "neumann":
        vals[1] = 0
    D = sp.csr_matrix((vals, ij), shape=(n + 1, 2))
    return D


class DiffOperators(object):
    """
    Class creates the differential operators that you need!
    """

    _aliases = {
        "aveF2CC": "average_face_to_cell",
        "aveF2CCV": "average_face_to_cell_vector",
        "aveFx2CC": "average_face_x_to_cell",
        "aveFy2CC": "average_face_y_to_cell",
        "aveFz2CC": "average_face_z_to_cell",
        "aveCC2F": "average_cell_to_face",
        "aveCCV2F": "average_cell_vector_to_face",
        "aveE2CC": "average_edge_to_cell",
        "aveE2CCV": "average_edge_to_cell_vector",
        "aveEx2CC": "average_edge_x_to_cell",
        "aveEy2CC": "average_edge_y_to_cell",
        "aveEz2CC": "average_edge_z_to_cell",
        "aveN2CC": "average_node_to_cell",
        "aveN2E": "average_node_to_edge",
        "aveN2F": "average_node_to_face",
    }

    ###########################################################################
    #                                                                         #
    #                             Face Divergence                             #
    #                                                                         #
    ###########################################################################
    @property
    def _face_x_divergence_stencil(self):
        """
        Face divergence operator in the x-direction (x-faces to cell centers)
        """
        if self.dim == 1:
            Dx = ddx(self.shape_cells[0])
        elif self.dim == 2:
            Dx = sp.kron(speye(self.shape_cells[1]), ddx(self.shape_cells[0]))
        elif self.dim == 3:
            Dx = kron3(
                speye(self.shape_cells[2]),
                speye(self.shape_cells[1]),
                ddx(self.shape_cells[0]),
            )
        return Dx

    @property
    def _face_y_divergence_stencil(self):
        """
        Face divergence operator in the y-direction (y-faces to cell centers)
        """
        if self.dim == 1:
            return None
        elif self.dim == 2:
            Dy = sp.kron(ddx(self.shape_cells[1]), speye(self.shape_cells[0]))
        elif self.dim == 3:
            Dy = kron3(
                speye(self.shape_cells[2]),
                ddx(self.shape_cells[1]),
                speye(self.shape_cells[0]),
            )
        return Dy

    @property
    def _face_z_divergence_stencil(self):
        """
        Face divergence operator in the z-direction (z-faces to cell centers)
        """
        if self.dim == 1 or self.dim == 2:
            return None
        elif self.dim == 3:
            Dz = kron3(
                ddx(self.shape_cells[2]),
                speye(self.shape_cells[1]),
                speye(self.shape_cells[0]),
            )
        return Dz

    @property
    def _face_divergence_stencil(self):
        # Compute faceDivergence stencil on faces
        if self.dim == 1:
            D = self._face_x_divergence_stencil
        elif self.dim == 2:
            D = sp.hstack(
                (self._face_x_divergence_stencil, self._face_y_divergence_stencil),
                format="csr",
            )
        elif self.dim == 3:
            D = sp.hstack(
                (
                    self._face_x_divergence_stencil,
                    self._face_y_divergence_stencil,
                    self._face_z_divergence_stencil,
                ),
                format="csr",
            )
        return D

    @property
    def face_divergence(self):
        """
        Construct divergence operator (face-stg to cell-centres).
        """
        if getattr(self, "_face_divergence", None) is None:
            # Get the stencil of +1, -1's
            D = self._face_divergence_stencil
            # Compute areas of cell faces & volumes
            S = self.face_areas
            V = self.cell_volumes
            self._face_divergence = sdiag(1 / V) * D * sdiag(S)
        return self._face_divergence

    @property
    def face_x_divergence(self):
        """
        Construct divergence operator in the x component (face-stg to
        cell-centres).
        """
        # Compute areas of cell faces & volumes
        S = self.reshape(self.face_areas, "F", "Fx", "V")
        V = self.cell_volumes
        return sdiag(1 / V) * self._face_x_divergence_stencil * sdiag(S)

    @property
    def face_y_divergence(self):
        if self.dim < 2:
            return None
        # Compute areas of cell faces & volumes
        S = self.reshape(self.face_areas, "F", "Fy", "V")
        V = self.cell_volumes
        return sdiag(1 / V) * self._face_y_divergence_stencil * sdiag(S)

    @property
    def face_z_divergence(self):
        """
        Construct divergence operator in the z component (face-stg to
        cell-centers).
        """
        if self.dim < 3:
            return None
        # Compute areas of cell faces & volumes
        S = self.reshape(self.face_areas, "F", "Fz", "V")
        V = self.cell_volumes
        return sdiag(1 / V) * self._face_z_divergence_stencil * sdiag(S)

    ###########################################################################
    #                                                                         #
    #                          Nodal Diff Operators                           #
    #                                                                         #
    ###########################################################################

    @property
    def _nodal_gradient_x_stencil(self):
        """
        Stencil for the nodal grad in the x-direction (nodes to x-edges)
        """
        if self.dim == 1:
            Gx = ddx(self.shape_cells[0])
        elif self.dim == 2:
            Gx = sp.kron(speye(self.shape_nodes[1]), ddx(self.shape_cells[0]))
        elif self.dim == 3:
            Gx = kron3(
                speye(self.shape_nodes[2]),
                speye(self.shape_nodes[1]),
                ddx(self.shape_cells[0]),
            )
        return Gx

    @property
    def _nodal_gradient_y_stencil(self):
        """
        Stencil for the nodal grad in the y-direction (nodes to y-edges)
        """
        if self.dim == 1:
            return None
        elif self.dim == 2:
            Gy = sp.kron(ddx(self.shape_cells[1]), speye(self.shape_nodes[0]))
        elif self.dim == 3:
            Gy = kron3(
                speye(self.shape_nodes[2]),
                ddx(self.shape_cells[1]),
                speye(self.shape_nodes[0]),
            )
        return Gy

    @property
    def _nodal_gradient_z_stencil(self):
        """
        Stencil for the nodal grad in the z-direction (nodes to z- edges)
        """
        if self.dim == 1 or self.dim == 2:
            return None
        else:
            Gz = kron3(
                ddx(self.shape_cells[2]),
                speye(self.shape_nodes[1]),
                speye(self.shape_nodes[0]),
            )
        return Gz

    @property
    def _nodal_gradient_stencil(self):
        """
        Stencil for the nodal grad
        """
        # Compute divergence operator on faces
        if self.dim == 1:
            G = self._nodal_gradient_x_stencil
        elif self.dim == 2:
            G = sp.vstack(
                (self._nodal_gradient_x_stencil, self._nodal_gradient_y_stencil),
                format="csr",
            )
        elif self.dim == 3:
            G = sp.vstack(
                (
                    self._nodal_gradient_x_stencil,
                    self._nodal_gradient_y_stencil,
                    self._nodal_gradient_z_stencil,
                ),
                format="csr",
            )
        return G

    @property
    def nodal_gradient(self):
        """
        Construct gradient operator (nodes to edges).
        """
        if getattr(self, "_nodal_gradient", None) is None:
            G = self._nodal_gradient_stencil
            L = self.edge_lengths
            self._nodal_gradient = sdiag(1 / L) * G
        return self._nodal_gradient

    @property
    def _nodal_laplacian_x_stencil(self):
        warnings.warn("Laplacian has not been tested rigorously.")

        Dx = ddx(self.shape_cells[0])
        Lx = -Dx.T * Dx

        if self.dim == 2:
            Lx = sp.kron(speye(self.shape_nodes[1]), Lx)
        elif self.dim == 3:
            Lx = kron3(speye(self.shape_nodes[2]), speye(self.shape_nodes[1]), Lx)
        return Lx

    @property
    def _nodal_laplacian_y_stencil(self):
        warnings.warn("Laplacian has not been tested rigorously.")

        if self.dim == 1:
            return None

        Dy = ddx(self.shape_cells[1])
        Ly = -Dy.T * Dy

        if self.dim == 2:
            Ly = sp.kron(Ly, speye(self.shape_nodes[0]))
        elif self.dim == 3:
            Ly = kron3(speye(self.shape_nodes[2]), Ly, speye(self.shape_nodes[0]))
        return Ly

    @property
    def _nodal_laplacian_z_stencil(self):
        warnings.warn("Laplacian has not been tested rigorously.")

        if self.dim == 1 or self.dim == 2:
            return None

        Dz = ddx(self.shape_cells[2])
        Lz = -Dz.T * Dz
        return kron3(Lz, speye(self.shape_nodes[1]), speye(self.shape_nodes[0]))

    @property
    def _nodal_laplacian_x(self):
        Hx = sdiag(1.0 / self.h[0])
        if self.dim == 2:
            Hx = sp.kron(speye(self.shape_nodes[1]), Hx)
        elif self.dim == 3:
            Hx = kron3(speye(self.shape_nodes[2]), speye(self.shape_nodes[1]), Hx)
        return Hx.T * self._nodal_gradient_x_stencil * Hx

    @property
    def _nodal_laplacian_y(self):
        Hy = sdiag(1.0 / self.h[1])
        if self.dim == 1:
            return None
        elif self.dim == 2:
            Hy = sp.kron(Hy, speye(self.shape_nodes[0]))
        elif self.dim == 3:
            Hy = kron3(speye(self.shape_nodes[2]), Hy, speye(self.shape_nodes[0]))
        return Hy.T * self._nodal_gradient_y_stencil * Hy

    @property
    def _nodal_laplacian_z(self):
        if self.dim == 1 or self.dim == 2:
            return None
        Hz = sdiag(1.0 / self.h[2])
        Hz = kron3(Hz, speye(self.shape_nodes[1]), speye(self.shape_nodes[0]))
        return Hz.T * self._nodal_laplacian_z_stencil * Hz

    @property
    def nodal_laplacian(self):
        """
        Construct laplacian operator (nodes to edges).
        """
        if getattr(self, "_nodal_laplacian", None) is None:
            warnings.warn("Laplacian has not been tested rigorously.")
            # Compute divergence operator on faces
            if self.dim == 1:
                self._nodal_laplacian = self._nodal_laplacian_x
            elif self.dim == 2:
                self._nodal_laplacian = (
                    self._nodal_laplacian_x + self._nodal_laplacian_y
                )
            elif self.dim == 3:
                self._nodal_laplacian = (
                    self._nodal_laplacian_x
                    + self._nodal_laplacian_y
                    + self._nodal_laplacian_z
                )
        return self._nodal_laplacian

    def edge_divergence_weak_form_robin(self, alpha=0.0, beta=1.0, gamma=0.0):
        r"""Robin boundary condition for the weak formulation of the edge divergence

        This function returns the necessary parts to form the full weak form of the
        edge divergence using the nodal gradient with appropriate boundary conditions.

        The `alpha`, `beta`, and `gamma` parameters can be scalars, or arrays. If they
        are arrays, they can either be the same length as the number of boundary faces,
        or boundary nodes. If multiple parameters are arrays, they must all be the same
        length.

        `beta` can not be 0.

        It is assumed here that quantity that is approximated on the boundary is the
        gradient of another quantity. See the Notes section for explicit details.

        Parameters
        ----------
        alpha, beta : scalar or array_like
            Parameters for the Robin boundary condition. array_like must be defined on
            either boundary faces or boundary nodes.
        gamma: scalar or array_like
            right hand side boundary conditions. If this parameter is array like, it can
            be fed either a (n_boundary_XXX,) shape array or an (n_boundary_XXX, n_rhs)
            shape array if multiple systems have the same alpha and beta parameters.

        Notes
        -----
        For these returned operators, it is assumed that the quantity on the boundary is
        related to the gradient of some other quantity.

        The weak form is obtained by multiplying the divergence by a (piecewise-constant)
        test function, and integrating over the cell, i.e.

        .. math:: \int_V y \nabla \cdot \vec{u} \partial V
            :label: weak_form

        This equation can be transformed to reduce the differentiability requirement on
        :math:`\vec{u}` to be,

        .. math:: -\int_V \vec{u} \cdot (\nabla y) \partial V + \int_{dV} y \vec{u} \cdot \hat{n} \partial S.
            :label: transformed

        Furthermore, when applying these types of transformations, the unknown vector
        :math:`\vec{u}` is usually related to some scalar potential as:

        .. math:: \vec{u} = \nabla \phi
            :label: gradient of potential

        Thus the robin conditions returned by these matrices apply to the quantity of
        :math:`\phi`.

        .. math::
            \alpha \phi + \beta \nabla \phi \cdot \hat{n} = \gamma

            \alpha \phi + \beta \vec{u} \cdot \hat{n} = \gamma

        The returned operators cannot be used to impose a Dirichlet condition on
        :math:`\phi`.

        """
        alpha = np.atleast_1d(alpha)
        beta = np.atleast_1d(beta)
        gamma = np.atleast_1d(gamma)

        if np.any(beta == 0.0):
            raise ValueError("beta cannot have a zero value")

        Pbn = self.project_node_to_boundary_node
        Pbf = self.project_face_to_boundary_face

        n_boundary_faces = Pbf.shape[0]
        n_boundary_nodes = Pbn.shape[0]

        if len(alpha) == 1:
            if len(beta) != 1:
                alpha = np.full(len(beta), alpha[0])
            elif len(gamma) != 1:
                alpha = np.full(len(gamma), alpha[0])
            else:
                alpha = np.full(n_boundary_faces, alpha[0])
        if len(beta) == 1:
            if len(alpha) != 1:
                beta = np.full(len(alpha), beta[0])
        if len(gamma) == 1:
            if len(alpha) != 1:
                gamma = np.full(len(alpha), gamma[0])

        if len(alpha) != len(beta) or len(beta) != len(gamma):
            raise ValueError("alpha, beta, and gamma must have the same length")

        if len(alpha) not in [n_boundary_faces, n_boundary_nodes]:
            raise ValueError(
                "The arrays must be of length n_boundary_faces or n_boundary_nodes"
            )

        AveN2F = self.average_node_to_face
        boundary_areas = Pbf @ self.face_areas
        AveBN2Bf = Pbf @ AveN2F @ Pbn.T

        # at the boundary, we have that u dot n = (gamma - alpha * phi)/beta
        if len(alpha) == n_boundary_faces:
            if gamma.ndim == 2:
                b = Pbn.T @ (
                    AveBN2Bf.T @ (gamma / beta[:, None] * boundary_areas[:, None])
                )
            else:
                b = Pbn.T @ (AveBN2Bf.T @ (gamma / beta * boundary_areas))
            B = sp.diags(Pbn.T @ (AveBN2Bf.T @ (-alpha / beta * boundary_areas)))
        else:
            if gamma.ndim == 2:
                b = Pbn.T @ (
                    gamma / beta[:, None] * (AveBN2Bf.T @ boundary_areas)[:, None]
                )
            else:
                b = Pbn.T @ (gamma / beta * (AveBN2Bf.T @ boundary_areas))
            B = sp.diags(Pbn.T @ (-alpha / beta * (AveBN2Bf.T @ boundary_areas)))
        return B, b

    ###########################################################################
    #                                                                         #
    #                                Cell Grad                                #
    #                                                                         #
    ###########################################################################

    _cell_gradient_BC_list = "neumann"

    def set_cell_gradient_BC(self, BC):
        """
        Function that sets the boundary conditions for cell-centred derivative
        operators.

        Examples
        --------
        ..code:: python

            # Neumann in all directions
            BC = 'neumann'

            # 3D, Dirichlet in y Neumann else
            BC = ['neumann', 'dirichlet', 'neumann']

            # 3D, Neumann in x on bottom of domain,  Dirichlet else
            BC = [['neumann', 'dirichlet'], 'dirichlet', 'dirichlet']
        """

        if isinstance(BC, str):
            BC = [BC] * self.dim
        if isinstance(BC, list):
            if len(BC) != self.dim:
                raise ValueError("BC list must be the size of your mesh")
        else:
            raise TypeError("BC must be a str or a list.")

        for i, bc_i in enumerate(BC):
            BC[i] = _validate_BC(bc_i)

        # ensure we create a new gradient next time we call it
        self._cell_gradient = None
        self._cell_gradient_BC = None
        self._cell_gradient_BC_list = BC
        return BC

    @property
    def stencil_cell_gradient_x(self):
        # TODO: remove this hard-coding
        BC = ["neumann", "neumann"]
        if self.dim == 1:
            G1 = _ddxCellGrad(self.shape_cells[0], BC)
        elif self.dim == 2:
            G1 = sp.kron(
                speye(self.shape_cells[1]), _ddxCellGrad(self.shape_cells[0], BC)
            )
        elif self.dim == 3:
            G1 = kron3(
                speye(self.shape_cells[2]),
                speye(self.shape_cells[1]),
                _ddxCellGrad(self.shape_cells[0], BC),
            )
        return G1

    @property
    def stencil_cell_gradient_y(self):
        if self.dim < 2:
            return None
        BC = ["neumann", "neumann"]  # TODO: remove this hard-coding
        n = self.vnC
        if self.dim == 2:
            G2 = sp.kron(_ddxCellGrad(n[1], BC), speye(n[0]))
        elif self.dim == 3:
            G2 = kron3(speye(n[2]), _ddxCellGrad(n[1], BC), speye(n[0]))
        return G2

    @property
    def stencil_cell_gradient_z(self):
        if self.dim < 3:
            return None
        BC = ["neumann", "neumann"]  # TODO: remove this hard-coding
        n = self.vnC
        G3 = kron3(_ddxCellGrad(n[2], BC), speye(n[1]), speye(n[0]))
        return G3

    @property
    def stencil_cell_gradient(self):
        BC = self.set_cell_gradient_BC(self._cell_gradient_BC_list)
        if self.dim == 1:
            G = _ddxCellGrad(self.shape_cells[0], BC[0])
        elif self.dim == 2:
            G1 = sp.kron(
                speye(self.shape_cells[1]), _ddxCellGrad(self.shape_cells[0], BC[0])
            )
            G2 = sp.kron(
                _ddxCellGrad(self.shape_cells[1], BC[1]), speye(self.shape_cells[0])
            )
            G = sp.vstack((G1, G2), format="csr")
        elif self.dim == 3:
            G1 = kron3(
                speye(self.shape_cells[2]),
                speye(self.shape_cells[1]),
                _ddxCellGrad(self.shape_cells[0], BC[0]),
            )
            G2 = kron3(
                speye(self.shape_cells[2]),
                _ddxCellGrad(self.shape_cells[1], BC[1]),
                speye(self.shape_cells[0]),
            )
            G3 = kron3(
                _ddxCellGrad(self.shape_cells[2], BC[2]),
                speye(self.shape_cells[1]),
                speye(self.shape_cells[0]),
            )
            G = sp.vstack((G1, G2, G3), format="csr")
        return G

    @property
    def cell_gradient(self):
        """
        The cell centered Gradient, takes you to cell faces.
        """
        if getattr(self, "_cell_gradient", None) is None:
            G = self.stencil_cell_gradient
            S = self.face_areas  # Compute areas of cell faces & volumes
            V = (
                self.aveCC2F * self.cell_volumes
            )  # Average volume between adjacent cells
            self._cell_gradient = sdiag(S / V) * G
        return self._cell_gradient

    def cell_gradient_weak_form_robin(self, alpha=1.0, beta=0.0, gamma=0.0):
        r"""Robin boundary condition for the weak formulation of the cell gradient

        This function returns the necessary parts for the weak form of the cell gradient
        operator to represent the Robin boundary conditions.

        The implementation assumes a ghost cell that mirrors the boundary cells across the boundary faces, with a
        piecewise linear approximation to the values at the ghost cell centers.

        The parameters can either be defined as a constant applied to the entire boundary,
        or as arrays that represent those values on the :meth:`discretize.base.BaseTensorMesh.boundary_faces`.

        The returned arrays represent the proper boundary conditions on a solution ``u``
        such that the inner product of the gradient of ``u`` with a test function y would
        be <``y, gradient*u``> ``= y.dot((-face_divergence.T*cell_volumes + A)*u + y.dot(b)``.

        The default values will produce a zero-dirichlet boundary condition.

        Parameters
        ----------
        alpha, beta : scalar or array_like
            Parameters for the Robin boundary condition. array_like must be defined on
            each boundary face.
        gamma: scalar or array_like
            right hand side boundary conditions. If this parameter is array like, it can
            be fed either a (n_boundary_faces,) shape array or an (n_boundary_faces, n_rhs)
            shape array if multiple systems have the same alpha and beta parameters.

        Returns
        -------
        A : scipy.sparse.csr_matrix
            Matrix to add to (-face_divergence.T * cell_volumes)
        b : numpy.ndarray
            Array to add to the result of the (-face_divergence.T * cell_volumes + A) @ u.

        Examples
        --------
        We first create a very simple 2D tensor mesh on the [0, 1] boundary:

        >>> import matplotlib.pyplot as plt
        >>> import scipy.sparse as sp
        >>> import discretize
        >>> mesh = discretize.TensorMesh([32, 32])

        Define the `alpha`, `beta`, and `gamma` parameters for a zero - Dirichlet
        condition on the boundary, this corresponds to setting:

        >>> alpha = 1.0
        >>> beta = 0.0
        >>> gamma = 0.0
        >>> A, b = mesh.cell_gradient_weak_form_robin(alpha, beta, gamma)

        We can then represent the operation of taking the weak form of the gradient of a
        function defined on cell centers with appropriate robin boundary conditions as:

        >>> V = sp.diags(mesh.cell_volumes)
        >>> D = mesh.face_divergence
        >>> phi = np.sin(np.pi * mesh.cell_centers[:, 0]) * np.sin(np.pi * mesh.cell_centers[:, 1])
        >>> phi_grad = (-D.T @ V + A) @ phi + b

        Notes
        -----
        The weak form is obtained by multiplying the gradient by a (piecewise-constant)
        test function, and integrating over the cell, i.e.

        .. math:: \int_V \vec{y} \cdot \nabla u \partial V
            :label: weak_form

        This equation can be transformed to reduce the differentiability requirement on `u`
        to be,

        .. math:: -\int_V u (\nabla \cdot \vec{y}) \partial V + \int_{dV} u \vec{y} \partial V.
            :label: transformed

        The first term in equation :eq:transformed is constructed using the matrix
        operators defined on this mesh as ``D`` = :meth:`discretize.operators.DiffOperators.face_divergence` and
        ``V``, a diagonal matrix of :meth:`discretize.base.BaseMesh.cell_volumes`, as

        .. math:: -(D*y)^T*V*u.

        This function returns the necessary matrices to complete the transformation of
        equation :eq:transformed. The second part of equation :eq:transformed becomes,

        .. math:: \int_V \nabla \cdot (\phi u) \partial V = \int_{\partial\Omega} \phi\vec{u}\cdot\hat{n} \partial a
            :label: boundary_conditions

        which is then approximated with the matrices returned here such that the full
        form of the weak formulation in a discrete form would be.

        .. math:: y^T(-D^T V + B)u + y^Tb
        """

        # get length between boundary cell_centers and boundary_faces
        Pf = self.project_face_to_boundary_face
        aveC2BF = Pf @ self.average_cell_to_face
        # distance from cell centers to ghost point on boundary faces
        if self.dim == 1:
            h = np.abs(self.boundary_faces - aveC2BF @ self.cell_centers)
        else:
            h = np.linalg.norm(
                self.boundary_faces - aveC2BF @ self.cell_centers, axis=1
            )

        # for the ghost point u_k = a*u_i + b where
        a = beta / h / (alpha + beta / h)
        A = sp.diags(a) @ aveC2BF

        gamma = np.asarray(gamma)
        if gamma.ndim > 1:
            b = (gamma) / (alpha + beta / h)[:, None]
        else:
            b = (gamma) / (alpha + beta / h)

        # value at boundary = A*cells + b
        M = self.boundary_face_scalar_integral
        A = M @ A
        b = M @ b

        return A, b

    @property
    def cell_gradient_BC(self):
        """
        The cell centered Gradient boundary condition matrix
        """
        if getattr(self, "_cell_gradient_BC", None) is None:
            BC = self.set_cell_gradient_BC(self._cell_gradient_BC_list)
            n = self.vnC
            if self.dim == 1:
                G = _ddxCellGradBC(n[0], BC[0])
            elif self.dim == 2:
                G1 = sp.kron(speye(n[1]), _ddxCellGradBC(n[0], BC[0]))
                G2 = sp.kron(_ddxCellGradBC(n[1], BC[1]), speye(n[0]))
                G = sp.block_diag((G1, G2), format="csr")
            elif self.dim == 3:
                G1 = kron3(speye(n[2]), speye(n[1]), _ddxCellGradBC(n[0], BC[0]))
                G2 = kron3(speye(n[2]), _ddxCellGradBC(n[1], BC[1]), speye(n[0]))
                G3 = kron3(_ddxCellGradBC(n[2], BC[2]), speye(n[1]), speye(n[0]))
                G = sp.block_diag((G1, G2, G3), format="csr")
            # Compute areas of cell faces & volumes
            S = self.face_areas
            V = (
                self.aveCC2F * self.cell_volumes
            )  # Average volume between adjacent cells
            self._cell_gradient_BC = sdiag(S / V) * G
        return self._cell_gradient_BC

    @property
    def cell_gradient_x(self):
        """
        Cell centered Gradient in the x dimension. Has neumann boundary
        conditions.
        """
        if getattr(self, "_cell_gradient_x", None) is None:
            G1 = self.stencil_cell_gradient_x
            # Compute areas of cell faces & volumes
            V = self.aveCC2F * self.cell_volumes
            L = self.reshape(self.face_areas / V, "F", "Fx", "V")
            self._cell_gradient_x = sdiag(L) * G1
        return self._cell_gradient_x

    @property
    def cell_gradient_y(self):
        if self.dim < 2:
            return None
        if getattr(self, "_cell_gradient_y", None) is None:
            G2 = self.stencil_cell_gradient_y
            # Compute areas of cell faces & volumes
            V = self.aveCC2F * self.cell_volumes
            L = self.reshape(self.face_areas / V, "F", "Fy", "V")
            self._cell_gradient_y = sdiag(L) * G2
        return self._cell_gradient_y

    @property
    def cell_gradient_z(self):
        """
        Cell centered Gradient in the x dimension. Has neumann boundary
        conditions.
        """
        if self.dim < 3:
            return None
        if getattr(self, "_cell_gradient_z", None) is None:
            G3 = self.stencil_cell_gradient_z
            # Compute areas of cell faces & volumes
            V = self.aveCC2F * self.cell_volumes
            L = self.reshape(self.face_areas / V, "F", "Fz", "V")
            self._cell_gradient_z = sdiag(L) * G3
        return self._cell_gradient_z

    ###########################################################################
    #                                                                         #
    #                                Edge Curl                                #
    #                                                                         #
    ###########################################################################

    @property
    def _edge_x_curl_stencil(self):
        n = self.vnC  # The number of cell centers in each direction

        D32 = kron3(ddx(n[2]), speye(n[1]), speye(n[0] + 1))
        D23 = kron3(speye(n[2]), ddx(n[1]), speye(n[0] + 1))
        # O1 = spzeros(np.shape(D32)[0], np.shape(D31)[1])
        O1 = spzeros((n[0] + 1) * n[1] * n[2], n[0] * (n[1] + 1) * (n[2] + 1))

        return sp.hstack((O1, -D32, D23))

    @property
    def _edge_y_curl_stencil(self):
        n = self.vnC  # The number of cell centers in each direction

        D31 = kron3(ddx(n[2]), speye(n[1] + 1), speye(n[0]))
        D13 = kron3(speye(n[2]), speye(n[1] + 1), ddx(n[0]))
        # O2 = spzeros(np.shape(D31)[0], np.shape(D32)[1])
        O2 = spzeros(n[0] * (n[1] + 1) * n[2], (n[0] + 1) * n[1] * (n[2] + 1))

        return sp.hstack((D31, O2, -D13))

    @property
    def _edge_z_curl_stencil(self):
        n = self.vnC  # The number of cell centers in each direction

        D21 = kron3(speye(n[2] + 1), ddx(n[1]), speye(n[0]))
        D12 = kron3(speye(n[2] + 1), speye(n[1]), ddx(n[0]))
        # O3 = spzeros(np.shape(D21)[0], np.shape(D13)[1])
        O3 = spzeros(n[0] * n[1] * (n[2] + 1), (n[0] + 1) * (n[1] + 1) * n[2])

        return sp.hstack((-D21, D12, O3))

    @property
    def _edge_curl_stencil(self):
        if self.dim <= 1:
            raise NotImplementedError("Edge Curl only programed for 2 or 3D.")

        # Compute divergence operator on faces
        if self.dim == 2:
            n = self.vnC  # The number of cell centers in each direction

            D21 = sp.kron(ddx(n[1]), speye(n[0]))
            D12 = sp.kron(speye(n[1]), ddx(n[0]))
            C = sp.hstack((-D21, D12), format="csr")
            return C

        elif self.dim == 3:

            # D32 = kron3(ddx(n[2]), speye(n[1]), speye(n[0]+1))
            # D23 = kron3(speye(n[2]), ddx(n[1]), speye(n[0]+1))
            # D31 = kron3(ddx(n[2]), speye(n[1]+1), speye(n[0]))
            # D13 = kron3(speye(n[2]), speye(n[1]+1), ddx(n[0]))
            # D21 = kron3(speye(n[2]+1), ddx(n[1]), speye(n[0]))
            # D12 = kron3(speye(n[2]+1), speye(n[1]), ddx(n[0]))

            # O1 = spzeros(np.shape(D32)[0], np.shape(D31)[1])
            # O2 = spzeros(np.shape(D31)[0], np.shape(D32)[1])
            # O3 = spzeros(np.shape(D21)[0], np.shape(D13)[1])

            # C = sp.vstack((sp.hstack((O1, -D32, D23)),
            #                sp.hstack((D31, O2, -D13)),
            #                sp.hstack((-D21, D12, O3))), format="csr")

            C = sp.vstack(
                (
                    self._edge_x_curl_stencil,
                    self._edge_y_curl_stencil,
                    self._edge_z_curl_stencil,
                ),
                format="csr",
            )

            return C

    @property
    def edge_curl(self):
        """
        Construct the 3D curl operator.
        """
        L = self.edge_lengths  # Compute lengths of cell edges
        S = self.face_areas  # Compute areas of cell faces

        if getattr(self, "_edge_curl", None) is None:

            if self.dim <= 1:
                raise NotImplementedError("Edge Curl only programed for 2 or 3D.")

            if self.dim == 2:
                self._edge_curl = self._edge_curl_stencil * sdiag(1 / S)
            elif self.dim == 3:
                self._edge_curl = sdiag(1 / S) * (self._edge_curl_stencil * sdiag(L))

        return self._edge_curl

    @property
    def boundary_face_scalar_integral(self):
        r"""Represents the operation of integrating a scalar function on the boundary

        This matrix represents the boundary surface integral of a scalar function
        multiplied with a finite volume test function on the mesh.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse matrix of shape (n_faces, n_boundary_faces)

        Notes
        -----
        The integral we are representing on the boundary of the mesh is

        .. math:: \int_{\Omega} u\vec{w} \cdot \hat{n} \partial \Omega

        In discrete form this is:

        .. math:: w^T * P * u_b

        where `w` is defined on all faces, and `u_b` is defined on boundary faces.
        """
        if self.dim == 1:
            return sp.csr_matrix(
                ([-1, 1], ([0, self.n_faces_x - 1], [0, 1])), shape=(self.n_faces_x, 2)
            )
        P = self.project_face_to_boundary_face

        w_h_dot_normal = np.sum(
            (P @ self.face_normals) * self.boundary_face_outward_normals, axis=-1
        )
        A = sp.diags(self.face_areas) @ P.T @ sp.diags(w_h_dot_normal)
        return A

    @property
    def boundary_edge_vector_integral(self):
        r"""Represents the operation of integrating a vector function on the boundary

        This matrix represents the boundary surface integral of a vector function
        multiplied with a finite volume test function on the mesh.

        In 1D and 2D, the operation assumes that the right array contains only a single
        component of the vector ``u``. In 3D, however, we must assume that ``u`` will
        contain each of the three vector components, and it must be ordered as,
        ``[edges_1_x, ... ,edge_N_x, edge_1_y, ..., edge_N_y, edge_1_z, ..., edge_N_z]``
        , where ``N`` is the number of boundary edges.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse matrix of shape (n_edges, n_boundary_edges) for 1D or 2D mesh,
            (n_edges, 3*n_boundary_edges) for a 3D mesh.

        Notes
        -----
        The integral we are representing on the boundary of the mesh is

        .. math:: \int_{\Omega} \vec{w} \cdot (\vec{u} \times \hat{n}) \partial \Omega

        In discrete form this is:

        .. math:: w^T * P * u_b

        where `w` is defined on all edges, and `u_b` is all three components defined on
        boundary edges.
        """
        Pe = self.project_edge_to_boundary_edge
        Pf = self.project_face_to_boundary_face
        dA = self.boundary_face_outward_normals * (Pf @ self.face_areas)[:, None]
        w = Pe @ self.edge_tangents

        n_boundary_edges = len(w)

        Av = Pf @ self.average_edge_to_face_vector @ Pe.T

        w_cross_n = np.cross(-w, Av.T @ dA)

        if self.dim == 2:
            return Pe.T @ sp.diags(w_cross_n, format="csr")
        return Pe.T @ sp.diags(
            w_cross_n.T,
            n_boundary_edges * np.arange(3),
            shape=(n_boundary_edges, 3 * n_boundary_edges),
        )

    @property
    def boundary_node_vector_integral(self):
        r"""Represents the operation of integrating a vector function dotted with the boundary normal

        This matrix represents the boundary surface integral of a vector function
        dotted with the boundary normal and multiplied with a scalar finite volume
        test function on the mesh.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse matrix of shape (n_nodes, ndim * n_boundary_nodes).

        Notes
        -----
        The integral we are representing on the boundary of the mesh is

        .. math:: \int_{\Omega} (w \vec{u}) \cdot \hat{n} \partial \Omega

        In discrete form this is:

        .. math:: w^T * P * u_b

        where `w` is defined on all nodes, and `u_b` is all three components defined on
        boundary nodes.
        """
        if self.dim == 1:
            return sp.csr_matrix(
                ([-1, 1], ([0, self.shape_nodes[0] - 1], [0, 1])),
                shape=(self.shape_nodes[0], 2),
            )
        Pn = self.project_node_to_boundary_node
        Pf = self.project_face_to_boundary_face
        n_boundary_nodes = Pn.shape[0]

        dA = self.boundary_face_outward_normals * (Pf @ self.face_areas)[:, None]

        Av = Pf @ self.average_node_to_face @ Pn.T

        u_dot_ds = Av.T @ dA
        diags = u_dot_ds.T
        offsets = n_boundary_nodes * np.arange(self.dim)

        return Pn.T @ sp.diags(
            diags, offsets, shape=(n_boundary_nodes, self.dim * n_boundary_nodes)
        )

    def get_BC_projections(self, BC, discretization="CC"):
        """
        The weak form boundary condition projection matrices.

        Examples
        --------

        .. code:: python

            # Neumann in all directions
            BC = 'neumann'

            # 3D, Dirichlet in y Neumann else
            BC = ['neumann', 'dirichlet', 'neumann']

            # 3D, Neumann in x on bottom of domain, Dirichlet else
            BC = [['neumann', 'dirichlet'], 'dirichlet', 'dirichlet']
        """

        if discretization != "CC":
            raise NotImplementedError(
                "Boundary conditions only implemented" "for CC discretization."
            )

        if isinstance(BC, str):
            BC = [BC for _ in self.vnC]  # Repeat the str self.dim times
        elif isinstance(BC, list):
            if len(BC) != self.dim:
                raise ValueError("BC list must be the size of your mesh")
        else:
            raise TypeError("BC must be a str or a list.")

        for i, bc_i in enumerate(BC):
            BC[i] = _validate_BC(bc_i)

        def projDirichlet(n, bc):
            bc = _validate_BC(bc)
            ij = ([0, n], [0, 1])
            vals = [0, 0]
            if bc[0] == "dirichlet":
                vals[0] = -1
            if bc[1] == "dirichlet":
                vals[1] = 1
            return sp.csr_matrix((vals, ij), shape=(n + 1, 2))

        def projNeumannIn(n, bc):
            bc = _validate_BC(bc)
            P = sp.identity(n + 1).tocsr()
            if bc[0] == "neumann":
                P = P[1:, :]
            if bc[1] == "neumann":
                P = P[:-1, :]
            return P

        def projNeumannOut(n, bc):
            bc = _validate_BC(bc)
            ij = ([0, 1], [0, n])
            vals = [0, 0]
            if bc[0] == "neumann":
                vals[0] = 1
            if bc[1] == "neumann":
                vals[1] = 1
            return sp.csr_matrix((vals, ij), shape=(2, n + 1))

        n = self.vnC
        indF = self.face_boundary_indices
        if self.dim == 1:
            Pbc = projDirichlet(n[0], BC[0])
            indF = indF[0] | indF[1]
            Pbc = Pbc * sdiag(self.face_areas[indF])

            Pin = projNeumannIn(n[0], BC[0])

            Pout = projNeumannOut(n[0], BC[0])

        elif self.dim == 2:
            Pbc1 = sp.kron(speye(n[1]), projDirichlet(n[0], BC[0]))
            Pbc2 = sp.kron(projDirichlet(n[1], BC[1]), speye(n[0]))
            Pbc = sp.block_diag((Pbc1, Pbc2), format="csr")
            indF = np.r_[(indF[0] | indF[1]), (indF[2] | indF[3])]
            Pbc = Pbc * sdiag(self.face_areas[indF])

            P1 = sp.kron(speye(n[1]), projNeumannIn(n[0], BC[0]))
            P2 = sp.kron(projNeumannIn(n[1], BC[1]), speye(n[0]))
            Pin = sp.block_diag((P1, P2), format="csr")

            P1 = sp.kron(speye(n[1]), projNeumannOut(n[0], BC[0]))
            P2 = sp.kron(projNeumannOut(n[1], BC[1]), speye(n[0]))
            Pout = sp.block_diag((P1, P2), format="csr")

        elif self.dim == 3:
            Pbc1 = kron3(speye(n[2]), speye(n[1]), projDirichlet(n[0], BC[0]))
            Pbc2 = kron3(speye(n[2]), projDirichlet(n[1], BC[1]), speye(n[0]))
            Pbc3 = kron3(projDirichlet(n[2], BC[2]), speye(n[1]), speye(n[0]))
            Pbc = sp.block_diag((Pbc1, Pbc2, Pbc3), format="csr")
            indF = np.r_[(indF[0] | indF[1]), (indF[2] | indF[3]), (indF[4] | indF[5])]
            Pbc = Pbc * sdiag(self.face_areas[indF])

            P1 = kron3(speye(n[2]), speye(n[1]), projNeumannIn(n[0], BC[0]))
            P2 = kron3(speye(n[2]), projNeumannIn(n[1], BC[1]), speye(n[0]))
            P3 = kron3(projNeumannIn(n[2], BC[2]), speye(n[1]), speye(n[0]))
            Pin = sp.block_diag((P1, P2, P3), format="csr")

            P1 = kron3(speye(n[2]), speye(n[1]), projNeumannOut(n[0], BC[0]))
            P2 = kron3(speye(n[2]), projNeumannOut(n[1], BC[1]), speye(n[0]))
            P3 = kron3(projNeumannOut(n[2], BC[2]), speye(n[1]), speye(n[0]))
            Pout = sp.block_diag((P1, P2, P3), format="csr")

        return Pbc, Pin, Pout

    def get_BC_projections_simple(self, discretization="CC"):
        """The weak form boundary condition projection matrices
        when mixed boundary condition is used
        """

        if discretization != "CC":
            raise NotImplementedError(
                "Boundary conditions only implemented" "for CC discretization."
            )

        def projBC(n):
            ij = ([0, n], [0, 1])
            vals = [0, 0]
            vals[0] = 1
            vals[1] = 1
            return sp.csr_matrix((vals, ij), shape=(n + 1, 2))

        def projDirichlet(n, bc):
            bc = _validate_BC(bc)
            ij = ([0, n], [0, 1])
            vals = [0, 0]
            if bc[0] == "dirichlet":
                vals[0] = -1
            if bc[1] == "dirichlet":
                vals[1] = 1
            return sp.csr_matrix((vals, ij), shape=(n + 1, 2))

        BC = [
            ["dirichlet", "dirichlet"],
            ["dirichlet", "dirichlet"],
            ["dirichlet", "dirichlet"],
        ]
        n = self.vnC
        indF = self.face_boundary_indices

        if self.dim == 1:
            Pbc = projDirichlet(n[0], BC[0])
            B = projBC(n[0])
            indF = indF[0] | indF[1]
            Pbc = Pbc * sdiag(self.face_areas[indF])

        elif self.dim == 2:
            Pbc1 = sp.kron(speye(n[1]), projDirichlet(n[0], BC[0]))
            Pbc2 = sp.kron(projDirichlet(n[1], BC[1]), speye(n[0]))
            Pbc = sp.block_diag((Pbc1, Pbc2), format="csr")
            B1 = sp.kron(speye(n[1]), projBC(n[0]))
            B2 = sp.kron(projBC(n[1]), speye(n[0]))
            B = sp.block_diag((B1, B2), format="csr")
            indF = np.r_[(indF[0] | indF[1]), (indF[2] | indF[3])]
            Pbc = Pbc * sdiag(self.face_areas[indF])

        elif self.dim == 3:
            Pbc1 = kron3(speye(n[2]), speye(n[1]), projDirichlet(n[0], BC[0]))
            Pbc2 = kron3(speye(n[2]), projDirichlet(n[1], BC[1]), speye(n[0]))
            Pbc3 = kron3(projDirichlet(n[2], BC[2]), speye(n[1]), speye(n[0]))
            Pbc = sp.block_diag((Pbc1, Pbc2, Pbc3), format="csr")
            B1 = kron3(speye(n[2]), speye(n[1]), projBC(n[0]))
            B2 = kron3(speye(n[2]), projBC(n[1]), speye(n[0]))
            B3 = kron3(projBC(n[2]), speye(n[1]), speye(n[0]))
            B = sp.block_diag((B1, B2, B3), format="csr")
            indF = np.r_[(indF[0] | indF[1]), (indF[2] | indF[3]), (indF[4] | indF[5])]
            Pbc = Pbc * sdiag(self.face_areas[indF])

        return Pbc, B.T

    ###########################################################################
    #                                                                         #
    #                                Averaging                                #
    #                                                                         #
    ###########################################################################

    @property
    def average_face_to_cell(self):
        "Construct the averaging operator on cell faces to cell centers."
        if getattr(self, "_average_face_to_cell", None) is None:
            if self.dim == 1:
                self._average_face_to_cell = self.aveFx2CC
            elif self.dim == 2:
                self._average_face_to_cell = (0.5) * sp.hstack(
                    (self.aveFx2CC, self.aveFy2CC), format="csr"
                )
            elif self.dim == 3:
                self._average_face_to_cell = (1.0 / 3.0) * sp.hstack(
                    (self.aveFx2CC, self.aveFy2CC, self.aveFz2CC), format="csr"
                )
        return self._average_face_to_cell

    @property
    def average_face_to_cell_vector(self):
        "Construct the averaging operator on cell faces to cell centers."
        if getattr(self, "_average_face_to_cell_vector", None) is None:
            if self.dim == 1:
                self._average_face_to_cell_vector = self.aveFx2CC
            elif self.dim == 2:
                self._average_face_to_cell_vector = sp.block_diag(
                    (self.aveFx2CC, self.aveFy2CC), format="csr"
                )
            elif self.dim == 3:
                self._average_face_to_cell_vector = sp.block_diag(
                    (self.aveFx2CC, self.aveFy2CC, self.aveFz2CC), format="csr"
                )
        return self._average_face_to_cell_vector

    @property
    def average_face_x_to_cell(self):
        """
        Construct the averaging operator on cell faces in the x direction to
        cell centers.
        """

        if getattr(self, "_average_face_x_to_cell", None) is None:
            n = self.vnC
            if self.dim == 1:
                self._average_face_x_to_cell = av(n[0])
            elif self.dim == 2:
                self._average_face_x_to_cell = sp.kron(speye(n[1]), av(n[0]))
            elif self.dim == 3:
                self._average_face_x_to_cell = kron3(speye(n[2]), speye(n[1]), av(n[0]))
        return self._average_face_x_to_cell

    @property
    def average_face_y_to_cell(self):
        """
        Construct the averaging operator on cell faces in the y direction to
        cell centers.
        """
        if self.dim < 2:
            return None
        if getattr(self, "_average_face_y_to_cell", None) is None:
            n = self.vnC
            if self.dim == 2:
                self._average_face_y_to_cell = sp.kron(av(n[1]), speye(n[0]))
            elif self.dim == 3:
                self._average_face_y_to_cell = kron3(speye(n[2]), av(n[1]), speye(n[0]))
        return self._average_face_y_to_cell

    @property
    def average_face_z_to_cell(self):
        """
        Construct the averaging operator on cell faces in the z direction to
        cell centers.
        """
        if self.dim < 3:
            return None
        if getattr(self, "_average_face_z_to_cell", None) is None:
            n = self.vnC
            if self.dim == 3:
                self._average_face_z_to_cell = kron3(av(n[2]), speye(n[1]), speye(n[0]))
        return self._average_face_z_to_cell

    @property
    def average_cell_to_face(self):
        "Construct the averaging operator on cell centers to faces."
        if getattr(self, "_average_cell_to_face", None) is None:
            if self.dim == 1:
                self._average_cell_to_face = av_extrap(self.shape_cells[0])
            elif self.dim == 2:
                self._average_cell_to_face = sp.vstack(
                    (
                        sp.kron(
                            speye(self.shape_cells[1]), av_extrap(self.shape_cells[0])
                        ),
                        sp.kron(
                            av_extrap(self.shape_cells[1]), speye(self.shape_cells[0])
                        ),
                    ),
                    format="csr",
                )
            elif self.dim == 3:
                self._average_cell_to_face = sp.vstack(
                    (
                        kron3(
                            speye(self.shape_cells[2]),
                            speye(self.shape_cells[1]),
                            av_extrap(self.shape_cells[0]),
                        ),
                        kron3(
                            speye(self.shape_cells[2]),
                            av_extrap(self.shape_cells[1]),
                            speye(self.shape_cells[0]),
                        ),
                        kron3(
                            av_extrap(self.shape_cells[2]),
                            speye(self.shape_cells[1]),
                            speye(self.shape_cells[0]),
                        ),
                    ),
                    format="csr",
                )
        return self._average_cell_to_face

    @property
    def average_cell_vector_to_face(self):
        """
        Construct the averaging operator on cell centers to
        faces as a vector.
        """
        if getattr(self, "_average_cell_vector_to_face", None) is None:
            if self.dim == 1:
                self._average_cell_vector_to_face = self.aveCC2F
            elif self.dim == 2:
                aveCCV2Fx = sp.kron(
                    speye(self.shape_cells[1]), av_extrap(self.shape_cells[0])
                )
                aveCC2VFy = sp.kron(
                    av_extrap(self.shape_cells[1]), speye(self.shape_cells[0])
                )
                self._average_cell_vector_to_face = sp.block_diag(
                    (aveCCV2Fx, aveCC2VFy), format="csr"
                )
            elif self.dim == 3:
                aveCCV2Fx = kron3(
                    speye(self.shape_cells[2]),
                    speye(self.shape_cells[1]),
                    av_extrap(self.shape_cells[0]),
                )
                aveCC2VFy = kron3(
                    speye(self.shape_cells[2]),
                    av_extrap(self.shape_cells[1]),
                    speye(self.shape_cells[0]),
                )
                aveCC2BFz = kron3(
                    av_extrap(self.shape_cells[2]),
                    speye(self.shape_cells[1]),
                    speye(self.shape_cells[0]),
                )
                self._average_cell_vector_to_face = sp.block_diag(
                    (aveCCV2Fx, aveCC2VFy, aveCC2BFz), format="csr"
                )
        return self._average_cell_vector_to_face

    @property
    def average_cell_to_edge(self):
        if getattr(self, "_average_cell_to_edge", None) is None:
            n = self.shape_cells
            if self.dim == 1:
                avg = sp.eye(n[0])
            elif self.dim == 2:
                avg = sp.vstack(
                    (
                        sp.kron(av_extrap(n[1]), speye(n[0])),
                        sp.kron(speye(n[1]), av_extrap(n[0])),
                    ),
                    format="csr",
                )
            elif self.dim == 3:
                avg = sp.vstack(
                    (
                        kron3(av_extrap(n[2]), av_extrap(n[1]), speye(n[0])),
                        kron3(av_extrap(n[2]), speye(n[1]), av_extrap(n[0])),
                        kron3(speye(n[2]), av_extrap(n[1]), av_extrap(n[0])),
                    ),
                    format="csr",
                )
            self._average_cell_to_edge = avg
        return self._average_cell_to_edge

    @property
    def average_edge_to_cell(self):
        "Construct the averaging operator on cell edges to cell centers."
        if getattr(self, "_average_edge_to_cell", None) is None:
            if self.dim == 1:
                self._avE2CC = self.aveEx2CC
            elif self.dim == 2:
                self._avE2CC = 0.5 * sp.hstack(
                    (self.aveEx2CC, self.aveEy2CC), format="csr"
                )
            elif self.dim == 3:
                self._avE2CC = (1.0 / 3) * sp.hstack(
                    (self.aveEx2CC, self.aveEy2CC, self.aveEz2CC), format="csr"
                )
        return self._avE2CC

    @property
    def average_edge_to_cell_vector(self):
        "Construct the averaging operator on cell edges to cell centers."
        if getattr(self, "_average_edge_to_cell_vector", None) is None:
            if self.dim == 1:
                self._average_edge_to_cell_vector = self.aveEx2CC
            elif self.dim == 2:
                self._average_edge_to_cell_vector = sp.block_diag(
                    (self.aveEx2CC, self.aveEy2CC), format="csr"
                )
            elif self.dim == 3:
                self._average_edge_to_cell_vector = sp.block_diag(
                    (self.aveEx2CC, self.aveEy2CC, self.aveEz2CC), format="csr"
                )
        return self._average_edge_to_cell_vector

    @property
    def average_edge_x_to_cell(self):
        """
        Construct the averaging operator on cell edges in the x direction to
        cell centers.
        """
        if getattr(self, "_average_edge_x_to_cell", None) is None:
            # The number of cell centers in each direction
            n = self.vnC
            if self.dim == 1:
                self._average_edge_x_to_cell = speye(n[0])
            elif self.dim == 2:
                self._average_edge_x_to_cell = sp.kron(av(n[1]), speye(n[0]))
            elif self.dim == 3:
                self._average_edge_x_to_cell = kron3(av(n[2]), av(n[1]), speye(n[0]))
        return self._average_edge_x_to_cell

    @property
    def average_edge_y_to_cell(self):
        """
        Construct the averaging operator on cell edges in the y direction to
        cell centers.
        """
        if self.dim < 2:
            return None
        if getattr(self, "_average_edge_y_to_cell", None) is None:
            # The number of cell centers in each direction
            n = self.vnC
            if self.dim == 2:
                self._average_edge_y_to_cell = sp.kron(speye(n[1]), av(n[0]))
            elif self.dim == 3:
                self._average_edge_y_to_cell = kron3(av(n[2]), speye(n[1]), av(n[0]))
        return self._average_edge_y_to_cell

    @property
    def average_edge_z_to_cell(self):
        """
        Construct the averaging operator on cell edges in the z direction to
        cell centers.
        """
        if self.dim < 3:
            return None
        if getattr(self, "_average_edge_z_to_cell", None) is None:
            # The number of cell centers in each direction
            n = self.vnC
            if self.dim == 3:
                self._average_edge_z_to_cell = kron3(speye(n[2]), av(n[1]), av(n[0]))
        return self._average_edge_z_to_cell

    @property
    def average_edge_to_face_vector(self):
        if self.dim == 1:
            return self.average_cell_to_face
        elif self.dim == 2:
            return sp.diags(
                [1, 1],
                [-self.n_faces_x, self.n_faces_y],
                shape=(self.n_faces, self.n_edges),
            )
        n1, n2, n3 = self.shape_cells
        ex_to_fy = kron3(av(n3), speye(n2 + 1), speye(n1))
        ex_to_fz = kron3(speye(n3 + 1), av(n2), speye(n1))

        ey_to_fx = kron3(av(n3), speye(n2), speye(n1 + 1))
        ey_to_fz = kron3(speye(n3 + 1), speye(n2), av(n1))

        ez_to_fx = kron3(speye(n3), av(n2), speye(n1 + 1))
        ez_to_fy = kron3(speye(n3), speye(n2 + 1), av(n1))

        e_to_f = sp.bmat(
            [
                [None, ey_to_fx, ez_to_fx],
                [ex_to_fy, None, ez_to_fy],
                [ex_to_fz, ey_to_fz, None],
            ],
            format="csr",
        )
        return e_to_f

    @property
    def average_node_to_cell(self):
        "Construct the averaging operator on cell nodes to cell centers."
        if getattr(self, "_average_node_to_cell", None) is None:
            # The number of cell centers in each direction
            if self.dim == 1:
                self._average_node_to_cell = av(self.shape_cells[0])
            elif self.dim == 2:
                self._average_node_to_cell = sp.kron(
                    av(self.shape_cells[1]), av(self.shape_cells[0])
                ).tocsr()
            elif self.dim == 3:
                self._average_node_to_cell = kron3(
                    av(self.shape_cells[2]),
                    av(self.shape_cells[1]),
                    av(self.shape_cells[0]),
                ).tocsr()
        return self._average_node_to_cell

    @property
    def _average_node_to_edge_x(self):
        """
        Averaging operator on cell nodes to x-edges
        """
        if self.dim == 1:
            aveN2Ex = av(self.shape_cells[0])
        elif self.dim == 2:
            aveN2Ex = sp.kron(speye(self.shape_nodes[1]), av(self.shape_cells[0]))
        elif self.dim == 3:
            aveN2Ex = kron3(
                speye(self.shape_nodes[2]),
                speye(self.shape_nodes[1]),
                av(self.shape_cells[0]),
            )
        return aveN2Ex

    @property
    def _average_node_to_edge_y(self):
        """
        Averaging operator on cell nodes to y-edges
        """
        if self.dim == 1:
            return None
        elif self.dim == 2:
            aveN2Ey = sp.kron(av(self.shape_cells[1]), speye(self.shape_nodes[0]))
        elif self.dim == 3:
            aveN2Ey = kron3(
                speye(self.shape_nodes[2]),
                av(self.shape_cells[1]),
                speye(self.shape_nodes[0]),
            )
        return aveN2Ey

    @property
    def _average_node_to_edge_z(self):
        if self.dim == 1 or self.dim == 2:
            return None
        elif self.dim == 3:
            aveN2Ez = kron3(
                av(self.shape_cells[2]),
                speye(self.shape_nodes[1]),
                speye(self.shape_nodes[0]),
            )
        return aveN2Ez

    @property
    def average_node_to_edge(self):
        """
        Construct the averaging operator on cell nodes to cell edges, keeping
        each dimension separate.
        """
        if getattr(self, "_average_node_to_edge", None) is None:
            # The number of cell centers in each direction
            if self.dim == 1:
                self._average_node_to_edge = self._average_node_to_edge_x
            elif self.dim == 2:
                self._average_node_to_edge = sp.vstack(
                    (self._average_node_to_edge_x, self._average_node_to_edge_y),
                    format="csr",
                )
            elif self.dim == 3:
                self._average_node_to_edge = sp.vstack(
                    (
                        self._average_node_to_edge_x,
                        self._average_node_to_edge_y,
                        self._average_node_to_edge_z,
                    ),
                    format="csr",
                )
        return self._average_node_to_edge

    @property
    def _average_node_to_face_x(self):
        if self.dim == 1:
            aveN2Fx = speye(self.shape_nodes[0])
        elif self.dim == 2:
            aveN2Fx = sp.kron(av(self.shape_cells[1]), speye(self.shape_nodes[0]))
        elif self.dim == 3:
            aveN2Fx = kron3(
                av(self.shape_cells[2]),
                av(self.shape_cells[1]),
                speye(self.shape_nodes[0]),
            )
        return aveN2Fx

    @property
    def _average_node_to_face_y(self):
        if self.dim == 1:
            return None
        elif self.dim == 2:
            aveN2Fy = sp.kron(speye(self.shape_nodes[1]), av(self.shape_cells[0]))
        elif self.dim == 3:
            aveN2Fy = kron3(
                av(self.shape_cells[2]),
                speye(self.shape_nodes[1]),
                av(self.shape_cells[0]),
            )
        return aveN2Fy

    @property
    def _average_node_to_face_z(self):
        if self.dim == 1 or self.dim == 2:
            return None
        else:
            aveN2Fz = kron3(
                speye(self.shape_nodes[2]),
                av(self.shape_cells[1]),
                av(self.shape_cells[0]),
            )
        return aveN2Fz

    @property
    def average_node_to_face(self):
        """
        Construct the averaging operator on cell nodes to cell faces, keeping
        each dimension separate.
        """
        if getattr(self, "_average_node_to_face", None) is None:
            # The number of cell centers in each direction
            if self.dim == 1:
                self._average_node_to_face = self._average_node_to_face_x
            elif self.dim == 2:
                self._average_node_to_face = sp.vstack(
                    (self._average_node_to_face_x, self._average_node_to_face_y),
                    format="csr",
                )
            elif self.dim == 3:
                self._average_node_to_face = sp.vstack(
                    (
                        self._average_node_to_face_x,
                        self._average_node_to_face_y,
                        self._average_node_to_face_z,
                    ),
                    format="csr",
                )
        return self._average_node_to_face

    @property
    def project_face_to_boundary_face(self):
        """Projects values defined on all faces to the boundary faces

        Returns
        -------
        scipy.sparse.csr_matrix
            Projection matrix with shape (n_boundary_faces, n_faces)
        """
        # Simple matrix which projects the values of the faces onto the boundary faces
        # Can also be used to "select" the boundary faces

        # Create a matrix that projects all faces onto boundary faces
        # The below should work for a regular structured mesh
        is_b = make_boundary_bool(self.shape_faces_x, dir="x")
        if self.dim > 1:
            is_b = np.r_[is_b, make_boundary_bool(self.shape_faces_y, dir="y")]
        if self.dim == 3:
            is_b = np.r_[is_b, make_boundary_bool(self.shape_faces_z, dir="z")]
        return sp.eye(self.n_faces, format="csr")[is_b]

    @property
    def project_edge_to_boundary_edge(self):
        """Projects values defined on all edges to the boundary edges

        Returns
        -------
        scipy.sparse.csr_matrix
            Projection matrix with shape (n_boundary_edges, n_edges)
        """
        # Simple matrix which projects the values of the faces onto the boundary faces
        # Can also be used to "select" the boundary faces

        # Create a matrix that projects all edges onto boundary edges
        # The below should work for a regular structured mesh
        if self.dim == 1:
            return None  # No edges are on the boundary in 1D

        is_b = np.r_[
            make_boundary_bool(self.shape_edges_x, dir="yz"),
            make_boundary_bool(self.shape_edges_y, dir="xz"),
        ]
        if self.dim == 3:
            is_b = np.r_[is_b, make_boundary_bool(self.shape_edges_z, dir="xy")]
        return sp.eye(self.n_edges, format="csr")[is_b]

    @property
    def project_node_to_boundary_node(self):
        """Projects values defined on all edges to the boundary edges

        Returns
        -------
        scipy.sparse.csr_matrix
            Projection matrix with shape (n_boundary_nodes, n_nodes)
        """
        # Simple matrix which projects the values of the nodes onto the boundary nodes
        # Can also be used to "select" the boundary nodes

        # Create a matrix that projects all nodes onto boundary nodes
        # The below should work for a regular structured mesh

        is_b = make_boundary_bool(self.shape_nodes)
        return sp.eye(self.n_nodes, format="csr")[is_b]

    # DEPRECATED
    cellGrad = deprecate_property("cell_gradient", "cellGrad", removal_version="1.0.0")
    cellGradBC = deprecate_property(
        "cell_gradient_BC", "cellGradBC", removal_version="1.0.0"
    )
    cellGradx = deprecate_property(
        "cell_gradient_x", "cellGradx", removal_version="1.0.0"
    )
    cellGrady = deprecate_property(
        "cell_gradient_y", "cellGrady", removal_version="1.0.0"
    )
    cellGradz = deprecate_property(
        "cell_gradient_z", "cellGradz", removal_version="1.0.0"
    )
    nodalGrad = deprecate_property(
        "nodal_gradient", "nodalGrad", removal_version="1.0.0"
    )
    nodalLaplacian = deprecate_property(
        "nodal_laplacian", "nodalLaplacian", removal_version="1.0.0"
    )
    faceDiv = deprecate_property("face_divergence", "faceDiv", removal_version="1.0.0")
    faceDivx = deprecate_property(
        "face_x_divergence", "faceDivx", removal_version="1.0.0"
    )
    faceDivy = deprecate_property(
        "face_y_divergence", "faceDivy", removal_version="1.0.0"
    )
    faceDivz = deprecate_property(
        "face_z_divergence", "faceDivz", removal_version="1.0.0"
    )
    edgeCurl = deprecate_property("edge_curl", "edgeCurl", removal_version="1.0.0")
    _cellGradStencil = deprecate_property(
        "stencil_cell_gradient", "_cellGradStencil", removal_version="1.0.0"
    )
    _cellGradxStencil = deprecate_property(
        "stencil_cell_gradient_x", "_cellGradxStencil", removal_version="1.0.0"
    )
    _cellGradyStencil = deprecate_property(
        "stencil_cell_gradient_y", "_cellGradyStencil", removal_version="1.0.0"
    )
    _cellGradzStencil = deprecate_property(
        "stencil_cell_gradient_z", "_cellGradzStencil", removal_version="1.0.0"
    )

    setCellGradBC = deprecate_method(
        "set_cell_gradient_BC", "setCellGradBC", removal_version="1.0.0"
    )
    getBCProjWF = deprecate_method(
        "get_BC_projections", "getBCProjWF", removal_version="1.0.0"
    )
    getBCProjWF_simple = deprecate_method(
        "get_BC_projections_simple", "getBCProjWF_simple", removal_version="1.0.0"
    )


DiffOperators.__module__ = "discretize.operators"
