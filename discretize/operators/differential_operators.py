import numpy as np
from scipy import sparse as sp
import warnings
from discretize.utils import sdiag, speye, kron3, spzeros, ddx, av, av_extrap
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

    def __init__(self):
        raise Exception(
            "DiffOperators is a base class providing differential"
            "operators on meshes and cannot run on its own."
            "Inherit to your favorite Mesh class."
        )

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
            D = sp.hstack((self._face_x_divergence_stencil, self._face_y_divergence_stencil), format="csr")
        elif self.dim == 3:
            D = sp.hstack(
                (self._face_x_divergence_stencil, self._face_y_divergence_stencil, self._face_z_divergence_stencil),
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
                (self._nodal_gradient_x_stencil, self._nodal_gradient_y_stencil), format="csr"
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
                self._nodal_laplacian = self._nodal_laplacian_x + self._nodal_laplacian_y
            elif self.dim == 3:
                self._nodal_laplacian = (
                    self._nodal_laplacian_x
                    + self._nodal_laplacian_y
                    + self._nodal_laplacian_z
                )
        return self._nodal_laplacian

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
                self._average_cell_vector_to_face = sp.block_diag((aveCCV2Fx, aveCC2VFy), format="csr")
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
                self._average_node_to_edge = sp.vstack((self._average_node_to_edge_x, self._average_node_to_edge_y), format="csr")
            elif self.dim == 3:
                self._average_node_to_edge = sp.vstack(
                    (self._average_node_to_edge_x, self._average_node_to_edge_y, self._average_node_to_edge_z), format="csr"
                )
        return self._average_node_to_edge

    @property
    def _average_node_to_face_x(self):
        if self.dim == 1:
            aveN2Fx = av(self.shape_cells[0])
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
                self._average_node_to_face = sp.vstack((self._average_node_to_face_x, self._average_node_to_face_y), format="csr")
            elif self.dim == 3:
                self._average_node_to_face = sp.vstack(
                    (self._average_node_to_face_x, self._average_node_to_face_y, self._average_node_to_face_z), format="csr"
                )
        return self._average_node_to_face

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
    _cellGradStencil = deprecate_property("stencil_cell_gradient", "_cellGradStencil", removal_version="1.0.0")
    _cellGradxStencil = deprecate_property("stencil_cell_gradient_x", "_cellGradxStencil", removal_version="1.0.0")
    _cellGradyStencil = deprecate_property("stencil_cell_gradient_y", "_cellGradyStencil", removal_version="1.0.0")
    _cellGradzStencil = deprecate_property("stencil_cell_gradient_z", "_cellGradzStencil", removal_version="1.0.0")

    setCellGradBC = deprecate_method(
        "set_cell_gradient_BC", "setCellGradBC", removal_version="1.0.0"
    )
    getBCProjWF = deprecate_method(
        "get_BC_projections", "getBCProjWF", removal_version="1.0.0"
    )
    getBCProjWF_simple = deprecate_method(
        "get_BC_projections_simple", "getBCProjWF_simple", removal_version="1.0.0"
    )
