"""Differential operators for meshes."""
import numpy as np
from scipy import sparse as sp
import warnings
from discretize.base import BaseMesh
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
    """Check if boundary condition 'bc' is valid.

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
    """Create 1D derivative operator from cell-centers to nodes.

    This means we go from n to n+1

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
    """Create 1D derivative operator from cell-centers to nodes.

    This means we go from n to n+1.

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


class DiffOperators(BaseMesh):
    """Class used for creating differential and averaging operators.

    ``DiffOperators`` is a class for managing the construction of
    differential and averaging operators at the highest level.
    The ``DiffOperator`` class is inherited by every ``discretize``
    mesh class. In practice, differential and averaging operators are
    not constructed by creating instances of ``DiffOperators``.
    Instead, the operators are constructed (and sometimes stored)
    when called as a property of the mesh.

    """

    _aliases = {
        "aveFx2CC": "average_face_x_to_cell",
        "aveFy2CC": "average_face_y_to_cell",
        "aveFz2CC": "average_face_z_to_cell",
        "aveEx2CC": "average_edge_x_to_cell",
        "aveEy2CC": "average_edge_y_to_cell",
        "aveEz2CC": "average_edge_z_to_cell",
    }

    ###########################################################################
    #                                                                         #
    #                             Face Divergence                             #
    #                                                                         #
    ###########################################################################
    @property
    def _face_x_divergence_stencil(self):
        """Stencil for face divergence operator in the x-direction (x-faces to cell centers)."""
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
        """Stencil for face divergence operator in the y-direction (y-faces to cell centers)."""
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
        """Stencil for face divergence operator in the z-direction (z-faces to cell centers)."""
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
        """Full stencil for face divergence operator (faces to cell centers)."""
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
    def face_divergence(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
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
        r"""X-derivative operator (x-faces to cell-centres).

        This property constructs a 2nd order x-derivative operator which maps
        from x-faces to cell centers. The operator is a sparse matrix
        :math:`\mathbf{D_x}` that can be applied as a matrix-vector product
        to a discrete scalar quantity :math:`\boldsymbol{\phi}` that lives on
        x-faces; i.e.::

            dphi_dx = Dx @ phi

        For a discrete vector whose x-component lives on x-faces, this operator
        can also be used to compute the contribution of the x-component toward
        the divergence.

        Returns
        -------
        (n_cells, n_faces_x) scipy.sparse.csr_matrix
            The numerical x-derivative operator from x-faces to cell centers

        Examples
        --------
        Below, we demonstrate 1) how to apply the face-x divergence operator,
        and 2) the mapping of the face-x divergence operator and its sparsity.
        Our example is carried out on a 2D mesh but it can
        be done equivalently for a 3D mesh.

        We start by importing the necessary packages and modules.

        >>> from discretize import TensorMesh
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> import matplotlib as mpl

        For a discrete scalar quantity :math:`\boldsymbol{\phi}` defined on the
        x-faces, we take the x-derivative by constructing the face-x divergence
        operator and multiplying as a matrix-vector product.

        >>> h = np.ones(40)
        >>> mesh = TensorMesh([h, h], "CC")

        Create a discrete quantity on x-faces

        >>> faces_x = mesh.faces_x
        >>> phi = np.exp(-(faces_x[:, 0] ** 2) / 8** 2)

        Construct the x-divergence operator and apply to vector

        >>> Dx = mesh.face_x_divergence
        >>> dphi_dx = Dx @ phi

        Plot the original function and the x-divergence

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(13, 6))
            >>> ax1 = fig.add_subplot(121)
            >>> w = np.r_[phi, np.ones(mesh.nFy)]  # Need vector on all faces for image plot
            >>> mesh.plot_image(w, ax=ax1, v_type="Fx")
            >>> ax1.set_title("Scalar on x-faces", fontsize=14)
            >>> ax2 = fig.add_subplot(122)
            >>> mesh.plot_image(dphi_dx, ax=ax2)
            >>> ax2.set_yticks([])
            >>> ax2.set_ylabel("")
            >>> ax2.set_title("X-derivative at cell center", fontsize=14)
            >>> plt.show()

        The discrete x-face divergence operator is a sparse matrix that maps
        from x-faces to cell centers. To demonstrate this, we construct
        a small 2D mesh. We then show the ordering of the elements in
        the original discrete quantity :math:`\boldsymbol{\phi}}` and its
        x-derivative :math:`\partial \boldsymbol{\phi}}/ \partial x` as well as a
        spy plot.

        .. collapse:: Expand to show scripting for plot

            >>> mesh = TensorMesh([[(1, 6)], [(1, 3)]])
            >>> fig = plt.figure(figsize=(10, 10))
            >>> ax1 = fig.add_subplot(211)
            >>> mesh.plot_grid(ax=ax1)
            >>> ax1.plot(
            ...     mesh.faces_x[:, 0], mesh.faces_x[:, 1], "g>", markersize=8
            ... )
            >>> for ii, loc in zip(range(mesh.nFx), mesh.faces_x):
            ...     ax1.text(loc[0]+0.05, loc[1]+0.02, "{0:d}".format(ii), color="g")
            >>> ax1.plot(
            ...     mesh.cell_centers[:, 0], mesh.cell_centers[:, 1], "ro", markersize=8
            ... )
            >>> for ii, loc in zip(range(mesh.nC), mesh.cell_centers):
            ...     ax1.text(loc[0]+0.05, loc[1]+0.02, "{0:d}".format(ii), color="r")

            >>> ax1.set_xticks([])
            >>> ax1.set_yticks([])
            >>> ax1.spines['bottom'].set_color('white')
            >>> ax1.spines['top'].set_color('white')
            >>> ax1.spines['left'].set_color('white')
            >>> ax1.spines['right'].set_color('white')
            >>> ax1.set_xlabel('X', fontsize=16, labelpad=-5)
            >>> ax1.set_ylabel('Y', fontsize=16, labelpad=-15)
            >>> ax1.set_title("Mapping of Face-X Divergence", fontsize=14, pad=15)
            >>> ax1.legend(
            ...     ['Mesh', '$\mathbf{\phi}$ (x-faces)', '$\partial \mathbf{phi}/\partial x$ (centers)'],
            ...     loc='upper right', fontsize=14
            ... )
            >>> ax2 = fig.add_subplot(212)
            >>> ax2.spy(mesh.face_x_divergence)
            >>> ax2.set_title("Spy Plot", fontsize=14, pad=5)
            >>> ax2.set_ylabel("Cell Index", fontsize=12)
            >>> ax2.set_xlabel("X-Face Index", fontsize=12)
            >>> plt.show()
        """
        # Compute areas of cell faces & volumes
        S = self.reshape(self.face_areas, "F", "Fx", "V")
        V = self.cell_volumes
        return sdiag(1 / V) * self._face_x_divergence_stencil * sdiag(S)

    @property
    def face_y_divergence(self):
        r"""Y-derivative operator (y-faces to cell-centres).

        This property constructs a 2nd order y-derivative operator which maps
        from y-faces to cell centers. The operator is a sparse matrix
        :math:`\mathbf{D_y}` that can be applied as a matrix-vector product
        to a discrete scalar quantity :math:`\boldsymbol{\phi}` that lives on
        y-faces; i.e.::

            dphi_dy = Dy @ phi

        For a discrete vector whose y-component lives on y-faces, this operator
        can also be used to compute the contribution of the y-component toward
        the divergence.

        Returns
        -------
        (n_cells, n_faces_y) scipy.sparse.csr_matrix
            The numerical y-derivative operator from y-faces to cell centers

        Examples
        --------
        Below, we demonstrate 1) how to apply the face-y divergence operator,
        and 2) the mapping of the face-y divergence operator and its sparsity.
        Our example is carried out on a 2D mesh but it can
        be done equivalently for a 3D mesh.

        We start by importing the necessary packages and modules.

        >>> from discretize import TensorMesh
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> import matplotlib as mpl

        For a discrete scalar quantity :math:`\boldsymbol{\phi}` defined on the
        y-faces, we take the y-derivative by constructing the face-y divergence
        operator and multiplying as a matrix-vector product.

        >>> h = np.ones(40)
        >>> mesh = TensorMesh([h, h], "CC")

        Create a discrete quantity on x-faces

        >>> faces_y = mesh.faces_y
        >>> phi = np.exp(-(faces_y[:, 1] ** 2) / 8** 2)

        Construct the y-divergence operator and apply to vector

        >>> Dy = mesh.face_y_divergence
        >>> dphi_dy = Dy @ phi

        Plot original function and the y-divergence

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(13, 6))
            >>> ax1 = fig.add_subplot(121)
            >>> w = np.r_[np.ones(mesh.nFx), phi]  # Need vector on all faces for image plot
            >>> mesh.plot_image(w, ax=ax1, v_type="Fy")
            >>> ax1.set_title("Scalar on y-faces", fontsize=14)
            >>> ax2 = fig.add_subplot(122)
            >>> mesh.plot_image(dphi_dy, ax=ax2)
            >>> ax2.set_yticks([])
            >>> ax2.set_ylabel("")
            >>> ax2.set_title("Y-derivative at cell center", fontsize=14)
            >>> plt.show()

        The discrete y-face divergence operator is a sparse matrix that maps
        from y-faces to cell centers. To demonstrate this, we construct
        a small 2D mesh. We then show the ordering of the elements in
        the original discrete quantity :math:`\boldsymbol{\phi}` and its
        y-derivative :math:`\partial \boldsymbol{\phi}/ \partial y` as well as a
        spy plot.

        .. collapse:: Expand to show scripting for plot

            >>> mesh = TensorMesh([[(1, 6)], [(1, 3)]])
            >>> fig = plt.figure(figsize=(10, 10))
            >>> ax1 = fig.add_subplot(211)
            >>> mesh.plot_grid(ax=ax1)
            >>> ax1.plot(
            ...     mesh.faces_y[:, 0], mesh.faces_y[:, 1], "g^", markersize=8
            ... )
            >>> for ii, loc in zip(range(mesh.nFy), mesh.faces_y):
            ...     ax1.text(loc[0]+0.05, loc[1]+0.02, "{0:d}".format(ii), color="g")
            >>> ax1.plot(
            ...     mesh.cell_centers[:, 0], mesh.cell_centers[:, 1], "ro", markersize=8
            ... )
            >>> for ii, loc in zip(range(mesh.nC), mesh.cell_centers):
            ...     ax1.text(loc[0]+0.05, loc[1]+0.02, "{0:d}".format(ii), color="r")

            >>> ax1.set_xticks([])
            >>> ax1.set_yticks([])
            >>> ax1.spines['bottom'].set_color('white')
            >>> ax1.spines['top'].set_color('white')
            >>> ax1.spines['left'].set_color('white')
            >>> ax1.spines['right'].set_color('white')
            >>> ax1.set_xlabel('X', fontsize=16, labelpad=-5)
            >>> ax1.set_ylabel('Y', fontsize=16, labelpad=-15)
            >>> ax1.set_title("Mapping of Face-Y Divergence", fontsize=14, pad=15)
            >>> ax1.legend(
            ...     ['Mesh','$\mathbf{\phi}$ (y-faces)','$\partial_y \mathbf{\phi}/\partial y$ (centers)'],
            ...     loc='upper right', fontsize=14
            ... )
            >>> ax2 = fig.add_subplot(212)
            >>> ax2.spy(mesh.face_y_divergence)
            >>> ax2.set_title("Spy Plot", fontsize=14, pad=5)
            >>> ax2.set_ylabel("Cell Index", fontsize=12)
            >>> ax2.set_xlabel("Y-Face Index", fontsize=12)
            >>> plt.show()
        """
        if self.dim < 2:
            return None
        # Compute areas of cell faces & volumes
        S = self.reshape(self.face_areas, "F", "Fy", "V")
        V = self.cell_volumes
        return sdiag(1 / V) * self._face_y_divergence_stencil * sdiag(S)

    @property
    def face_z_divergence(self):
        r"""Z-derivative operator (z-faces to cell-centres).

        This property constructs a 2nd order z-derivative operator which maps
        from z-faces to cell centers. The operator is a sparse matrix
        :math:`\mathbf{D_z}` that can be applied as a matrix-vector product
        to a discrete scalar quantity :math:`\boldsymbol{\phi}` that lives on
        z-faces; i.e.::

            dphi_dz = Dz @ phi

        For a discrete vector whose z-component lives on z-faces, this operator
        can also be used to compute the contribution of the z-component toward
        the divergence.

        Returns
        -------
        (n_cells, n_faces_z) scipy.sparse.csr_matrix
            The numerical z-derivative operator from z-faces to cell centers

        Examples
        --------
        Below, we demonstrate 2) how to apply the face-z divergence operator,
        and 2) the mapping of the face-z divergence operator and its sparsity.
        Our example is carried out on a 3D mesh.

        We start by importing the necessary packages and modules.

        >>> from discretize import TensorMesh
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> import matplotlib as mpl

        For a discrete scalar quantity :math:`\boldsymbol{\phi}` defined on the
        z-faces, we take the z-derivative by constructing the face-z divergence
        operator and multiplying as a matrix-vector product.

        >>> h = np.ones(40)
        >>> mesh = TensorMesh([h, h, h], "CCC")

        Create a discrete quantity on z-faces

        >>> faces_z = mesh.faces_z
        >>> phi = np.exp(-(faces_z[:, 2] ** 2) / 8** 2)

        Construct the z-divergence operator and apply to vector

        >>> Dz = mesh.face_z_divergence
        >>> dphi_dz = Dz @ phi

        Plot the original function and the z-divergence

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(13, 6))
            >>> ax1 = fig.add_subplot(121)
            >>> w = np.r_[np.ones(mesh.nFx+mesh.nFz), phi]  # Need vector on all faces for image plot
            >>> mesh.plot_slice(w, ax=ax1, v_type="Fz", normal='Y', ind=20)
            >>> ax1.set_title("Scalar on z-faces (y-slice)", fontsize=14)
            >>> ax2 = fig.add_subplot(122)
            >>> mesh.plot_slice(dphi_dz, ax=ax2, normal='Y', ind=20)
            >>> ax2.set_yticks([])
            >>> ax2.set_ylabel("")
            >>> ax2.set_title("Z-derivative at cell center (y-slice)", fontsize=14)
            >>> plt.show()

        The discrete z-face divergence operator is a sparse matrix that maps
        from z-faces to cell centers. To demonstrate this, we construct
        a small 3D mesh. We then show the ordering of the elements in
        the original discrete quantity :math:`\boldsymbol{\phi}` and its
        z-derivative :math:`\partial \boldsymbol{\phi}/ \partial z` as well as a
        spy plot.

        .. collapse:: Expand to show scripting for plot

            >>> mesh = TensorMesh([[(1, 3)], [(1, 2)], [(1, 2)]])
            >>> fig = plt.figure(figsize=(9, 12))
            >>> ax1 = fig.add_axes([0, 0.35, 1, 0.6], projection='3d', elev=10, azim=-82)
            >>> mesh.plot_grid(ax=ax1)
            >>> ax1.plot(
            ...     mesh.faces_z[:, 0], mesh.faces_z[:, 1], mesh.faces_z[:, 2], "g^", markersize=10
            ... )
            >>> for ii, loc in zip(range(mesh.nFz), mesh.faces_z):
            ...     ax1.text(loc[0] + 0.05, loc[1] + 0.05, loc[2], "{0:d}".format(ii), color="g")

            >>> ax1.plot(
            ...    mesh.cell_centers[:, 0], mesh.cell_centers[:, 1], mesh.cell_centers[:, 2],
            ...    "ro", markersize=10
            ... )
            >>> for ii, loc in zip(range(mesh.nC), mesh.cell_centers):
            ...     ax1.text(loc[0] + 0.05, loc[1] + 0.05, loc[2], "{0:d}".format(ii), color="r")
            >>> ax1.legend(
            ...     ['Mesh','$\mathbf{\phi}$ (z-faces)','$\partial \mathbf{\phi}/\partial z$ (centers)'],
            ...     loc='upper right', fontsize=14
            ... )

            Manually make axis properties invisible

            >>> ax1.set_xticks([])
            >>> ax1.set_yticks([])
            >>> ax1.set_zticks([])
            >>> ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            >>> ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            >>> ax1.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            >>> ax1.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            >>> ax1.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            >>> ax1.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            >>> ax1.set_xlabel('X', labelpad=-15, fontsize=16)
            >>> ax1.set_ylabel('Y', labelpad=-20, fontsize=16)
            >>> ax1.set_zlabel('Z', labelpad=-20, fontsize=16)
            >>> ax1.set_title("Mapping of Face-Z Divergence", fontsize=16, pad=-15)

            Spy plot the operator,

            >>> ax2 = fig.add_axes([0.05, 0.05, 0.9, 0.3])
            >>> ax2.spy(mesh.face_z_divergence)
            >>> ax2.set_title("Spy Plot", fontsize=16, pad=5)
            >>> ax2.set_ylabel("Cell Index", fontsize=12)
            >>> ax2.set_xlabel("Z-Face Index", fontsize=12)
            >>> plt.show()
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
        """Stencil for the nodal gradient in the x-direction (nodes to x-edges)."""
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
        """Stencil for the nodal gradient in the y-direction (nodes to y-edges)."""
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
        """Stencil for the nodal gradient in the z-direction (nodes to z-edges)."""
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
        """Full stencil for the nodal gradient (nodes to edges)."""
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
    def nodal_gradient(self):  # NOQA D102
        if getattr(self, "_nodal_gradient", None) is None:
            G = self._nodal_gradient_stencil
            L = self.edge_lengths
            self._nodal_gradient = sdiag(1 / L) * G
        return self._nodal_gradient

    @property
    def _nodal_laplacian_x_stencil(self):
        """Stencil for the nodal Laplacian in the x-direction (nodes to nodes)."""
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
        """Stencil for the nodal Laplacian in the y-direction (nodes to nodes)."""
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
        """Stencil for the nodal Laplacian in the z-direction (nodes to nodes)."""
        warnings.warn("Laplacian has not been tested rigorously.")

        if self.dim == 1 or self.dim == 2:
            return None

        Dz = ddx(self.shape_cells[2])
        Lz = -Dz.T * Dz
        return kron3(Lz, speye(self.shape_nodes[1]), speye(self.shape_nodes[0]))

    @property
    def _nodal_laplacian_x(self):
        """Construct the nodal Laplacian in the x-direction (nodes to nodes)."""
        Hx = sdiag(1.0 / self.h[0])
        if self.dim == 2:
            Hx = sp.kron(speye(self.shape_nodes[1]), Hx)
        elif self.dim == 3:
            Hx = kron3(speye(self.shape_nodes[2]), speye(self.shape_nodes[1]), Hx)
        return Hx.T * self._nodal_gradient_x_stencil * Hx

    @property
    def _nodal_laplacian_y(self):
        """Construct the nodal Laplacian in the y-direction (nodes to nodes)."""
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
        """Construct the nodal Laplacian in the z-direction (nodes to nodes)."""
        if self.dim == 1 or self.dim == 2:
            return None
        Hz = sdiag(1.0 / self.h[2])
        Hz = kron3(Hz, speye(self.shape_nodes[1]), speye(self.shape_nodes[0]))
        return Hz.T * self._nodal_laplacian_z_stencil * Hz

    @property
    def nodal_laplacian(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
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
        r"""Create Robin conditions pieces for weak form of the edge divergence operator (edges to nodes).

        This method returns the pieces required to impose Robin boundary conditions
        for the discrete weak form divergence operator that maps from edges to nodes.
        These pieces are needed when constructing the discrete representation
        of the inner product :math:`\langle \psi , \nabla \cdot \vec{u} \rangle`
        according to the finite volume method.

        To implement the boundary conditions, we assume

        .. math::
            \vec{u} = \nabla \phi

        for some scalar function :math:`\phi`. Boundary conditions are imposed
        on the scalar function according to the Robin condition:

        .. math::
            \alpha \phi + \beta \frac{\partial \phi}{\partial n} = \gamma

        The user supplies values for :math:`\alpha`, :math:`\beta` and :math:`\gamma`
        for all boundary nodes or faces. For the values supplied,
        **edge_divergence_weak_form_robin** returns the matrix :math:`\mathbf{B}`
        and vector :math:`\mathbf{b}` required for the discrete representation
        of :math:`\langle \psi , \nabla \cdot \vec{u} \rangle`.
        *See the notes section for a comprehensive description.*

        Parameters
        ----------
        alpha : scalar or array_like
            Defines :math:`\alpha` for Robin boundary condition. Can be defined as a
            scalar or array_like. If array_like, the length of the array must be equal
            to the number of boundary faces or boundary nodes.
        beta : scalar or array_like
            Defines :math:`\beta` for Robin boundary condition. Can be defined as a
            scalar or array_like. If array_like, must have the same length as *alpha*.
            Cannot be zero.
        gamma : scalar or array_like
            Defines :math:`\gamma` for Robin boundary condition. If array like, *gamma*
            can have shape (n_boundary_xxx,). Can also have shape (n_boundary_xxx, n_rhs)
            if multiple systems have the same *alpha* and *beta* parameters.

        Returns
        -------
        B : (n_nodes, n_nodes) scipy.sparse.dia_matrix
            A sparse matrix dependent on the values of *alpha*, *beta* and *gamma* supplied
        b : (n_nodes) numpy.ndarray or (n_nodes, n_rhs) numpy.ndarray
            A vector dependent on the values of *alpha*, *beta* and *gamma* supplied

        Notes
        -----
        For the divergence of a vector :math:`\vec{u}`, the weak form is implemented by taking
        the inner product with a piecewise-constant test function :math:`\psi` and integrating
        over the domain:

        .. math::
            \langle \psi , \nabla \cdot \vec{u} \rangle \; = \int_\Omega \psi \, (\nabla \cdot \vec{u}) \, dv

        For a discrete representation of the vector :math:`\vec{u}` that lives on mesh edges,
        the divergence operator must map from edges to nodes. To implement boundary conditions in this
        case, we must use the integration by parts to re-express the inner product as:

        .. math::
            \langle \psi , \nabla \cdot \vec{u} \rangle \, = - \int_V \vec{u} \cdot \nabla \psi \, dV
            + \oint_{\partial \Omega} \psi \, (\hat{n} \cdot \vec{u}) \, da

        Assuming :math:`\vec{u} = \nabla \phi`, the above equation becomes:

        .. math::
            \langle \psi , \nabla \cdot \vec{u} \rangle \, = - \int_V \nabla \phi \cdot \nabla \psi \, dV
            + \oint_{\partial \Omega} \psi \, \frac{\partial \phi}{\partial n} \, da

        Substituting in the Robin conditions:

        .. math::
            \langle \psi , \nabla \cdot \vec{u} \rangle \, = - \int_V \nabla \phi \cdot \nabla \psi \, dV
            + \oint_{\partial \Omega} \psi \, \frac{(\gamma - \alpha \Phi)}{\beta} \, da

        therefore, `beta` cannot be zero.

        The discrete approximation to the above expression is given by:

        .. math::
            \langle \psi , \nabla \cdot \vec{u} \rangle \,
            \approx - \boldsymbol{\psi^T \big ( G_n^T M_e G_n - B \big ) \phi + \psi^T b}

        where

        .. math::
            \boldsymbol{u} = \boldsymbol{G_n \, \phi}

        :math:`\mathbf{G_n}` is the :py:attr:`~discretize.operators.DiffOperators.nodal_gradient`
        and :math:`\mathbf{M_e}` is the edge inner product matrix
        (see :py:attr:`~discretize.operators.InnerProducts.get_edge_inner_product`).
        **edge_divergence_weak_form_robin** returns the matrix :math:`\mathbf{B}`
        and vector :math:`\mathbf{b}` based on the parameters *alpha* , *beta*
        and *gamma* provided.

        Examples
        --------
        Here we construct all of the pieces required for the discrete
        representation of :math:`\langle \psi , \nabla \cdot \vec{u} \rangle`
        for specified Robin boundary conditions. We define
        :math:`\mathbf{u}` on the edges, and :math:`\boldsymbol{\psi}`
        and :math:`\boldsymbol{\psi}` on the nodes.
        We begin by creating a small 2D tensor mesh:

        >>> from discretize import TensorMesh
        >>> import numpy as np
        >>> import scipy.sparse as sp
        >>> h = np.ones(32)
        >>> mesh = TensorMesh([h, h])

        We then define `alpha`, `beta`, and `gamma` parameters for a zero Neumann
        condition on the boundary faces. This corresponds to setting:

        >>> alpha = 0.0
        >>> beta = 1.0
        >>> gamma = 0.0

        Next, we construct all of the necessary pieces required to take
        the discrete inner product:

        >>> B, b = mesh.edge_divergence_weak_form_robin(alpha, beta, gamma)
        >>> Me = mesh.get_edge_inner_product()
        >>> Gn = mesh.nodal_gradient

        In practice, these pieces are usually re-arranged when used to
        solve PDEs with the finite volume method. Because the boundary
        conditions are applied to the scalar potential :math:`\phi`,
        we create a function which computes the discrete inner product for any
        :math:`\boldsymbol{\psi}` and :math:`\boldsymbol{\phi}` where
        :math:`\mathbf{u} = \boldsymbol{G \, \phi}`:

        >>> def inner_product(psi, phi):
        ...     return psi @ (-Gn.T @ Me @ Gn + B) @ phi + psi @ b
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
        """Set boundary conditions for derivative operators acting on cell-centered quantities.

        This method is used to set zero Dirichlet and/or zero Neumann boundary
        conditions for differential operators that act on cell-centered quantities.
        The user may apply the same boundary conditions to all boundaries, or
        define the boundary conditions of boundary face (x, y and z) separately.
        The user may also apply boundary conditions to the lower and upper boundary
        face separately.

        Cell gradient boundary conditions are enforced when constructing
        the following properties:

            - :py:attr:`~discretize.operators.DiffOperators.cell_gradient`
            - :py:attr:`~discretize.operators.DiffOperators.cell_gradient_x`
            - :py:attr:`~discretize.operators.DiffOperators.cell_gradient_y`
            - :py:attr:`~discretize.operators.DiffOperators.cell_gradient_z`
            - :py:attr:`~discretize.operators.DiffOperators.stencil_cell_gradient`
            - :py:attr:`~discretize.operators.DiffOperators.stencil_cell_gradient_x`
            - :py:attr:`~discretize.operators.DiffOperators.stencil_cell_gradient_y`
            - :py:attr:`~discretize.operators.DiffOperators.stencil_cell_gradient_z`

        By default, the mesh assumes a zero Neumann boundary condition on the
        entire boundary. To define robin boundary conditions, see
        :py:attr:`~discretize.operators.DiffOperators.cell_gradient_weak_form_robin`.

        Parameters
        ----------
        BC : str or list [dim,]
            Define the boundary conditions using the string 'dirichlet' for zero
            Dirichlet conditions and 'neumann' for zero Neumann conditions. See
            *examples* for several implementations.

        Examples
        --------
        Here we demonstrate how to apply zero Dirichlet and/or Neumann boundary
        conditions for cell-centers differential operators.

        >>> from discretize import TensorMesh
        >>> mesh = TensorMesh([[(1, 20)], [(1, 20)], [(1, 20)]])

        Define zero Neumann conditions for all boundaries

        >>> BC = 'neumann'
        >>> mesh.set_cell_gradient_BC(BC)

        Define zero Dirichlet on y boundaries and zero Neumann otherwise

        >>> BC = ['neumann', 'dirichlet', 'neumann']
        >>> mesh.set_cell_gradient_BC(BC)

        Define zero Neumann on the bottom x-boundary and zero Dirichlet otherwise

        >>> BC = [['neumann', 'dirichlet'], 'dirichlet', 'dirichlet']
        >>> mesh.set_cell_gradient_BC(BC)
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
        r"""Differencing operator along x-direction (cell centers to x-faces).

        This property constructs a differencing operator along the x-axis
        that acts on cell centered quantities; i.e. the stencil for the
        x-component of the cell gradient. The operator computes the
        differences between the values at adjacent cell centers along the
        x-direction, and places the result on the x-faces. The operator is a sparse
        matrix :math:`\mathbf{G_x}` that can be applied as a matrix-vector
        product to a cell centered quantity :math:`\boldsymbol{\phi}`, i.e.::

            diff_phi_x = Gx @ phi

        By default, the operator assumes zero-Neumann boundary conditions
        on the scalar quantity. Before calling **stencil_cell_gradient_x** however,
        the user can set a mix of zero Dirichlet and zero Neumann boundary
        conditions using :py:attr:`~discretize.operators.DiffOperators.set_cell_gradient_BC`.
        When **stencil_cell_gradient_x** is called, the boundary conditions are
        enforced for the differencing operator.

        Returns
        -------
        (n_faces_x, n_cells) scipy.sparse.csr_matrix
            The stencil for the x-component of the cell gradient

        Examples
        --------
        Below, we demonstrate how to set boundary conditions for the
        x-component cell gradient stencil, construct the operator and apply it
        to a discrete scalar quantity. The mapping of the operator and
        its sparsity is also illustrated. Our example is carried out on a 2D
        mesh but it can be done equivalently for a 3D mesh.

        We start by importing the necessary packages and modules.

        >>> from discretize import TensorMesh
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> import matplotlib as mpl

        We then construct a mesh and define a scalar function at cell
        centers. In this case, the scalar represents some block within
        a homogeneous medium.

        Create a uniform grid

        >>> h = np.ones(40)
        >>> mesh = TensorMesh([h, h], "CC")

        Create a discrete scalar at cell centers

        >>> centers = mesh.cell_centers
        >>> phi = np.zeros(mesh.nC)
        >>> k = (np.abs(mesh.cell_centers[:, 0]) < 10.) & (np.abs(mesh.cell_centers[:, 1]) < 10.)
        >>> phi[k] = 1.

        Before constructing the stencil gradient operator, we must define
        the boundary conditions; zero Neumann for our example. Even though
        we are only computing the difference along x, we define boundary
        conditions for all boundary faces. Once the
        operator is created, it is applied as a matrix-vector product.

        >>> mesh.set_cell_gradient_BC(['neumann', 'neumann'])
        >>> Gx = mesh.stencil_cell_gradient_x
        >>> diff_phi_x = Gx @ phi

        Now we plot the original scalar, and the differencing taken along the
        x axes.

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(13, 5))
            >>> ax1 = fig.add_subplot(121)
            >>> mesh.plot_image(phi, ax=ax1)
            >>> ax1.set_title("Scalar at cell centers", fontsize=14)

            >>> ax2 = fig.add_subplot(122)
            >>> v = np.r_[diff_phi_x, np.zeros(mesh.nFy)]  # Define vector for plotting fun
            >>> mesh.plot_image(v, ax=ax2, v_type="Fx")
            >>> ax2.set_yticks([])
            >>> ax2.set_ylabel("")
            >>> ax2.set_title("Difference (x-axis)", fontsize=14)
            >>> plt.show()

        The x-component cell gradient stencil is a sparse differencing matrix
        that maps from cell centers to x-faces. To demonstrate this, we construct
        a small 2D mesh. We then show the ordering of the elements
        and a spy plot.

        >>> mesh = TensorMesh([[(1, 3)], [(1, 4)]])
        >>> mesh.set_cell_gradient_BC('neumann')

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(12, 8))
            >>> ax1 = fig.add_subplot(121)
            >>> mesh.plot_grid(ax=ax1)
            >>> ax1.set_title("Mapping of Stencil", fontsize=14, pad=15)

            >>> ax1.plot(mesh.cell_centers[:, 0], mesh.cell_centers[:, 1], "ro", markersize=8)
            >>> for ii, loc in zip(range(mesh.nC), mesh.cell_centers):
            ...     ax1.text(loc[0] + 0.05, loc[1] + 0.02, "{0:d}".format(ii), color="r")
            >>> ax1.plot(mesh.faces_x[:, 0], mesh.faces_x[:, 1], "g>", markersize=8)
            >>> for ii, loc in zip(range(mesh.nFx), mesh.faces_x):
            ...     ax1.text(loc[0] + 0.05, loc[1] + 0.02, "{0:d}".format(ii), color="g")
            >>> ax1.set_xticks([])
            >>> ax1.set_yticks([])
            >>> ax1.spines['bottom'].set_color('white')
            >>> ax1.spines['top'].set_color('white')
            >>> ax1.spines['left'].set_color('white')
            >>> ax1.spines['right'].set_color('white')
            >>> ax1.set_xlabel('X', fontsize=16, labelpad=-5)
            >>> ax1.set_ylabel('Y', fontsize=16, labelpad=-15)
            >>> ax1.legend(
            ...     ['Mesh', '$\mathbf{\phi}$ (centers)', '$\mathbf{Gx^* \phi}$ (x-faces)'],
            ...     loc='upper right', fontsize=14
            ... )
            >>> ax2 = fig.add_subplot(122)
            >>> ax2.spy(mesh.stencil_cell_gradient_x)
            >>> ax2.set_title("Spy Plot", fontsize=14, pad=5)
            >>> ax2.set_ylabel("X-Face Index", fontsize=12)
            >>> ax2.set_xlabel("Cell Index", fontsize=12)
            >>> plt.show()
        """
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
        r"""Differencing operator along y-direction (cell centers to y-faces).

        This property constructs a differencing operator along the x-axis
        that acts on cell centered quantities; i.e. the stencil for the
        y-component of the cell gradient. The operator computes the
        differences between the values at adjacent cell centers along the
        y-direction, and places the result on the y-faces. The operator is a sparse
        matrix :math:`\mathbf{G_y}` that can be applied as a matrix-vector
        product to a cell centered quantity :math:`\boldsymbol{\phi}`, i.e.::

            diff_phi_y = Gy @ phi

        By default, the operator assumes zero-Neumann boundary conditions
        on the scalar quantity. Before calling **stencil_cell_gradient_y** however,
        the user can set a mix of zero Dirichlet and zero Neumann boundary
        conditions using :py:attr:`~discretize.operators.DiffOperators.set_cell_gradient_BC`.
        When **stencil_cell_gradient_y** is called, the boundary conditions are
        enforced for the differencing operator.

        Returns
        -------
        (n_faces_y, n_cells) scipy.sparse.csr_matrix
            The stencil for the y-component of the cell gradient

        Examples
        --------
        Below, we demonstrate how to set boundary conditions for the
        y-component cell gradient stencil, construct the operator and apply it
        to a discrete scalar quantity. The mapping of the operator and
        its sparsity is also illustrated. Our example is carried out on a 2D
        mesh but it can be done equivalently for a 3D mesh.

        We start by importing the necessary packages and modules.

        >>> from discretize import TensorMesh
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> import matplotlib as mpl

        We then construct a mesh and define a scalar function at cell
        centers. In this case, the scalar represents some block within
        a homogeneous medium.

        Create a uniform grid

        >>> h = np.ones(40)
        >>> mesh = TensorMesh([h, h], "CC")

        Create a discrete scalar at cell centers

        >>> centers = mesh.cell_centers
        >>> phi = np.zeros(mesh.nC)
        >>> k = (np.abs(mesh.cell_centers[:, 0]) < 10.) & (np.abs(mesh.cell_centers[:, 1]) < 10.)
        >>> phi[k] = 1.

        Before constructing the operator, we must define
        the boundary conditions; zero Neumann for our example. Even though
        we are only computing the difference along y, we define boundary
        conditions for all boundary faces. Once the
        operator is created, it is applied as a matrix-vector product.

        >>> mesh.set_cell_gradient_BC(['neumann', 'neumann'])
        >>> Gy = mesh.stencil_cell_gradient_y
        >>> diff_phi_y = Gy @ phi

        Now we plot the original scalar, and the differencing taken along the
        y-axis.

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(13, 5))
            >>> ax1 = fig.add_subplot(121)
            >>> mesh.plot_image(phi, ax=ax1)
            >>> ax1.set_title("Scalar at cell centers", fontsize=14)
            >>> ax2 = fig.add_subplot(122)
            >>> v = np.r_[np.zeros(mesh.nFx), diff_phi_y]  # Define vector for plotting fun
            >>> mesh.plot_image(v, ax=ax2, v_type="Fy")
            >>> ax2.set_yticks([])
            >>> ax2.set_ylabel("")
            >>> ax2.set_title("Difference (y-axis)", fontsize=14)
            >>> plt.show()

        The y-component cell gradient stencil is a sparse differencing matrix
        that maps from cell centers to y-faces. To demonstrate this, we construct
        a small 2D mesh. We then show the ordering of the elements
        and a spy plot.

        >>> mesh = TensorMesh([[(1, 3)], [(1, 4)]])
        >>> mesh.set_cell_gradient_BC('neumann')

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(12, 8))
            >>> ax1 = fig.add_subplot(121)
            >>> mesh.plot_grid(ax=ax1)
            >>> ax1.set_title("Mapping of Stencil", fontsize=14, pad=15)
            >>> ax1.plot(mesh.cell_centers[:, 0], mesh.cell_centers[:, 1], "ro", markersize=8)
            >>> for ii, loc in zip(range(mesh.nC), mesh.cell_centers):
            ...     ax1.text(loc[0] + 0.05, loc[1] + 0.02, "{0:d}".format(ii), color="r")
            >>> ax1.plot(mesh.faces_y[:, 0], mesh.faces_y[:, 1], "g^", markersize=8)
            >>> for ii, loc in zip(range(mesh.nFy), mesh.faces_y):
            ...     ax1.text(loc[0] + 0.05, loc[1] + 0.02, "{0:d}".format(ii + mesh.nFx), color="g")

            >>> ax1.set_xticks([])
            >>> ax1.set_yticks([])
            >>> ax1.spines['bottom'].set_color('white')
            >>> ax1.spines['top'].set_color('white')
            >>> ax1.spines['left'].set_color('white')
            >>> ax1.spines['right'].set_color('white')
            >>> ax1.set_xlabel('X', fontsize=16, labelpad=-5)
            >>> ax1.set_ylabel('Y', fontsize=16, labelpad=-15)
            >>> ax1.legend(
            ...     ['Mesh', '$\mathbf{\phi}$ (centers)', '$\mathbf{Gy^* \phi}$ (y-faces)'],
            ...     loc='upper right', fontsize=14
            ... )

            >>> ax2 = fig.add_subplot(122)
            >>> ax2.spy(mesh.stencil_cell_gradient_y)
            >>> ax2.set_title("Spy Plot", fontsize=14, pad=5)
            >>> ax2.set_ylabel("Y-Face Index", fontsize=12)
            >>> ax2.set_xlabel("Cell Index", fontsize=12)
            >>> plt.show()
        """
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
        r"""Differencing operator along z-direction (cell centers to z-faces).

        This property constructs a differencing operator along the z-axis
        that acts on cell centered quantities; i.e. the stencil for the
        z-component of the cell gradient. The operator computes the
        differences between the values at adjacent cell centers along the
        z-direction, and places the result on the z-faces. The operator is a sparse
        matrix :math:`\mathbf{G_z}` that can be applied as a matrix-vector
        product to a cell centered quantity :math:`\boldsymbol{\phi}`, i.e.::

            diff_phi_z = Gz @ phi

        By default, the operator assumes zero-Neumann boundary conditions
        on the scalar quantity. Before calling **stencil_cell_gradient_z** however,
        the user can set a mix of zero Dirichlet and zero Neumann boundary
        conditions using :py:attr:`~discretize.operators.DiffOperators.set_cell_gradient_BC`.
        When **stencil_cell_gradient_z** is called, the boundary conditions are
        enforced for the differencing operator.

        Returns
        -------
        (n_faces_z, n_cells) scipy.sparse.csr_matrix
            The stencil for the z-component of the cell gradient

        Examples
        --------
        Below, we demonstrate how to set boundary conditions for the
        z-component cell gradient stencil, construct the operator and apply it
        to a discrete scalar quantity. The mapping of the operator and
        its sparsity is also illustrated.

        We start by importing the necessary packages and modules.

        >>> from discretize import TensorMesh
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> import matplotlib as mpl

        We then construct a mesh and define a scalar function at cell
        centers. In this case, the scalar represents some block within
        a homogeneous medium.

        Create a uniform grid

        >>> h = np.ones(40)
        >>> mesh = TensorMesh([h, h, h], "CCC")

        Create a discrete scalar at cell centers

        >>> centers = mesh.cell_centers
        >>> phi = np.zeros(mesh.nC)
        >>> k = (
        ...     (np.abs(mesh.cell_centers[:, 0]) < 10.) &
        ...     (np.abs(mesh.cell_centers[:, 1]) < 10.) &
        ...     (np.abs(mesh.cell_centers[:, 2]) < 10.)
        ... )
        >>> phi[k] = 1.

        Before constructing the operator, we must define
        the boundary conditions; zero Neumann for our example. Even though
        we are only computing the difference along z, we define boundary
        conditions for all boundary faces. Once the
        operator is created, it is applied as a matrix-vector product.

        >>> mesh.set_cell_gradient_BC(['neumann', 'neumann', 'neumann'])
        >>> Gz = mesh.stencil_cell_gradient_z
        >>> diff_phi_z = Gz @ phi

        Now we plot the original scalar, and the differencing taken along the
        z-axis for a slice at y = 0.

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(13, 5))
            >>> ax1 = fig.add_subplot(121)
            >>> mesh.plot_slice(phi, ax=ax1, normal='Y', slice_loc=0)
            >>> ax1.set_title("Scalar at cell centers", fontsize=14)
            >>> ax2 = fig.add_subplot(122)
            >>> v = np.r_[np.zeros(mesh.nFx+mesh.nFy), diff_phi_z]  # Define vector for plotting fun
            >>> mesh.plot_slice(v, ax=ax2, v_type='Fz', normal='Y', slice_loc=0)
            >>> ax2.set_title("Difference (z-axis)", fontsize=14)
            >>> plt.show()

        The z-component cell gradient stencil is a sparse differencing matrix
        that maps from cell centers to z-faces. To demonstrate this, we provide
        a spy plot

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(9, 9))
            >>> ax1 = fig.add_subplot(111)
            >>> ax1.spy(mesh.stencil_cell_gradient_z, ms=1)
            >>> ax1.set_title("Spy Plot", fontsize=16, pad=5)
            >>> ax1.set_xlabel("Cell Index", fontsize=12)
            >>> ax1.set_ylabel("Z-Face Index", fontsize=12)
            >>> plt.show()
        """
        if self.dim < 3:
            return None
        BC = ["neumann", "neumann"]  # TODO: remove this hard-coding
        n = self.vnC
        G3 = kron3(_ddxCellGrad(n[2], BC), speye(n[1]), speye(n[0]))
        return G3

    @property
    def stencil_cell_gradient(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
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
        r"""Cell gradient operator (cell centers to faces).

        This property constructs the 2nd order numerical gradient operator
        that maps from cell centers to faces. The operator is a sparse matrix
        :math:`\mathbf{G_c}` that can be applied as a matrix-vector product
        to a discrete scalar quantity :math:`\boldsymbol{\phi}` that lives
        at the cell centers; i.e.::

            grad_phi = Gc @ phi

        By default, the operator assumes zero-Neumann boundary conditions
        on the scalar quantity. Before calling **cell_gradient** however,
        the user can set a mix of zero Dirichlet and zero Neumann boundary
        conditions using :py:attr:`~discretize.operators.DiffOperators.set_cell_gradient_BC`.
        When **cell_gradient** is called, the boundary conditions are
        enforced for the gradient operator. Once constructed, the operator is
        stored as a property of the mesh. *See notes*.

        This operator is defined mostly as a helper property, and is not necessarily
        recommended to use when solving PDE's.

        Returns
        -------
        (n_faces, n_cells) scipy.sparse.csr_matrix
            The numerical gradient operator from cell centers to faces


        Notes
        -----
        In continuous space, the gradient operator is defined as:

        .. math::
            \vec{u} = \nabla \phi = \frac{\partial \phi}{\partial x}\hat{x}
            + \frac{\partial \phi}{\partial y}\hat{y}
            + \frac{\partial \phi}{\partial z}\hat{z}

        Where :math:`\boldsymbol{\phi}` is the discrete representation of the continuous variable
        :math:`\phi` at cell centers and :math:`\mathbf{u}` is the discrete
        representation of :math:`\vec{u}` on the faces, **cell_gradient** constructs a
        discrete linear operator :math:`\mathbf{G_c}` such that:

        .. math::
            \mathbf{u} = \mathbf{G_c} \, \boldsymbol{\phi}

        Second order ghost points are used to enforce boundary conditions and map
        appropriately to boundary faces. Along each axes direction, we are
        effectively computing the derivative by taking the difference between the
        values at adjacent cell centers and dividing by their distance.

        Examples
        --------
        Below, we demonstrate how to set boundary conditions for the cell gradient
        operator, construct the cell gradient operator and apply it to a discrete
        scalar quantity. The mapping of the operator and its sparsity is also
        illustrated. Our example is carried out on a 2D mesh but it can
        be done equivalently for a 3D mesh.

        We start by importing the necessary packages and modules.

        >>> from discretize import TensorMesh
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> import matplotlib as mpl

        We then construct a mesh and define a scalar function at cell
        centers which is zero on the boundaries (zero Dirichlet).

        Create a uniform grid

        >>> h = np.ones(20)
        >>> mesh = TensorMesh([h, h], "CC")

        Create a discrete scalar on nodes

        >>> centers = mesh.cell_centers
        >>> phi = np.exp(-(centers[:, 0] ** 2 + centers[:, 1] ** 2) / 4 ** 2)

        Once the operator is created, the gradient is performed as a
        matrix-vector product.

        Construct the gradient operator and apply to vector

        >>> Gc = mesh.cell_gradient
        >>> grad_phi = Gc @ phi

        Plot the results

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(13, 6))
            >>> ax1 = fig.add_subplot(121)
            >>> mesh.plot_image(phi, ax=ax1)
            >>> ax1.set_title("Scalar at cell centers", fontsize=14)
            >>> ax2 = fig.add_subplot(122)
            >>> mesh.plot_image(
            ...     grad_phi, ax=ax2, v_type="F", view="vec",
            ...     stream_opts={"color": "w", "density": 1.0}
            ... )
            >>> ax2.set_yticks([])
            >>> ax2.set_ylabel("")
            >>> ax2.set_title("Gradient at faces", fontsize=14)
            >>> plt.show()

        The cell gradient operator is a sparse matrix that maps
        from cell centers to faces. To demonstrate this, we construct
        a small 2D mesh. We then show the ordering of the elements in
        the original discrete quantity :math:`\boldsymbol{\phi}` and its
        discrete gradient as well as a spy plot.

        >>> mesh = TensorMesh([[(1, 3)], [(1, 6)]])
        >>> mesh.set_cell_gradient_BC('dirichlet')

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(12, 10))
            >>> ax1 = fig.add_subplot(121)
            >>> mesh.plot_grid(ax=ax1)
            >>> ax1.set_title("Mapping of Gradient Operator", fontsize=14, pad=15)
            >>> ax1.plot(mesh.cell_centers[:, 0], mesh.cell_centers[:, 1], "ro", markersize=8)
            >>> for ii, loc in zip(range(mesh.nC), mesh.cell_centers):
            ...     ax1.text(loc[0] + 0.05, loc[1] + 0.02, "{0:d}".format(ii), color="r")

            >>> ax1.plot(mesh.faces_x[:, 0], mesh.faces_x[:, 1], "g^", markersize=8)
            >>> for ii, loc in zip(range(mesh.nFx), mesh.faces_x):
            ...     ax1.text(loc[0] + 0.05, loc[1] + 0.02, "{0:d}".format(ii), color="g")
            >>> ax1.plot(mesh.faces_y[:, 0], mesh.faces_y[:, 1], "g>", markersize=8)
            >>> for ii, loc in zip(range(mesh.nFy), mesh.faces_y):
            ...     ax1.text(loc[0] + 0.05, loc[1] + 0.02, "{0:d}".format((ii + mesh.nFx)), color="g")

            >>> ax1.set_xticks([])
            >>> ax1.set_yticks([])
            >>> ax1.spines['bottom'].set_color('white')
            >>> ax1.spines['top'].set_color('white')
            >>> ax1.spines['left'].set_color('white')
            >>> ax1.spines['right'].set_color('white')
            >>> ax1.set_xlabel('X', fontsize=16, labelpad=-5)
            >>> ax1.set_ylabel('Y', fontsize=16, labelpad=-15)

            >>> ax1.legend(
            ...     ['Mesh', '$\mathbf{\phi}$ (centers)', '$\mathbf{u}$ (faces)'],
            ...     loc='upper right', fontsize=14
            ... )
            >>> ax2 = fig.add_subplot(122)
            >>> ax2.spy(mesh.cell_gradient)
            >>> ax2.set_title("Spy Plot", fontsize=14, pad=5)
            >>> ax2.set_ylabel("Face Index", fontsize=12)
            >>> ax2.set_xlabel("Cell Index", fontsize=12)
            >>> plt.show()
        """
        if getattr(self, "_cell_gradient", None) is None:
            G = self.stencil_cell_gradient
            S = self.face_areas  # Compute areas of cell faces & volumes
            V = (
                self.aveCC2F * self.cell_volumes
            )  # Average volume between adjacent cells
            self._cell_gradient = sdiag(S / V) * G
        return self._cell_gradient

    def cell_gradient_weak_form_robin(self, alpha=0.0, beta=1.0, gamma=0.0):
        r"""Create Robin conditions pieces for weak form of the cell gradient operator (cell centers to faces).

        This method returns the pieces required to impose Robin boundary conditions
        for the discrete weak form gradient operator that maps from cell centers to faces.
        These pieces are needed when constructing the discrete representation
        of the inner product :math:`\langle \vec{u} , \nabla \phi \rangle`
        according to the finite volume method.

        Where general boundary conditions are defined on :math:`\phi`
        according to the Robin condition:

        .. math::
            \alpha \phi + \beta \frac{\partial \phi}{\partial n} = \gamma

        the user supplies values for :math:`\alpha`, :math:`\beta` and :math:`\gamma`
        for all boundary faces. **cell_gradient_weak_form_robin** returns the matrix
        :math:`\mathbf{B}` and vector :math:`\mathbf{b}` required for the discrete
        representation of :math:`\langle \vec{u} , \nabla \phi \rangle`.
        *See the notes section for a comprehensive description.*

        Parameters
        ----------
        alpha : scalar or (n_boundary_faces) array_like
            Defines :math:`\alpha` for Robin boundary condition. Can be defined as a
            scalar or array_like. If array_like, the length of the array must be equal
            to the number of boundary faces.
        beta : scalar or (n_boundary_faces) array_like
            Defines :math:`\beta` for Robin boundary condition. Can be defined as a
            scalar or array_like. If array_like, must have the same length as *alpha* .
        gamma: scalar or (n_boundary_faces) array_like or (n_boundary_faces, n_rhs) array_like
            Defines :math:`\gamma` for Robin boundary condition. If array like, *gamma*
            can have shape (n_boundary_face,). Can also have shape (n_boundary_faces, n_rhs)
            if multiple systems have the same *alpha* and *beta* parameters.

        Returns
        -------
        B : (n_faces, n_cells) scipy.sparse.csr_matrix
            A sparse matrix dependent on the values of *alpha*, *beta* and *gamma* supplied
        b : (n_faces) numpy.ndarray or (n_faces, n_rhs) numpy.ndarray
            A vector dependent on the values of *alpha*, *beta* and *gamma* supplied

        Notes
        -----
        For the gradient of a scalar :math:`\phi`, the weak form is implemented by taking
        the inner product with a piecewise-constant test function :math:`\vec{u}` and integrating
        over the domain:

        .. math::
            \langle \vec{u} , \nabla \phi \rangle \; = \int_\Omega \vec{u} \cdot (\nabla \phi) \, dv

        For a discrete representation of :math:`\phi` at cell centers, the gradient operator
        maps from cell centers to faces. To implement the boundary conditions in this
        case, we must use integration by parts and re-express the inner product as:

        .. math::
            \langle \vec{u} , \nabla \phi \rangle \; = - \int_V \phi \, (\nabla \cdot \vec{u} ) \, dV
            + \oint_{\partial \Omega} \phi \hat{n} \cdot \vec{u} \, da

        where the Robin condition is applied to :math:`\phi` on the
        boundary. The discrete approximation to the above expression is given by:

        .. math::
            \langle \vec{u} , \nabla \phi \rangle \; \approx - \boldsymbol{u^T \big ( D^T M_c - B \big ) \phi + u^T b}

        where :math:`\mathbf{D}` is the :py:attr:`~discretize.operators.DiffOperators.face_divergence`
        and :math:`\mathbf{M_c}` is the cell center inner product matrix
        (just a diagonal matrix comprised of the cell volumes).
        **cell_gradient_weak_form_robin** returns the matrix :math:`\mathbf{B}`
        and vector :math:`\mathbf{b}` based on the parameters *alpha* , *beta*
        and *gamma* provided.

        Examples
        --------
        Here we form all of pieces required to construct the discrete representation
        of the inner product between :math:`\mathbf{u}` for specified Robin boundary
        conditions. We define :math:`\boldsymbol{\phi}` at cell centers and
        :math:`\mathbf{u}` on the faces. We begin by creating a small 2D tensor mesh:

        >>> from discretize import TensorMesh
        >>> import numpy as np
        >>> import scipy.sparse as sp

        >>> h = np.ones(32)
        >>> mesh = TensorMesh([h, h])

        We then define `alpha`, `beta`, and `gamma` parameters for a zero Dirichlet
        condition on the boundary faces. This corresponds to setting:

        >>> alpha = 1.0
        >>> beta = 0.0
        >>> gamma = 0.0

        Next, we construct all of the necessary pieces required to take
        the discrete inner product:

        >>> B, b = mesh.cell_gradient_weak_form_robin(alpha, beta, gamma)
        >>> Mc = sp.diags(mesh.cell_volumes)
        >>> Df = mesh.face_divergence

        In practice, these pieces are usually re-arranged when used to
        solve PDEs with the finite volume method. However, if you wanted
        to create a function which computes the discrete inner product for any
        :math:`\mathbf{u}` and :math:`\boldsymbol{\phi}`:

        >>> def inner_product(u, phi):
        ...     return u @ (-Df.T @ Mc + B) @ phi + u @ b
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
        """Boundary conditions matrix for the cell gradient operator (Deprecated)."""
        warnings.warn(
            "cell_gradient_BC is deprecated and is not longer used. See cell_gradient"
        )
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
        r"""X-derivative operator (cell centers to x-faces).

        This property constructs an x-derivative operator that acts on
        cell centered quantities; i.e. the x-component of the cell gradient operator.
        When applied, the x-derivative is mapped to x-faces. The operator is a
        sparse matrix :math:`\mathbf{G_x}` that can be applied as a matrix-vector
        product to a cell centered quantity :math:`\boldsymbol{\phi}`, i.e.::

            grad_phi_x = Gx @ phi

        By default, the operator assumes zero-Neumann boundary conditions
        on the scalar quantity. Before calling **cell_gradient_x** however,
        the user can set a mix of zero Dirichlet and zero Neumann boundary
        conditions using :py:attr:`~discretize.operators.DiffOperators.set_cell_gradient_BC`.
        When **cell_gradient_x** is called, the boundary conditions are
        enforced for the differencing operator.

        Returns
        -------
        (n_faces_x, n_cells) scipy.sparse.csr_matrix
            X-derivative operator (x-component of the cell gradient)

        Examples
        --------
        Below, we demonstrate how to set boundary conditions for the
        x-component cell gradient, construct the operator and apply it
        to a discrete scalar quantity. The mapping of the operator and
        its sparsity is also illustrated. Our example is carried out on a 2D
        mesh but it can be done equivalently for a 3D mesh.

        We start by importing the necessary packages and modules.

        >>> from discretize import TensorMesh
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> import matplotlib as mpl

        We then construct a mesh and define a scalar function at cell
        centers.

        >>> h = np.ones(40)
        >>> mesh = TensorMesh([h, h], "CC")
        >>> centers = mesh.cell_centers
        >>> phi = np.exp(-(centers[:, 0] ** 2) / 8** 2)

        Before constructing the operator, we must define
        the boundary conditions; zero Neumann for our example. Even though
        we are only computing the derivative along x, we define boundary
        conditions for all boundary faces. Once the
        operator is created, it is applied as a matrix-vector product.

        >>> mesh.set_cell_gradient_BC(['neumann', 'neumann'])
        >>> Gx = mesh.stencil_cell_gradient_x
        >>> grad_phi_x = Gx @ phi

        Now we plot the original scalar, and the x-derivative.

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(13, 5))
            >>> ax1 = fig.add_subplot(121)
            >>> mesh.plot_image(phi, ax=ax1)
            >>> ax1.set_title("Scalar at cell centers", fontsize=14)

            >>> ax2 = fig.add_subplot(122)
            >>> v = np.r_[grad_phi_x, np.zeros(mesh.nFy)]  # Define vector for plotting fun
            >>> mesh.plot_image(v, ax=ax2, v_type="Fx")
            >>> ax2.set_yticks([])
            >>> ax2.set_ylabel("")
            >>> ax2.set_title("X-derivative at x-faces", fontsize=14)
            >>> plt.show()

        The operator is a sparse x-derivative matrix
        that maps from cell centers to x-faces. To demonstrate this, we construct
        a small 2D mesh. We then show the ordering of the elements
        and a spy plot.

        >>> mesh = TensorMesh([[(1, 3)], [(1, 4)]])
        >>> mesh.set_cell_gradient_BC('neumann')

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(12, 8))
            >>> ax1 = fig.add_subplot(121)
            >>> mesh.plot_grid(ax=ax1)
            >>> ax1.set_title("Mapping of Operator", fontsize=14, pad=15)
            >>> ax1.plot(mesh.cell_centers[:, 0], mesh.cell_centers[:, 1], "ro", markersize=8)
            >>> for ii, loc in zip(range(mesh.nC), mesh.cell_centers):
            ...     ax1.text(loc[0] + 0.05, loc[1] + 0.02, "{0:d}".format(ii), color="r")

            >>> ax1.plot(mesh.faces_x[:, 0], mesh.faces_x[:, 1], "g>", markersize=8)
            >>> for ii, loc in zip(range(mesh.nFx), mesh.faces_x):
            ...     ax1.text(loc[0] + 0.05, loc[1] + 0.02, "{0:d}".format(ii), color="g")

            >>> ax1.set_xticks([])
            >>> ax1.set_yticks([])
            >>> ax1.spines['bottom'].set_color('white')
            >>> ax1.spines['top'].set_color('white')
            >>> ax1.spines['left'].set_color('white')
            >>> ax1.spines['right'].set_color('white')
            >>> ax1.set_xlabel('X', fontsize=16, labelpad=-5)
            >>> ax1.set_ylabel('Y', fontsize=16, labelpad=-15)
            >>> ax1.legend(
            ...     ['Mesh', '$\mathbf{\phi}$ (centers)', '$\mathbf{Gx^* \phi}$ (x-faces)'],
            ...     loc='upper right', fontsize=14
            ... )

            >>> ax2 = fig.add_subplot(122)
            >>> ax2.spy(mesh.cell_gradient_x)
            >>> ax2.set_title("Spy Plot", fontsize=14, pad=5)
            >>> ax2.set_ylabel("X-Face Index", fontsize=12)
            >>> ax2.set_xlabel("Cell Index", fontsize=12)
            >>> plt.show()
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
        r"""Y-derivative operator (cell centers to y-faces).

        This property constructs a y-derivative operator that acts on
        cell centered quantities; i.e. the y-component of the cell gradient operator.
        When applied, the y-derivative is mapped to y-faces. The operator is a
        sparse matrix :math:`\mathbf{G_y}` that can be applied as a matrix-vector
        product to a cell centered quantity :math:`\boldsymbol{\phi}`, i.e.::

            grad_phi_y = Gy @ phi

        By default, the operator assumes zero-Neumann boundary conditions
        on the scalar quantity. Before calling **cell_gradient_y** however,
        the user can set a mix of zero Dirichlet and zero Neumann boundary
        conditions using :py:attr:`~discretize.operators.DiffOperators.set_cell_gradient_BC`.
        When **cell_gradient_y** is called, the boundary conditions are
        enforced for the differencing operator.

        Returns
        -------
        (n_faces_y, n_cells) scipy.sparse.csr_matrix
            Y-derivative operator (y-component of the cell gradient)

        Examples
        --------
        Below, we demonstrate how to set boundary conditions for the
        y-component cell gradient, construct the operator and apply it
        to a discrete scalar quantity. The mapping of the operator and
        its sparsity is also illustrated. Our example is carried out on a 2D
        mesh but it can be done equivalently for a 3D mesh.

        We start by importing the necessary packages and modules.

        >>> from discretize import TensorMesh
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> import matplotlib as mpl

        We then construct a mesh and define a scalar function at cell
        centers.

        >>> h = np.ones(40)
        >>> mesh = TensorMesh([h, h], "CC")
        >>> centers = mesh.cell_centers
        >>> phi = np.exp(-(centers[:, 1] ** 2) / 8** 2)

        Before constructing the operator, we must define
        the boundary conditions; zero Neumann for our example. Even though
        we are only computing the derivative along y, we define boundary
        conditions for all boundary faces. Once the
        operator is created, it is applied as a matrix-vector product.

        >>> mesh.set_cell_gradient_BC(['neumann', 'neumann'])
        >>> Gy = mesh.cell_gradient_y
        >>> grad_phi_y = Gy @ phi

        Now we plot the original scalar is y-derivative.

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(13, 5))
            >>> ax1 = fig.add_subplot(121)
            >>> mesh.plot_image(phi, ax=ax1)
            >>> ax1.set_title("Scalar at cell centers", fontsize=14)

            >>> ax2 = fig.add_subplot(122)
            >>> v = np.r_[np.zeros(mesh.nFx), grad_phi_y]  # Define vector for plotting fun
            >>> mesh.plot_image(v, ax=ax2, v_type="Fy")
            >>> ax2.set_yticks([])
            >>> ax2.set_ylabel("")
            >>> ax2.set_title("Y-derivative at y-faces", fontsize=14)
            >>> plt.show()

        The operator is a sparse y-derivative matrix
        that maps from cell centers to y-faces. To demonstrate this, we construct
        a small 2D mesh. We then show the ordering of the elements
        and a spy plot.

        >>> mesh = TensorMesh([[(1, 3)], [(1, 4)]])
        >>> mesh.set_cell_gradient_BC('neumann')

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(12, 8))
            >>> ax1 = fig.add_subplot(121)
            >>> mesh.plot_grid(ax=ax1)
            >>> ax1.set_title("Mapping of Operator", fontsize=14, pad=15)
            >>> ax1.plot(mesh.cell_centers[:, 0], mesh.cell_centers[:, 1], "ro", markersize=8)
            >>> for ii, loc in zip(range(mesh.nC), mesh.cell_centers):
            ...     ax1.text(loc[0] + 0.05, loc[1] + 0.02, "{0:d}".format(ii), color="r")
            >>> ax1.plot(mesh.faces_y[:, 0], mesh.faces_y[:, 1], "g^", markersize=8)
            >>> for ii, loc in zip(range(mesh.nFy), mesh.faces_y):
            ...     ax1.text(loc[0] + 0.05, loc[1] + 0.02, "{0:d}".format(ii + mesh.nFx), color="g")

            >>> ax1.set_xticks([])
            >>> ax1.set_yticks([])
            >>> ax1.spines['bottom'].set_color('white')
            >>> ax1.spines['top'].set_color('white')
            >>> ax1.spines['left'].set_color('white')
            >>> ax1.spines['right'].set_color('white')
            >>> ax1.set_xlabel('X', fontsize=16, labelpad=-5)
            >>> ax1.set_ylabel('Y', fontsize=16, labelpad=-15)
            >>> ax1.legend(
            ...     ['Mesh', '$\mathbf{\phi}$ (centers)', '$\mathbf{Gy^* \phi}$ (y-faces)'],
            ...     loc='upper right', fontsize=14
            ... )

            >>> ax2 = fig.add_subplot(122)
            >>> ax2.spy(mesh.stencil_cell_gradient_y)
            >>> ax2.set_title("Spy Plot", fontsize=14, pad=5)
            >>> ax2.set_ylabel("Y-Face Index", fontsize=12)
            >>> ax2.set_xlabel("Cell Index", fontsize=12)
            >>> plt.show()
        """
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
        r"""Z-derivative operator (cell centers to z-faces).

        This property constructs an z-derivative operator that acts on
        cell centered quantities; i.e. the z-component of the cell gradient operator.
        When applied, the z-derivative is mapped to z-faces. The operator is a
        sparse matrix :math:`\mathbf{G_z}` that can be applied as a matrix-vector
        product to a cell centered quantity :math:`\boldsymbol{\phi}`, i.e.::

            grad_phi_z = Gz @ phi

        By default, the operator assumes zero-Neumann boundary conditions
        on the scalar quantity. Before calling **cell_gradient_z** however,
        the user can set a mix of zero Dirichlet and zero Neumann boundary
        conditions using :py:attr:`~discretize.operators.DiffOperators.set_cell_gradient_BC`.
        When **cell_gradient_z** is called, the boundary conditions are
        enforced for the differencing operator.

        Returns
        -------
        (n_faces_z, n_cells) scipy.sparse.csr_matrix
            Z-derivative operator (z-component of the cell gradient)

        Examples
        --------
        Below, we demonstrate how to set boundary conditions for the
        z-component of the cell gradient, construct the operator and apply it
        to a discrete scalar quantity. The mapping of the operator and
        its sparsity is also illustrated.

        We start by importing the necessary packages and modules.

        >>> from discretize import TensorMesh
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> import matplotlib as mpl

        We then construct a mesh and define a scalar function at cell
        centers.

        >>> h = np.ones(40)
        >>> mesh = TensorMesh([h, h, h], "CCC")
        >>> centers = mesh.cell_centers
        >>> phi = np.exp(-(centers[:, 2] ** 2) / 8** 2)

        Before constructing the operator, we must define
        the boundary conditions; zero Neumann for our example. Even though
        we are only computing the derivative along z, we define boundary
        conditions for all boundary faces. Once the
        operator is created, it is applied as a matrix-vector product.

        >>> mesh.set_cell_gradient_BC(['neumann', 'neumann', 'neumann'])
        >>> Gz = mesh.cell_gradient_z
        >>> grad_phi_z = Gz @ phi

        Now we plot the original scalar and the z-derivative for a slice at y = 0.

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(13, 5))
            >>> ax1 = fig.add_subplot(121)
            >>> mesh.plot_slice(phi, ax=ax1, normal='Y', slice_loc=0)
            >>> ax1.set_title("Scalar at cell centers", fontsize=14)

            >>> ax2 = fig.add_subplot(122)
            >>> v = np.r_[np.zeros(mesh.nFx+mesh.nFy), grad_phi_z]  # Define vector for plotting fun
            >>> mesh.plot_slice(v, ax=ax2, v_type='Fz', normal='Y', slice_loc=0)
            >>> ax2.set_title("Z-derivative (z-faces)", fontsize=14)
            >>> plt.show()

        The z-component cell gradient is a sparse derivative matrix
        that maps from cell centers to z-faces. To demonstrate this, we provide
        a spy plot

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(9, 9))
            >>> ax1 = fig.add_subplot(111)
            >>> ax1.spy(mesh.cell_gradient_z, ms=1)
            >>> ax1.set_title("Spy Plot", fontsize=16, pad=5)
            >>> ax1.set_xlabel("Cell Index", fontsize=12)
            >>> ax1.set_ylabel("Z-Face Index", fontsize=12)
            >>> plt.show()
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
        """Stencil for the edge curl operator in the x-direction."""
        n = self.vnC  # The number of cell centers in each direction

        D32 = kron3(ddx(n[2]), speye(n[1]), speye(n[0] + 1))
        D23 = kron3(speye(n[2]), ddx(n[1]), speye(n[0] + 1))
        # O1 = spzeros(np.shape(D32)[0], np.shape(D31)[1])
        O1 = spzeros((n[0] + 1) * n[1] * n[2], n[0] * (n[1] + 1) * (n[2] + 1))

        return sp.hstack((O1, -D32, D23))

    @property
    def _edge_y_curl_stencil(self):
        """Stencil for the edge curl operator in the y-direction."""
        n = self.vnC  # The number of cell centers in each direction

        D31 = kron3(ddx(n[2]), speye(n[1] + 1), speye(n[0]))
        D13 = kron3(speye(n[2]), speye(n[1] + 1), ddx(n[0]))
        # O2 = spzeros(np.shape(D31)[0], np.shape(D32)[1])
        O2 = spzeros(n[0] * (n[1] + 1) * n[2], (n[0] + 1) * n[1] * (n[2] + 1))

        return sp.hstack((D31, O2, -D13))

    @property
    def _edge_z_curl_stencil(self):
        """Stencil for the edge curl operator in the z-direction."""
        n = self.vnC  # The number of cell centers in each direction

        D21 = kron3(speye(n[2] + 1), ddx(n[1]), speye(n[0]))
        D12 = kron3(speye(n[2] + 1), speye(n[1]), ddx(n[0]))
        # O3 = spzeros(np.shape(D21)[0], np.shape(D13)[1])
        O3 = spzeros(n[0] * n[1] * (n[2] + 1), (n[0] + 1) * (n[1] + 1) * n[2])

        return sp.hstack((-D21, D12, O3))

    @property
    def _edge_curl_stencil(self):
        """Full stencil for the edge curl operator."""
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
    def edge_curl(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        if getattr(self, "_edge_curl", None) is None:
            if self.dim <= 1:
                raise NotImplementedError("Edge Curl only programed for 2 or 3D.")
            L = self.edge_lengths  # Compute lengths of cell edges
            if self.dim == 2:
                S = self.cell_volumes
            elif self.dim == 3:
                S = self.face_areas
            self._edge_curl = sdiag(1 / S) @ self._edge_curl_stencil @ sdiag(L)
        return self._edge_curl

    @property
    def boundary_face_scalar_integral(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
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
        r"""Integrate a vector function on the boundary.

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

        .. math:: w^T * P @ u_b

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
        r"""Integrate a vector function dotted with the boundary normal.

        This matrix represents the boundary surface integral of a vector function
        dotted with the boundary normal and multiplied with a scalar finite volume
        test function on the mesh.

        Returns
        -------
        (n_nodes, dim * n_boundary_nodes) scipy.sparse.csr_matrix
            Sparse matrix of shape.

        Notes
        -----
        The integral we are representing on the boundary of the mesh is

        .. math:: \int_{\Omega} (w \vec{u}) \cdot \hat{n} \partial \Omega

        In discrete form this is:

        .. math:: w^T * P @ u_b

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
        """Create the weak form boundary condition projection matrices.

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
        """Create weak form boundary condition projection matrices for mixed boundary condition."""
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
    def average_face_to_cell(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
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
    def average_face_to_cell_vector(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
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
        r"""Averaging operator from x-faces to cell centers (scalar quantities).

        This property constructs a 2nd order averaging operator that maps scalar
        quantities from x-faces to cell centers. This averaging operator is
        used when a discrete scalar quantity defined on x-faces must be
        projected to cell centers. Once constructed, the operator is stored
        permanently as a property of the mesh. *See notes*.

        Returns
        -------
        (n_cells, n_faces_x) scipy.sparse.csr_matrix
            The scalar averaging operator from x-faces to cell centers

        Notes
        -----
        Let :math:`\boldsymbol{\phi_x}` be a discrete scalar quantity that
        lives on x-faces. **average_face_x_to_cell** constructs a discrete
        linear operator :math:`\mathbf{A_{xc}}` that projects
        :math:`\boldsymbol{\phi_x}` to cell centers, i.e.:

        .. math::
            \boldsymbol{\phi_c} = \mathbf{A_{xc}} \, \boldsymbol{\phi_x}

        where :math:`\boldsymbol{\phi_c}` approximates the value of the scalar
        quantity at cell centers. For each cell, we are simply averaging
        the values defined on its x-faces. The operation is implemented as a
        matrix vector product, i.e.::

            phi_c = Axc @ phi_x

        Examples
        --------
        Here we compute the values of a scalar function on the x-faces. We then create
        an averaging operator to approximate the function at cell centers. We choose
        to define a scalar function that is strongly discontinuous in some places to
        demonstrate how the averaging operator will smooth out discontinuities.

        We start by importing the necessary packages and defining a mesh.

        >>> from discretize import TensorMesh
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        >>> h = np.ones(40)
        >>> mesh = TensorMesh([h, h], x0="CC")

        Create a scalar variable on x-faces:

        >>> phi_x = np.zeros(mesh.nFx)
        >>> xy = mesh.faces_x
        >>> phi_x[(xy[:, 1] > 0)] = 25.0
        >>> phi_x[(xy[:, 1] < -10.0) & (xy[:, 0] > -10.0) & (xy[:, 0] < 10.0)] = 50.0

        Next, we construct the averaging operator and apply it to
        the discrete scalar quantity to approximate the value at cell centers.

        >>> Axc = mesh.average_face_x_to_cell
        >>> phi_c = Axc @ phi_x

        And plot the results:

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(11, 5))
            >>> ax1 = fig.add_subplot(121)
            >>> v = np.r_[phi_x, np.zeros(mesh.nFy)]  # create vector for plotting function
            >>> mesh.plot_image(v, ax=ax1, v_type="Fx")
            >>> ax1.set_title("Variable at x-faces", fontsize=16)
            >>> ax2 = fig.add_subplot(122)
            >>> mesh.plot_image(phi_c, ax=ax2, v_type="CC")
            >>> ax2.set_title("Averaged to cell centers", fontsize=16)
            >>> plt.show()

        Below, we show a spy plot illustrating the sparsity and mapping
        of the operator

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(9, 9))
            >>> ax1 = fig.add_subplot(111)
            >>> ax1.spy(Axc, ms=1)
            >>> ax1.set_title("X-Face Index", fontsize=12, pad=5)
            >>> ax1.set_ylabel("Cell Index", fontsize=12)
            >>> plt.show()
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
        r"""Averaging operator from y-faces to cell centers (scalar quantities).

        This property constructs a 2nd order averaging operator that maps scalar
        quantities from y-faces to cell centers. This averaging operator is
        used when a discrete scalar quantity defined on x-faces must be
        projected to cell centers. Once constructed, the operator is stored
        permanently as a property of the mesh. *See notes*.

        Returns
        -------
        (n_cells, n_faces_y) scipy.sparse.csr_matrix
            The scalar averaging operator from y-faces to cell centers

        Notes
        -----
        Let :math:`\boldsymbol{\phi_y}` be a discrete scalar quantity that
        lives on y-faces. **average_face_y_to_cell** constructs a discrete
        linear operator :math:`\mathbf{A_{yc}}` that projects
        :math:`\boldsymbol{\phi_y}` to cell centers, i.e.:

        .. math::
            \boldsymbol{\phi_c} = \mathbf{A_{yc}} \, \boldsymbol{\phi_y}

        where :math:`\boldsymbol{\phi_c}` approximates the value of the scalar
        quantity at cell centers. For each cell, we are simply averaging
        the values defined on its y-faces. The operation is implemented as a
        matrix vector product, i.e.::

            phi_c = Ayc @ phi_y

        Examples
        --------
        Here we compute the values of a scalar function on the y-faces. We then create
        an averaging operator to approximate the function at cell centers. We choose
        to define a scalar function that is strongly discontinuous in some places to
        demonstrate how the averaging operator will smooth out discontinuities.

        We start by importing the necessary packages and defining a mesh.

        >>> from discretize import TensorMesh
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        >>> h = np.ones(40)
        >>> mesh = TensorMesh([h, h], x0="CC")

        Create a scalar variable on y-faces,

        >>> phi_y = np.zeros(mesh.nFy)
        >>> xy = mesh.faces_y
        >>> phi_y[(xy[:, 1] > 0)] = 25.0
        >>> phi_y[(xy[:, 1] < -10.0) & (xy[:, 0] > -10.0) & (xy[:, 0] < 10.0)] = 50.0

        Next, we construct the averaging operator and apply it to
        the discrete scalar quantity to approximate the value at cell centers.

        >>> Ayc = mesh.average_face_y_to_cell
        >>> phi_c = Ayc @ phi_y

        And finally, plot the results:

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(11, 5))
            >>> ax1 = fig.add_subplot(121)
            >>> v = np.r_[np.zeros(mesh.nFx), phi_y]  # create vector for plotting function
            >>> mesh.plot_image(v, ax=ax1, v_type="Fy")
            >>> ax1.set_title("Variable at y-faces", fontsize=16)
            >>> ax2 = fig.add_subplot(122)
            >>> mesh.plot_image(phi_c, ax=ax2, v_type="CC")
            >>> ax2.set_title("Averaged to cell centers", fontsize=16)
            >>> plt.show()

        Below, we show a spy plot illustrating the sparsity and mapping
        of the operator

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(9, 9))
            >>> ax1 = fig.add_subplot(111)
            >>> ax1.spy(Ayc, ms=1)
            >>> ax1.set_title("Y-Face Index", fontsize=12, pad=5)
            >>> ax1.set_ylabel("Cell Index", fontsize=12)
            >>> plt.show()
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
        r"""Averaging operator from z-faces to cell centers (scalar quantities).

        This property constructs a 2nd order averaging operator that maps scalar
        quantities from z-faces to cell centers. This averaging operator is
        used when a discrete scalar quantity defined on z-faces must be
        projected to cell centers. Once constructed, the operator is stored
        permanently as a property of the mesh. *See notes*.

        Returns
        -------
        (n_cells, n_faces_z) scipy.sparse.csr_matrix
            The scalar averaging operator from z-faces to cell centers

        Notes
        -----
        Let :math:`\boldsymbol{\phi_z}` be a discrete scalar quantity that
        lives on z-faces. **average_face_z_to_cell** constructs a discrete
        linear operator :math:`\mathbf{A_{zc}}` that projects
        :math:`\boldsymbol{\phi_z}` to cell centers, i.e.:

        .. math::
            \boldsymbol{\phi_c} = \mathbf{A_{zc}} \, \boldsymbol{\phi_z}

        where :math:`\boldsymbol{\phi_c}` approximates the value of the scalar
        quantity at cell centers. For each cell, we are simply averaging
        the values defined on its z-faces. The operation is implemented as a
        matrix vector product, i.e.::

            phi_c = Azc @ phi_z

        Examples
        --------
        Here we compute the values of a scalar function on the z-faces. We then create
        an averaging operator to approximate the function at cell centers. We choose
        to define a scalar function that is strongly discontinuous in some places to
        demonstrate how the averaging operator will smooth out discontinuities.

        We start by importing the necessary packages and defining a mesh.

        >>> from discretize import TensorMesh
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        >>> h = np.ones(40)
        >>> mesh = TensorMesh([h, h, h], x0="CCC")

        Create a scalar variable on z-faces

        >>> phi_z = np.zeros(mesh.nFz)
        >>> xyz = mesh.faces_z
        >>> phi_z[(xyz[:, 2] > 0)] = 25.0
        >>> phi_z[(xyz[:, 2] < -10.0) & (xyz[:, 0] > -10.0) & (xyz[:, 0] < 10.0)] = 50.0

        Next, we construct the averaging operator and apply it to
        the discrete scalar quantity to approximate the value at cell centers.
        We plot the original scalar and its average at cell centers for a
        slice at y=0.

        >>> Azc = mesh.average_face_z_to_cell
        >>> phi_c = Azc @ phi_z

        And plot the results:

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(11, 5))
            >>> ax1 = fig.add_subplot(121)
            >>> v = np.r_[np.zeros(mesh.nFx+mesh.nFy), phi_z]  # create vector for plotting
            >>> mesh.plot_slice(v, ax=ax1, normal='Y', slice_loc=0, v_type="Fz")
            >>> ax1.set_title("Variable at z-faces", fontsize=16)
            >>> ax2 = fig.add_subplot(122)
            >>> mesh.plot_image(phi_c, ax=ax2, normal='Y', slice_loc=0, v_type="CC")
            >>> ax2.set_title("Averaged to cell centers", fontsize=16)
            >>> plt.show()

        Below, we show a spy plot illustrating the sparsity and mapping
        of the operator

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(9, 9))
            >>> ax1 = fig.add_subplot(111)
            >>> ax1.spy(Azc, ms=1)
            >>> ax1.set_title("Z-Face Index", fontsize=12, pad=5)
            >>> ax1.set_ylabel("Cell Index", fontsize=12)
            >>> plt.show()
        """
        if self.dim < 3:
            return None
        if getattr(self, "_average_face_z_to_cell", None) is None:
            n = self.vnC
            if self.dim == 3:
                self._average_face_z_to_cell = kron3(av(n[2]), speye(n[1]), speye(n[0]))
        return self._average_face_z_to_cell

    @property
    def average_cell_to_face(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
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
    def average_cell_vector_to_face(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
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
    def average_cell_to_edge(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
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
    def average_edge_to_cell(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
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
    def average_edge_to_cell_vector(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
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
        r"""Averaging operator from x-edges to cell centers (scalar quantities).

        This property constructs a 2nd order averaging operator that maps scalar
        quantities from x-edges to cell centers. This averaging operator is
        used when a discrete scalar quantity defined on x-edges must be
        projected to cell centers. Once constructed, the operator is stored
        permanently as a property of the mesh. *See notes*.

        Returns
        -------
        (n_cells, n_edges_x) scipy.sparse.csr_matrix
            The scalar averaging operator from x-edges to cell centers

        Notes
        -----
        Let :math:`\boldsymbol{\phi_x}` be a discrete scalar quantity that
        lives on x-edges. **average_edge_x_to_cell** constructs a discrete
        linear operator :math:`\mathbf{A_{xc}}` that projects
        :math:`\boldsymbol{\phi_x}` to cell centers, i.e.:

        .. math::
            \boldsymbol{\phi_c} = \mathbf{A_{xc}} \, \boldsymbol{\phi_x}

        where :math:`\boldsymbol{\phi_c}` approximates the value of the scalar
        quantity at cell centers. For each cell, we are simply averaging
        the values defined on its x-edges. The operation is implemented as a
        matrix vector product, i.e.::

            phi_c = Axc @ phi_x

        Examples
        --------
        Here we compute the values of a scalar function on the x-edges. We then create
        an averaging operator to approximate the function at cell centers. We choose
        to define a scalar function that is strongly discontinuous in some places to
        demonstrate how the averaging operator will smooth out discontinuities.

        We start by importing the necessary packages and defining a mesh.

        >>> from discretize import TensorMesh
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> h = np.ones(40)
        >>> mesh = TensorMesh([h, h], x0="CC")

        Then we create a scalar variable on x-edges,

        >>> phi_x = np.zeros(mesh.nEx)
        >>> xy = mesh.edges_x
        >>> phi_x[(xy[:, 1] > 0)] = 25.0
        >>> phi_x[(xy[:, 1] < -10.0) & (xy[:, 0] > -10.0) & (xy[:, 0] < 10.0)] = 50.0

        Next, we construct the averaging operator and apply it to
        the discrete scalar quantity to approximate the value at cell centers.

        >>> Axc = mesh.average_edge_x_to_cell
        >>> phi_c = Axc @ phi_x

        And plot the results,

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(11, 5))
            >>> ax1 = fig.add_subplot(121)
            >>> v = np.r_[phi_x, np.zeros(mesh.nEy)]  # create vector for plotting function
            >>> mesh.plot_image(v, ax=ax1, v_type="Ex")
            >>> ax1.set_title("Variable at x-edges", fontsize=16)
            >>> ax2 = fig.add_subplot(122)
            >>> mesh.plot_image(phi_c, ax=ax2, v_type="CC")
            >>> ax2.set_title("Averaged to cell centers", fontsize=16)
            >>> plt.show()

        Below, we show a spy plot illustrating the sparsity and mapping
        of the operator

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(9, 9))
            >>> ax1 = fig.add_subplot(111)
            >>> ax1.spy(Axc, ms=1)
            >>> ax1.set_title("X-Edge Index", fontsize=12, pad=5)
            >>> ax1.set_ylabel("Cell Index", fontsize=12)
            >>> plt.show()
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
        r"""Averaging operator from y-edges to cell centers (scalar quantities).

        This property constructs a 2nd order averaging operator that maps scalar
        quantities from y-edges to cell centers. This averaging operator is
        used when a discrete scalar quantity defined on y-edges must be
        projected to cell centers. Once constructed, the operator is stored
        permanently as a property of the mesh. *See notes*.

        Returns
        -------
        (n_cells, n_edges_y) scipy.sparse.csr_matrix
            The scalar averaging operator from y-edges to cell centers

        Notes
        -----
        Let :math:`\boldsymbol{\phi_y}` be a discrete scalar quantity that
        lives on y-edges. **average_edge_y_to_cell** constructs a discrete
        linear operator :math:`\mathbf{A_{yc}}` that projects
        :math:`\boldsymbol{\phi_y}` to cell centers, i.e.:

        .. math::
            \boldsymbol{\phi_c} = \mathbf{A_{yc}} \, \boldsymbol{\phi_y}

        where :math:`\boldsymbol{\phi_c}` approximates the value of the scalar
        quantity at cell centers. For each cell, we are simply averaging
        the values defined on its y-edges. The operation is implemented as a
        matrix vector product, i.e.::

            phi_c = Ayc @ phi_y

        Examples
        --------
        Here we compute the values of a scalar function on the y-edges. We then create
        an averaging operator to approximate the function at cell centers. We choose
        to define a scalar function that is strongly discontinuous in some places to
        demonstrate how the averaging operator will smooth out discontinuities.

        We start by importing the necessary packages and defining a mesh.

        >>> from discretize import TensorMesh
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> h = np.ones(40)
        >>> mesh = TensorMesh([h, h], x0="CC")

        Then we create a scalar variable on y-edges,

        >>> phi_y = np.zeros(mesh.nEy)
        >>> xy = mesh.edges_y
        >>> phi_y[(xy[:, 1] > 0)] = 25.0
        >>> phi_y[(xy[:, 1] < -10.0) & (xy[:, 0] > -10.0) & (xy[:, 0] < 10.0)] = 50.0

        Next, we construct the averaging operator and apply it to
        the discrete scalar quantity to approximate the value at cell centers.

        >>> Ayc = mesh.average_edge_y_to_cell
        >>> phi_c = Ayc @ phi_y

        And plot the results,

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(11, 5))
            >>> ax1 = fig.add_subplot(121)
            >>> v = np.r_[np.zeros(mesh.nEx), phi_y]  # create vector for plotting function
            >>> mesh.plot_image(v, ax=ax1, v_type="Ey")
            >>> ax1.set_title("Variable at y-edges", fontsize=16)
            >>> ax2 = fig.add_subplot(122)
            >>> mesh.plot_image(phi_c, ax=ax2, v_type="CC")
            >>> ax2.set_title("Averaged to cell centers", fontsize=16)
            >>> plt.show()

        Below, we show a spy plot illustrating the sparsity and mapping
        of the operator

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(9, 9))
            >>> ax1 = fig.add_subplot(111)
            >>> ax1.spy(Ayc, ms=1)
            >>> ax1.set_title("Y-Edge Index", fontsize=12, pad=5)
            >>> ax1.set_ylabel("Cell Index", fontsize=12)
            >>> plt.show()
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
        r"""Averaging operator from z-edges to cell centers (scalar quantities).

        This property constructs a 2nd order averaging operator that maps scalar
        quantities from z-edges to cell centers. This averaging operator is
        used when a discrete scalar quantity defined on z-edges must be
        projected to cell centers. Once constructed, the operator is stored
        permanently as a property of the mesh. *See notes*.

        Returns
        -------
        (n_cells, n_edges_z) scipy.sparse.csr_matrix
            The scalar averaging operator from z-edges to cell centers

        Notes
        -----
        Let :math:`\boldsymbol{\phi_z}` be a discrete scalar quantity that
        lives on z-edges. **average_edge_z_to_cell** constructs a discrete
        linear operator :math:`\mathbf{A_{zc}}` that projects
        :math:`\boldsymbol{\phi_z}` to cell centers, i.e.:

        .. math::
            \boldsymbol{\phi_c} = \mathbf{A_{zc}} \, \boldsymbol{\phi_z}

        where :math:`\boldsymbol{\phi_c}` approximates the value of the scalar
        quantity at cell centers. For each cell, we are simply averaging
        the values defined on its z-edges. The operation is implemented as a
        matrix vector product, i.e.::

            phi_c = Azc @ phi_z

        Examples
        --------
        Here we compute the values of a scalar function on the z-edges. We then create
        an averaging operator to approximate the function at cell centers. We choose
        to define a scalar function that is strongly discontinuous in some places to
        demonstrate how the averaging operator will smooth out discontinuities.

        We start by importing the necessary packages and defining a mesh.

        >>> from discretize import TensorMesh
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> h = np.ones(40)
        >>> mesh = TensorMesh([h, h, h], x0="CCC")

        The we create a scalar variable on z-edges,

        >>> phi_z = np.zeros(mesh.nEz)
        >>> xyz = mesh.edges_z
        >>> phi_z[(xyz[:, 2] > 0)] = 25.0
        >>> phi_z[(xyz[:, 2] < -10.0) & (xyz[:, 0] > -10.0) & (xyz[:, 0] < 10.0)] = 50.0

        Next, we construct the averaging operator and apply it to
        the discrete scalar quantity to approximate the value at cell centers.
        We plot the original scalar and its average at cell centers for a
        slice at y=0.

        >>> Azc = mesh.average_edge_z_to_cell
        >>> phi_c = Azc @ phi_z

        Plot the results,

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(11, 5))
            >>> ax1 = fig.add_subplot(121)
            >>> v = np.r_[np.zeros(mesh.nEx+mesh.nEy), phi_z]  # create vector for plotting
            >>> mesh.plot_slice(v, ax=ax1, normal='Y', slice_loc=0, v_type="Ez")
            >>> ax1.set_title("Variable at z-edges", fontsize=16)
            >>> ax2 = fig.add_subplot(122)
            >>> mesh.plot_image(phi_c, ax=ax2, normal='Y', slice_loc=0, v_type="CC")
            >>> ax2.set_title("Averaged to cell centers", fontsize=16)
            >>> plt.show()

        Below, we show a spy plot illustrating the sparsity and mapping
        of the operator

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(9, 9))
            >>> ax1 = fig.add_subplot(111)
            >>> ax1.spy(Azc, ms=1)
            >>> ax1.set_title("Z-Edge Index", fontsize=12, pad=5)
            >>> ax1.set_ylabel("Cell Index", fontsize=12)
            >>> plt.show()
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
    def average_edge_to_face_vector(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
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
    def average_node_to_cell(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
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
        """Averaging operator on cell nodes to x-edges."""
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
        """Averaging operator on cell nodes to y-edges."""
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
        """Averaging operator on cell nodes to z-edges."""
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
    def average_node_to_edge(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
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
    def average_node_to_face(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
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
    def project_face_to_boundary_face(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
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
    def project_edge_to_boundary_edge(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
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
    def project_node_to_boundary_node(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        # Simple matrix which projects the values of the nodes onto the boundary nodes
        # Can also be used to "select" the boundary nodes

        # Create a matrix that projects all nodes onto boundary nodes
        # The below should work for a regular structured mesh

        is_b = make_boundary_bool(self.shape_nodes)
        return sp.eye(self.n_nodes, format="csr")[is_b]

    # DEPRECATED
    cellGrad = deprecate_property(
        "cell_gradient", "cellGrad", removal_version="1.0.0", future_warn=True
    )
    cellGradBC = deprecate_property(
        "cell_gradient_BC", "cellGradBC", removal_version="1.0.0", future_warn=True
    )
    cellGradx = deprecate_property(
        "cell_gradient_x", "cellGradx", removal_version="1.0.0", future_warn=True
    )
    cellGrady = deprecate_property(
        "cell_gradient_y", "cellGrady", removal_version="1.0.0", future_warn=True
    )
    cellGradz = deprecate_property(
        "cell_gradient_z", "cellGradz", removal_version="1.0.0", future_warn=True
    )
    faceDivx = deprecate_property(
        "face_x_divergence", "faceDivx", removal_version="1.0.0", future_warn=True
    )
    faceDivy = deprecate_property(
        "face_y_divergence", "faceDivy", removal_version="1.0.0", future_warn=True
    )
    faceDivz = deprecate_property(
        "face_z_divergence", "faceDivz", removal_version="1.0.0", future_warn=True
    )
    _cellGradStencil = deprecate_property(
        "stencil_cell_gradient",
        "_cellGradStencil",
        removal_version="1.0.0",
        future_warn=True,
    )
    _cellGradxStencil = deprecate_property(
        "stencil_cell_gradient_x",
        "_cellGradxStencil",
        removal_version="1.0.0",
        future_warn=True,
    )
    _cellGradyStencil = deprecate_property(
        "stencil_cell_gradient_y",
        "_cellGradyStencil",
        removal_version="1.0.0",
        future_warn=True,
    )
    _cellGradzStencil = deprecate_property(
        "stencil_cell_gradient_z",
        "_cellGradzStencil",
        removal_version="1.0.0",
        future_warn=True,
    )

    setCellGradBC = deprecate_method(
        "set_cell_gradient_BC",
        "setCellGradBC",
        removal_version="1.0.0",
        future_warn=True,
    )
    getBCProjWF = deprecate_method(
        "get_BC_projections", "getBCProjWF", removal_version="1.0.0", future_warn=True
    )
    getBCProjWF_simple = deprecate_method(
        "get_BC_projections_simple",
        "getBCProjWF_simple",
        removal_version="1.0.0",
        future_warn=True,
    )
