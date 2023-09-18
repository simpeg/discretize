"""Module housing the TensorMesh implementation."""
import itertools
import numpy as np

from discretize.base import BaseRectangularMesh, BaseTensorMesh
from discretize.operators import DiffOperators, InnerProducts
from discretize.mixins import InterfaceMixins, TensorMeshIO
from discretize.utils import mkvc
from discretize.utils.code_utils import deprecate_property

from .tensor_cell import TensorCell


class TensorMesh(
    DiffOperators,
    InnerProducts,
    BaseTensorMesh,
    BaseRectangularMesh,
    TensorMeshIO,
    InterfaceMixins,
):
    """
    Tensor mesh class.

    Tensor meshes are numerical grids whose cell centers, nodes, faces, edges, widths,
    volumes, etc... can be directly expressed as tensor products. The axes defining
    coordinates of the mesh are orthogonal. And cell properties along one axis do
    not vary with respect to the position along any other axis.

    Parameters
    ----------
    h : (dim) iterable of int, numpy.ndarray, or tuple
        Defines the cell widths along each axis. The length of the iterable object is
        equal to the dimension of the mesh (1, 2 or 3). For a 3D mesh, the list would
        have the form *[hx, hy, hz]* .

        Along each axis, the user has 3 choices for defining the cells widths:

        - :class:`int` -> A unit interval is equally discretized into `N` cells.
        - :class:`numpy.ndarray` -> The widths are explicity given for each cell
        - the widths are defined as a :class:`list` of :class:`tuple` of the form *(dh, nc, [npad])*
          where *dh* is the cell width, *nc* is the number of cells, and *npad* (optional)
          is a padding factor denoting exponential increase/decrease in the cell width
          for each cell; e.g. *[(2., 10, -1.3), (2., 50), (2., 10, 1.3)]*

    origin : (dim) iterable, default: 0
        Define the origin or 'anchor point' of the mesh; i.e. the bottom-left-frontmost
        corner. By default, the mesh is anchored such that its origin is at *[0, 0, 0]* .

        For each dimension (x, y or z), The user may set the origin 2 ways:

        - a ``scalar`` which explicitly defines origin along that dimension.
        - **{'0', 'C', 'N'}** a :class:`str` specifying whether the zero coordinate along
          each axis is the first node location ('0'), in the center ('C') or the last
          node location ('N') (see Examples).

    See Also
    --------
    utils.unpack_widths :
        The function used to expand a tuple to generate widths.

    Examples
    --------
    An example of a 2D tensor mesh is shown below. Here we use a list of tuple to
    define the discretization along the x-axis and a numpy array to define the
    discretization along the y-axis. We also use a string argument to center the
    x-axis about x = 0 and set the top of the mesh to y = 0.

    >>> from discretize import TensorMesh
    >>> import matplotlib.pyplot as plt

    >>> ncx = 10      # number of core mesh cells in x
    >>> dx = 5        # base cell width x
    >>> npad_x = 3    # number of padding cells in x
    >>> exp_x = 1.25  # expansion rate of padding cells in x
    >>> ncy = 24      # total number of mesh cells in y
    >>> dy = 5        # base cell width y

    >>> hx = [(dx, npad_x, -exp_x), (dx, ncx), (dx, npad_x, exp_x)]
    >>> hy = dy * np.ones(ncy)
    >>> mesh = TensorMesh([hx, hy], origin='CN')

    >>> fig = plt.figure(figsize=(5,5))
    >>> ax = fig.add_subplot(111)
    >>> mesh.plot_grid(ax=ax)
    >>> plt.show()
    """

    _meshType = "TENSOR"
    _aliases = {
        **DiffOperators._aliases,
        **BaseRectangularMesh._aliases,
        **BaseTensorMesh._aliases,
    }

    def __repr__(self):
        """Plain text representation."""
        fmt = "\n  {}: {:,} cells\n\n".format(type(self).__name__, self.nC)
        fmt += 22 * " " + "MESH EXTENT" + 13 * " " + "CELL WIDTH      FACTOR\n"
        fmt += "  dir    nC        min           max         min       max "
        fmt += "     max\n  ---   ---  " + 27 * "-" + "  " + 18 * "-" + "  ------\n"

        # Get attributes and put into table.
        attrs = self._repr_attributes()
        for i in range(self.dim):
            name = attrs["names"][i]
            iattr = attrs[name]
            fmt += "   {}".format(name)
            fmt += " {:6}".format(iattr["nC"])
            for p in ["min", "max"]:
                fmt += " {:13,.2f}".format(iattr[p])
            for p in ["h_min", "h_max"]:
                fmt += " {:9,.2f}".format(iattr[p])
            fmt += "{:8,.2f}".format(iattr["max_fact"])
            fmt += "\n"  # End row

        fmt += "\n"
        return fmt

    def _repr_html_(self):
        """HTML representation."""
        style = " style='padding: 5px 20px 5px 20px;'"

        fmt = "<table>\n"
        fmt += "  <tr>\n"
        fmt += "    <td style='font-weight: bold; font-size: 1.2em; text-align"
        fmt += ": center;' colspan='3'>{}</td>\n".format(type(self).__name__)
        fmt += "    <td style='font-size: 1.2em; text-align: center;'"
        fmt += "colspan='4'>{:,} cells</td>\n".format(self.nC)
        fmt += "  </tr>\n"

        fmt += "  <tr>\n"
        fmt += "    <th></th>\n"
        fmt += "    <th></th>\n"
        fmt += "    <th colspan='2'" + style + ">MESH EXTENT</th>\n"
        fmt += "    <th colspan='2'" + style + ">CELL WIDTH</th>\n"
        fmt += "    <th" + style + ">FACTOR</th>\n"
        fmt += "  </tr>\n"

        fmt += "  <tr>\n"
        fmt += "    <th" + style + ">dir</th>\n"
        fmt += "    <th" + style + ">nC</th>\n"
        fmt += "    <th" + style + ">min</th>\n"
        fmt += "    <th" + style + ">max</th>\n"
        fmt += "    <th" + style + ">min</th>\n"
        fmt += "    <th" + style + ">max</th>\n"
        fmt += "    <th" + style + ">max</th>\n"
        fmt += "  </tr>\n"

        # Get attributes and put into table.
        attrs = self._repr_attributes()
        for i in range(self.dim):
            name = attrs["names"][i]
            iattr = attrs[name]
            fmt += "  <tr>\n"  # Start row
            fmt += "    <td" + style + ">{}</td>\n".format(name)
            fmt += "    <td" + style + ">{}</td>\n".format(iattr["nC"])
            for p in ["min", "max", "h_min", "h_max", "max_fact"]:
                fmt += "    <td" + style + ">{:,.2f}</td>\n".format(iattr[p])
            fmt += "  </tr>\n"  # End row

        fmt += "</table>\n"
        return fmt

    def __iter__(self):
        """Iterate over the cells."""
        iterator = (self[i] for i in range(len(self)))
        return iterator

    def __getitem__(self, indices):
        """
        Return the boundaries of a single cell of the mesh.

        Parameters
        ----------
        indices : int, slice, or tuple of int and slices
            Indices of a cell in the mesh.
            It can be a single integer or a single slice (for ravelled
            indices), or a tuple combining integers and slices for each
            direction.

        Returns
        -------
        TensorCell or list of TensorCell
        """
        # Handle non tuple indices
        if isinstance(indices, slice):
            cells = [self[i] for i in _slice_to_index(indices, len(self))]
            return cells
        if np.issubdtype(type(indices), np.integer):
            indices = self._sanitize_indices(indices)
            indices = np.unravel_index(indices, self.shape_cells, order="F")
        # Handle tuple indices
        if not isinstance(indices, tuple):
            raise ValueError(
                f"Invalid indices '{indices}'. "
                "It should be an int, a slice or a tuple of int and slices."
            )
        if len(indices) != self.dim:
            raise ValueError(
                f"Invalid number of indices '{len(indices)}'. "
                f"It should match the number of dimensions of the mesh ({self.dim})."
            )
        # Int indices only
        all_indices_are_ints = all(np.issubdtype(type(i), np.integer) for i in indices)
        if all_indices_are_ints:
            indices = self._sanitize_indices(indices)
            return self._get_cell(indices)
        # Slice and int indices
        indices_per_dim = [
            _slice_to_index(index, self.shape_cells[dim])
            if isinstance(index, slice)
            else [self._sanitize_indices(index, dim=dim)]
            for dim, index in enumerate(indices)
        ]
        # Combine the indices_per_dim using itertools.product.
        # Because we want to follow a FORTRAN order, we need to reverse the
        # order of the indices_per_dim and the indices.
        indices = (i[::-1] for i in itertools.product(*indices_per_dim[::-1]))
        cells = [self._get_cell(i) for i in indices]
        if not cells:
            return None
        return cells

    def _sanitize_indices(self, indices, dim=None):
        """
        Sanitize integer indices for cell in the mesh.

        Convert negative indices into their corresponding positive values
        within the mesh. It works with a tuple of indices or with
        single int (ravelled indices).

        Parameters
        ----------
        indices : int or tuple of int
            Indices of a single mesh cell. It can contain negative indices.
        dim : int or None
            Corresponding dimension of ``indices``, if it's a single int. If
            None and ``indices`` is an int, then ``indices`` will be assumed to
            be a ravelled index. If ``indices`` is a tuple, ``dim`` is ignored.

        Returns
        -------
        int or tuple of int
        """
        if isinstance(indices, tuple):
            indices = tuple(
                index if index >= 0 else index + self.shape_cells[i]
                for i, index in enumerate(indices)
            )
        elif indices < 0:
            if dim is None:
                indices = indices + self.n_cells
            else:
                indices = indices + self.shape_cells[dim]
        return indices

    def _get_cell(self, indices):
        """Return a single cell in the mesh.

        Parameters
        ----------
        indices : tuple of int
            Tuple containing the indices of the cell. Must have the same number
            of elements as the mesh dimensions.

        Returns
        -------
        TensorCell
        """
        assert all(index >= 0 for index in indices)
        if self.dim == 1:
            (i,) = indices
            x1, x2 = self.nodes_x[i], self.nodes_x[i + 1]
            origin = np.array([x1])
            h = np.array([x2 - x1])
        if self.dim == 2:
            i, j = indices
            x1, x2 = self.nodes_x[i], self.nodes_x[i + 1]
            y1, y2 = self.nodes_y[j], self.nodes_y[j + 1]
            origin = np.array([x1, y1])
            h = np.array([x2 - x1, y2 - y1])
        if self.dim == 3:
            i, j, k = indices
            x1, x2 = self.nodes_x[i], self.nodes_x[i + 1]
            y1, y2 = self.nodes_y[j], self.nodes_y[j + 1]
            z1, z2 = self.nodes_z[k], self.nodes_z[k + 1]
            origin = np.array([x1, y1, z1])
            h = np.array([x2 - x1, y2 - y1, z2 - z1])
        return TensorCell(h, origin, indices, self.shape_cells)

    # --------------- Geometries ---------------------
    @property
    def cell_volumes(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        if getattr(self, "_cell_volumes", None) is None:
            vh = self.h
            # Compute cell volumes
            if self.dim == 1:
                self._cell_volumes = mkvc(vh[0])
            elif self.dim == 2:
                # Cell sizes in each direction
                self._cell_volumes = mkvc(np.outer(vh[0], vh[1]))
            elif self.dim == 3:
                # Cell sizes in each direction
                self._cell_volumes = mkvc(np.outer(mkvc(np.outer(vh[0], vh[1])), vh[2]))
        return self._cell_volumes

    @property
    def face_x_areas(self):
        """Return the areas of the x-faces.

        Calling this property will compute and return the areas of faces
        whose normal vector is along the x-axis.

        Returns
        -------
        (n_faces_x) numpy.ndarray
            The quantity returned depends on the dimensions of the mesh:

            - *1D:* Numpy array of ones whose length is equal to the number of nodes
            - *2D:* Areas of x-faces (equivalent to the lengths of y-edges)
            - *3D:* Areas of x-faces
        """
        if getattr(self, "_face_x_areas", None) is None:
            # Ensure that we are working with column vectors
            vh = self.h
            # The number of cell centers in each direction
            n = self.vnC
            # Compute areas of cell faces
            if self.dim == 1:
                areaFx = np.ones(n[0] + 1)
            elif self.dim == 2:
                areaFx = np.outer(np.ones(n[0] + 1), vh[1])
            elif self.dim == 3:
                areaFx = np.outer(np.ones(n[0] + 1), mkvc(np.outer(vh[1], vh[2])))
            self._face_x_areas = mkvc(areaFx)
        return self._face_x_areas

    @property
    def face_y_areas(self):
        """Return the areas of the y-faces.

        Calling this property will compute and return the areas of faces
        whose normal vector is along the y-axis. Note that only 2D and 3D
        tensor meshes have z-faces.

        Returns
        -------
        (n_faces_y) numpy.ndarray
            The quantity returned depends on the dimensions of the mesh:

            - *1D:* N/A since 1D meshes do not have y-faces
            - *2D:* Areas of y-faces (equivalent to the lengths of x-edges)
            - *3D:* Areas of y-faces
        """
        if getattr(self, "_face_y_areas", None) is None:
            # Ensure that we are working with column vectors
            vh = self.h
            # The number of cell centers in each direction
            n = self.vnC
            # Compute areas of cell faces
            if self.dim == 1:
                raise Exception("1D meshes do not have y-Faces")
            elif self.dim == 2:
                areaFy = np.outer(vh[0], np.ones(n[1] + 1))
            elif self.dim == 3:
                areaFy = np.outer(vh[0], mkvc(np.outer(np.ones(n[1] + 1), vh[2])))
            self._face_y_areas = mkvc(areaFy)
        return self._face_y_areas

    @property
    def face_z_areas(self):
        """Return the areas of the z-faces.

        Calling this property will compute and return the areas of faces
        whose normal vector is along the z-axis. Note that only 3D tensor
        meshes will have z-faces.

        Returns
        -------
        (n_faces_z) numpy.ndarray
            The quantity returned depends on the dimensions of the mesh:

            - *1D:* N/A since 1D meshes do not have z-faces
            - *2D:* N/A since 2D meshes do not have z-faces
            - *3D:* Areas of z-faces
        """
        if getattr(self, "_face_z_areas", None) is None:
            # Ensure that we are working with column vectors
            vh = self.h
            # The number of cell centers in each direction
            n = self.vnC
            # Compute areas of cell faces
            if self.dim == 1 or self.dim == 2:
                raise Exception("{}D meshes do not have z-Faces".format(self.dim))
            elif self.dim == 3:
                areaFz = np.outer(vh[0], mkvc(np.outer(vh[1], np.ones(n[2] + 1))))
            self._face_z_areas = mkvc(areaFz)
        return self._face_z_areas

    @property
    def face_areas(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        if self.dim == 1:
            return self.face_x_areas
        elif self.dim == 2:
            return np.r_[self.face_x_areas, self.face_y_areas]
        elif self.dim == 3:
            return np.r_[self.face_x_areas, self.face_y_areas, self.face_z_areas]

    @property
    def edge_x_lengths(self):
        """Return the x-edge lengths.

        Calling this property will compute and return the lengths of edges
        parallel to the x-axis.

        Returns
        -------
        (n_edges_x) numpy.ndarray
            X-edge lengths
        """
        if getattr(self, "_edge_x_lengths", None) is None:
            # Ensure that we are working with column vectors
            vh = self.h
            # The number of cell centers in each direction
            n = self.vnC
            # Compute edge lengths
            if self.dim == 1:
                edgeEx = vh[0]
            elif self.dim == 2:
                edgeEx = np.outer(vh[0], np.ones(n[1] + 1))
            elif self.dim == 3:
                edgeEx = np.outer(
                    vh[0], mkvc(np.outer(np.ones(n[1] + 1), np.ones(n[2] + 1)))
                )
            self._edge_x_lengths = mkvc(edgeEx)
        return self._edge_x_lengths

    @property
    def edge_y_lengths(self):
        """Return the y-edge lengths.

        Calling this property will compute and return the lengths of edges
        parallel to the y-axis.

        Returns
        -------
        (n_edges_y) numpy.ndarray
            The quantity returned depends on the dimensions of the mesh:

            - *1D:* N/A since 1D meshes do not have y-edges
            - *2D:* Returns y-edge lengths
            - *3D:* Returns y-edge lengths
        """
        if getattr(self, "_edge_y_lengths", None) is None:
            # Ensure that we are working with column vectors
            vh = self.h
            # The number of cell centers in each direction
            n = self.vnC
            # Compute edge lengths
            if self.dim == 1:
                raise Exception("1D meshes do not have y-edges")
            elif self.dim == 2:
                edgeEy = np.outer(np.ones(n[0] + 1), vh[1])
            elif self.dim == 3:
                edgeEy = np.outer(
                    np.ones(n[0] + 1), mkvc(np.outer(vh[1], np.ones(n[2] + 1)))
                )
            self._edge_y_lengths = mkvc(edgeEy)
        return self._edge_y_lengths

    @property
    def edge_z_lengths(self):
        """Return the z-edge lengths.

        Calling this property will compute and return the lengths of edges
        parallel to the z-axis.

        Returns
        -------
        (n_edges_z) numpy.ndarray
            The quantity returned depends on the dimensions of the mesh:

            - *1D:* N/A since 1D meshes do not have z-edges
            - *2D:* N/A since 2D meshes do not have z-edges
            - *3D:* Returns z-edge lengths
        """
        if getattr(self, "_edge_z_lengths", None) is None:
            # Ensure that we are working with column vectors
            vh = self.h
            # The number of cell centers in each direction
            n = self.vnC
            # Compute edge lengths
            if self.dim == 1 or self.dim == 2:
                raise Exception("{}D meshes do not have y-edges".format(self.dim))
            elif self.dim == 3:
                edgeEz = np.outer(
                    np.ones(n[0] + 1), mkvc(np.outer(np.ones(n[1] + 1), vh[2]))
                )
            self._edge_z_lengths = mkvc(edgeEz)
        return self._edge_z_lengths

    @property
    def edge_lengths(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        if self.dim == 1:
            return self.edge_x_lengths
        elif self.dim == 2:
            return np.r_[self.edge_x_lengths, self.edge_y_lengths]
        elif self.dim == 3:
            return np.r_[self.edge_x_lengths, self.edge_y_lengths, self.edge_z_lengths]
        return self._edge

    @property
    def face_boundary_indices(self):
        """Return the indices of the x, (y and z) boundary faces.

        For x, (y and z) faces, this property returns the indices of the faces
        on the boundaries. That is, the property returns the indices of the x-faces
        that lie on the x-boundary; likewise for y and z. Note that each
        Cartesian direction will have both a lower and upper boundary,
        and the property will return the indices corresponding to the lower
        and upper boundaries separately.

        E.g. for a 2D domain, there are 2 x-boundaries and 2 y-boundaries (4 in total).
        In this case, the return is a list of length 4 organized
        [ind_Bx1, ind_Bx2, ind_By1, ind_By2]::

                       By2
                + ------------- +
                |               |
                |               |
            Bx1 |               | Bx2
                |               |
                |               |
                + ------------- +
                       By1


        Returns
        -------
        (dim * 2) list of numpy.ndarray of bool
            The length of list returned depends on the dimension of the mesh.
            And the length of each array containing the indices depends on the
            number of faces in each direction. For 1D, 2D and 3D
            tensor meshes, the returns take the following form:

            - *1D:* returns [ind_Bx1, ind_Bx2]
            - *2D:* returns [ind_Bx1, ind_Bx2, ind_By1, ind_By2]
            - *3D:* returns [ind_Bx1, ind_Bx2, ind_By1, ind_By2, ind_Bz1, ind_Bz2]

        Examples
        --------
        Here, we construct a 4 by 3 cell 2D tensor mesh and return the indices
        of the x and y-boundary faces. In this case there are 3 x-faces on each
        x-boundary, and there are 4 y-faces on each y-boundary.

        >>> from discretize import TensorMesh
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        >>> hx = [1, 1, 1, 1]
        >>> hy = [2, 2, 2]
        >>> mesh = TensorMesh([hx, hy])
        >>> ind_Bx1, ind_Bx2, ind_By1, ind_By2 = mesh.face_boundary_indices

        >>> ax = plt.subplot(111)
        >>> mesh.plot_grid(ax=ax)
        >>> ax.scatter(*mesh.faces_x[ind_Bx1].T)
        >>> plt.show()
        """
        if self.dim == 1:
            indxd = self.gridFx == min(self.gridFx)
            indxu = self.gridFx == max(self.gridFx)
            return indxd, indxu
        elif self.dim == 2:
            indxd = self.gridFx[:, 0] == min(self.gridFx[:, 0])
            indxu = self.gridFx[:, 0] == max(self.gridFx[:, 0])
            indyd = self.gridFy[:, 1] == min(self.gridFy[:, 1])
            indyu = self.gridFy[:, 1] == max(self.gridFy[:, 1])
            return indxd, indxu, indyd, indyu
        elif self.dim == 3:
            indxd = self.gridFx[:, 0] == min(self.gridFx[:, 0])
            indxu = self.gridFx[:, 0] == max(self.gridFx[:, 0])
            indyd = self.gridFy[:, 1] == min(self.gridFy[:, 1])
            indyu = self.gridFy[:, 1] == max(self.gridFy[:, 1])
            indzd = self.gridFz[:, 2] == min(self.gridFz[:, 2])
            indzu = self.gridFz[:, 2] == max(self.gridFz[:, 2])
            return indxd, indxu, indyd, indyu, indzd, indzu

    @property
    def cell_nodes(self):
        """The index of all nodes for each cell.

        The nodes for each cell are listed following an "F" order: the first
        coordinate (``x``) changes faster than the second one (``y``). If the
        mesh is 3D, the second coordinate (``y``) changes faster than the third
        one (``z``).

        Returns
        -------
        numpy.ndarray of int
            Index array of shape (n_cells, 4) if 2D, or (n_cells, 8) if 3D

        Notes
        -----
        For a 2D mesh, the nodes indices for a single cell are returned in the
        following order:

        .. code::

            2 -- 3
            |    |
            0 -- 1

        For a 3D mesh, the nodes indices for a single cell are returned in the
        following order:

        .. code::

              6-----7
             /|    /|
            4-----5 |
            | |   | |
            | 2---|-3
            |/    |/
            0-----1

        """
        order = "F"
        nodes_indices = np.arange(self.n_nodes).reshape(self.shape_nodes, order=order)
        if self.dim == 1:
            cell_nodes = [
                nodes_indices[:-1].reshape(-1, order=order),
                nodes_indices[1:].reshape(-1, order=order),
            ]
        elif self.dim == 2:
            cell_nodes = [
                nodes_indices[:-1, :-1].reshape(-1, order=order),
                nodes_indices[1:, :-1].reshape(-1, order=order),
                nodes_indices[:-1, 1:].reshape(-1, order=order),
                nodes_indices[1:, 1:].reshape(-1, order=order),
            ]
        else:
            cell_nodes = [
                nodes_indices[:-1, :-1, :-1].reshape(-1, order=order),
                nodes_indices[1:, :-1, :-1].reshape(-1, order=order),
                nodes_indices[:-1, 1:, :-1].reshape(-1, order=order),
                nodes_indices[1:, 1:, :-1].reshape(-1, order=order),
                nodes_indices[:-1, :-1, 1:].reshape(-1, order=order),
                nodes_indices[1:, :-1, 1:].reshape(-1, order=order),
                nodes_indices[:-1, 1:, 1:].reshape(-1, order=order),
                nodes_indices[1:, 1:, 1:].reshape(-1, order=order),
            ]
        cell_nodes = np.stack(cell_nodes, axis=-1)
        return cell_nodes

    @property
    def cell_boundary_indices(self):
        """Return the indices of the x, (y and z) boundary cells.

        This property returns the indices of the cells on the x, (y and z)
        boundaries, respectively. Note that each axis direction will
        have both a lower and upper boundary. The property will
        return the indices corresponding to the lower and upper
        boundaries separately.

        E.g. for a 2D domain, there are 2 x-boundaries and 2 y-boundaries (4 in total).
        In this case, the return is a list of length 4 organized
        [ind_Bx1, ind_Bx2, ind_By1, ind_By2]::

                       By2
                + ------------- +
                |               |
                |               |
            Bx1 |               | Bx2
                |               |
                |               |
                + ------------- +
                       By1


        Returns
        -------
        (2 * dim) list of numpy.ndarray of bool
            The length of list returned depends on the dimension of the mesh (= 2 x dim).
            And the length of each array containing the indices is equal to
            the number of cells in the mesh. For 1D, 2D and 3D
            tensor meshes, the returns take the following form:

            - *1D:* returns [ind_Bx1, ind_Bx2]
            - *2D:* returns [ind_Bx1, ind_Bx2, ind_By1, ind_By2]
            - *3D:* returns [ind_Bx1, ind_Bx2, ind_By1, ind_By2, ind_Bz1, ind_Bz2]

        Examples
        --------
        Here, we construct a 4 by 3 cell 2D tensor mesh and return the indices
        of the x and y-boundary cells. In this case there are 3 cells touching
        each x-boundary, and there are 4 cells touching each y-boundary.

        >>> from discretize import TensorMesh
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        >>> hx = [1, 1, 1, 1]
        >>> hy = [2, 2, 2]
        >>> mesh = TensorMesh([hx, hy])
        >>> ind_Bx1, ind_Bx2, ind_By1, ind_By2 = mesh.cell_boundary_indices

        >>> ax = plt.subplot(111)
        >>> mesh.plot_grid(ax=ax)
        >>> ax.scatter(*mesh.cell_centers[ind_Bx1].T)
        >>> plt.show()
        """
        if self.dim == 1:
            indxd = self.gridCC == min(self.gridCC)
            indxu = self.gridCC == max(self.gridCC)
            return indxd, indxu
        elif self.dim == 2:
            indxd = self.gridCC[:, 0] == min(self.gridCC[:, 0])
            indxu = self.gridCC[:, 0] == max(self.gridCC[:, 0])
            indyd = self.gridCC[:, 1] == min(self.gridCC[:, 1])
            indyu = self.gridCC[:, 1] == max(self.gridCC[:, 1])
            return indxd, indxu, indyd, indyu
        elif self.dim == 3:
            indxd = self.gridCC[:, 0] == min(self.gridCC[:, 0])
            indxu = self.gridCC[:, 0] == max(self.gridCC[:, 0])
            indyd = self.gridCC[:, 1] == min(self.gridCC[:, 1])
            indyu = self.gridCC[:, 1] == max(self.gridCC[:, 1])
            indzd = self.gridCC[:, 2] == min(self.gridCC[:, 2])
            indzu = self.gridCC[:, 2] == max(self.gridCC[:, 2])
            return indxd, indxu, indyd, indyu, indzd, indzu

    def _repr_attributes(self):
        """Represent attributes of the mesh."""
        attrs = {}
        attrs["names"] = ["x", "y", "z"][: self.dim]

        # Loop over dimensions.
        for i in range(self.dim):
            name = attrs["names"][i]  # Name of this dimension
            attrs[name] = {}

            # Get min/max node.
            n_vector = getattr(self, "nodes_" + name)
            attrs[name]["min"] = np.nanmin(n_vector)
            attrs[name]["max"] = np.nanmax(n_vector)

            # Get min/max cell width.
            h_vector = self.h[i]
            attrs[name]["h_min"] = np.nanmin(h_vector)
            attrs[name]["h_max"] = np.nanmax(h_vector)

            # Get max stretching factor.
            if len(h_vector) < 2:
                attrs[name]["max_fact"] = 1.0
            else:
                attrs[name]["max_fact"] = np.nanmax(
                    np.r_[h_vector[:-1] / h_vector[1:], h_vector[1:] / h_vector[:-1]]
                )

            # Add number of cells.
            attrs[name]["nC"] = self.shape_cells[i]

        return attrs

    # DEPRECATIONS
    areaFx = deprecate_property(
        "face_x_areas", "areaFx", removal_version="1.0.0", error=True
    )
    areaFy = deprecate_property(
        "face_y_areas", "areaFy", removal_version="1.0.0", error=True
    )
    areaFz = deprecate_property(
        "face_z_areas", "areaFz", removal_version="1.0.0", error=True
    )
    edgeEx = deprecate_property(
        "edge_x_lengths", "edgeEx", removal_version="1.0.0", error=True
    )
    edgeEy = deprecate_property(
        "edge_y_lengths", "edgeEy", removal_version="1.0.0", error=True
    )
    edgeEz = deprecate_property(
        "edge_z_lengths", "edgeEz", removal_version="1.0.0", error=True
    )
    faceBoundaryInd = deprecate_property(
        "face_boundary_indices",
        "faceBoundaryInd",
        removal_version="1.0.0",
        error=True,
    )
    cellBoundaryInd = deprecate_property(
        "cell_boundary_indices",
        "cellBoundaryInd",
        removal_version="1.0.0",
        error=True,
    )


def _slice_to_index(index_slice, end):
    """Generate indices from a slice.

    Parameters
    ----------
    index_slice : slice
        Slice for cell indices along a single dimension
    end : int
        End of the slice. Will use this value as the stop in case the
        `index_slice.stop` is None.

    Returns
    -------
    Generator
    """
    if (start := index_slice.start) is None:
        start = 0
    if (stop := index_slice.stop) is None:
        stop = end
    if (step := index_slice.step) is None:
        step = 1
    if start < 0:
        start += end
    if stop < 0:
        stop += end
    if step < 0:
        return reversed(range(start, stop, abs(step)))
    return range(start, stop, step)
