"""Cell class for TensorMesh."""

import itertools
import numpy as np


class TensorCell:
    """
    Representation of a cell in a TensorMesh.

    Parameters
    ----------
    h : (dim) numpy.ndarray
        Array with the cell widths along each direction. For a 2D mesh, it
        must have two elements (``hx``, ``hy``). For a 3D mesh it must have
        three elements (``hx``, ``hy``, ``hz``).
    origin : (dim) numpy.ndarray
        Array with the coordinates of the origin of the cell, i.e. the
        bottom-left-frontmost corner.
    index_unraveled : (dim) tuple
        Array with the unraveled indices of the cell in its parent mesh.
    mesh_shape : (dim) tuple
        Shape of the parent mesh.

    Examples
    --------
    Define a simple :class:`discretize.TensorMesh`.

    >>> from discretize import TensorMesh
    >>> mesh = TensorMesh([5, 8, 10])

    We can obtain a particular cell in the mesh by its index:

    >>> cell = mesh[3]
    >>> cell
    TensorCell(h=[0.2   0.125 0.1  ], origin=[0.6 0.  0. ], index=3, mesh_shape=(5, 8, 10))

    And then obtain information about it, like its
    :attr:`discretize.tensor_cell.TensorCell.origin`:

    >>> cell.origin
    array([0.6, 0. , 0. ])

    Or its
    :attr:`discretize.tensor_cell.TensorCell.bounds`:

    >>> cell.bounds
    array([0.6  , 0.8  , 0.   , 0.125, 0.   , 0.1  ])

    We can also get its neighboring cells:

    >>> neighbours = cell.get_neighbors(mesh)
    >>> for neighbor in neighbours:
    ...     print(neighbor.center)
    [0.5    0.0625 0.05  ]
    [0.9    0.0625 0.05  ]
    [0.7    0.1875 0.05  ]
    [0.7    0.0625 0.15  ]


    Alternatively, we can iterate over all cells in the mesh with a simple
    *for loop* or list comprehension:

    >>> cells = [cell for cell in mesh]
    >>> len(cells)
    400

    """

    def __init__(self, h, origin, index_unraveled, mesh_shape):
        self._h = h
        self._origin = origin
        self._index_unraveled = index_unraveled
        self._mesh_shape = mesh_shape

    def __repr__(self):
        """Represent a TensorCell."""
        attributes = ", ".join(
            [
                f"{attr}={getattr(self, attr)}"
                for attr in ("h", "origin", "index", "mesh_shape")
            ]
        )
        return f"TensorCell({attributes})"

    def __eq__(self, other):
        """Check if this cell is the same as other one."""
        if not isinstance(other, TensorCell):
            raise TypeError(
                f"Cannot compare an object of type '{other.__class__.__name__}' "
                "with a TensorCell"
            )
        are_equal = (
            np.all(self.h == other.h)
            and np.all(self.origin == other.origin)
            and self.index == other.index
            and self.mesh_shape == other.mesh_shape
        )
        return are_equal

    @property
    def h(self):
        """Cell widths."""
        return self._h

    @property
    def origin(self):
        """Coordinates of the origin of the cell."""
        return self._origin

    @property
    def index(self):
        """Index of the cell in a TensorMesh."""
        return np.ravel_multi_index(
            self.index_unraveled, dims=self.mesh_shape, order="F"
        )

    @property
    def index_unraveled(self):
        """Unraveled index of the cell in a TensorMesh."""
        return self._index_unraveled

    @property
    def mesh_shape(self):
        """Shape of the parent mesh."""
        return self._mesh_shape

    @property
    def dim(self):
        """Dimensions of the cell (1, 2 or 3)."""
        return len(self.h)

    @property
    def center(self):
        """
        Coordinates of the cell center.

        Returns
        -------
        center : (dim) array
            Array with the coordinates of the cell center.
        """
        center = np.array(self.origin) + np.array(self.h) / 2
        return center

    @property
    def bounds(self):
        """
        Bounds of the cell.

        Coordinates that define the bounds of the cell. Bounds are returned in
        the following order: ``x1``, ``x2``, ``y1``, ``y2``, ``z1``, ``z2``.

        Returns
        -------
        bounds : (2 * dim) array
            Array with the cell bounds.
        """
        bounds = np.array(
            [
                origin_i + factor * h_i
                for origin_i, h_i in zip(self.origin, self.h)
                for factor in (0, 1)
            ]
        )
        return bounds

    @property
    def neighbors(self):
        """
        Indices for this cell's neighbors within its parent mesh.

        Returns
        -------
        list of list of int
        """
        neighbor_indices = []
        for dim in range(self.dim):
            for delta in (-1, 1):
                index = list(self.index_unraveled)
                index[dim] += delta
                if 0 <= index[dim] < self._mesh_shape[dim]:
                    neighbor_indices.append(
                        np.ravel_multi_index(index, dims=self.mesh_shape, order="F")
                    )
        return neighbor_indices

    @property
    def nodes(self):
        """
        Indices for this cell's nodes within its parent mesh.

        Returns
        -------
        list of int
        """
        # Define shape of nodes in parent mesh
        nodes_shape = [s + 1 for s in self.mesh_shape]
        # Get indices of nodes per dimension
        nodes_index_per_dim = [[index, index + 1] for index in self.index_unraveled]
        # Combine the nodes_index_per_dim using itertools.product.
        # Because we want to follow a FORTRAN order, we need to reverse the
        # order of the nodes_index_per_dim and the indices.
        nodes_indices = [i[::-1] for i in itertools.product(*nodes_index_per_dim[::-1])]
        # Ravel indices
        nodes_indices = [
            np.ravel_multi_index(index, dims=nodes_shape, order="F")
            for index in nodes_indices
        ]
        return nodes_indices

    @property
    def edges(self):
        """
        Indices for this cell's edges within its parent mesh.

        Returns
        -------
        list of int
        """
        if self.dim == 1:
            edges_indices = [self.index]
        elif self.dim == 2:
            # Get shape of edges grids (for edges_x and edges_y)
            edges_x_shape = [self.mesh_shape[0], self.mesh_shape[1] + 1]
            edges_y_shape = [self.mesh_shape[0] + 1, self.mesh_shape[1]]
            # Calculate total amount of edges_x
            n_edges_x = edges_x_shape[0] * edges_x_shape[1]
            # Get indices of edges_x
            edges_x_indices = [
                [self.index_unraveled[0], self.index_unraveled[1] + delta]
                for delta in (0, 1)
            ]
            edges_x_indices = [
                np.ravel_multi_index(index, dims=edges_x_shape, order="F")
                for index in edges_x_indices
            ]
            # Get indices of edges_y
            edges_y_indices = [
                [self.index_unraveled[0] + delta, self.index_unraveled[1]]
                for delta in (0, 1)
            ]
            edges_y_indices = [
                n_edges_x + np.ravel_multi_index(index, dims=edges_y_shape, order="F")
                for index in edges_y_indices
            ]
            edges_indices = edges_x_indices + edges_y_indices
        elif self.dim == 3:
            edges_x_shape = [
                n if i == 0 else n + 1 for i, n in enumerate(self.mesh_shape)
            ]
            edges_y_shape = [
                n if i == 1 else n + 1 for i, n in enumerate(self.mesh_shape)
            ]
            edges_z_shape = [
                n if i == 2 else n + 1 for i, n in enumerate(self.mesh_shape)
            ]
            # Calculate total amount of edges_x and edges_y
            n_edges_x = edges_x_shape[0] * edges_x_shape[1] * edges_x_shape[2]
            n_edges_y = edges_y_shape[0] * edges_y_shape[1] * edges_y_shape[2]
            # Get indices of edges_x
            edges_x_indices = [
                [
                    self.index_unraveled[0],
                    self.index_unraveled[1] + delta_y,
                    self.index_unraveled[2] + delta_z,
                ]
                for delta_z in (0, 1)
                for delta_y in (0, 1)
            ]
            edges_x_indices = [
                np.ravel_multi_index(index, dims=edges_x_shape, order="F")
                for index in edges_x_indices
            ]
            # Get indices of edges_y
            edges_y_indices = [
                [
                    self.index_unraveled[0] + delta_x,
                    self.index_unraveled[1],
                    self.index_unraveled[2] + delta_z,
                ]
                for delta_z in (0, 1)
                for delta_x in (0, 1)
            ]
            edges_y_indices = [
                n_edges_x + np.ravel_multi_index(index, dims=edges_y_shape, order="F")
                for index in edges_y_indices
            ]
            # Get indices of edges_z
            edges_z_indices = [
                [
                    self.index_unraveled[0] + delta_x,
                    self.index_unraveled[1] + delta_y,
                    self.index_unraveled[2],
                ]
                for delta_y in (0, 1)
                for delta_x in (0, 1)
            ]
            edges_z_indices = [
                n_edges_x
                + n_edges_y
                + np.ravel_multi_index(index, dims=edges_z_shape, order="F")
                for index in edges_z_indices
            ]
            edges_indices = edges_x_indices + edges_y_indices + edges_z_indices
        return edges_indices

    @property
    def faces(self):
        """
        Indices for cell's faces within its parent mesh.

        Returns
        -------
        list of int
        """
        if self.dim == 1:
            faces_indices = [self.index, self.index + 1]
        elif self.dim == 2:
            # Get shape of faces grids
            # (faces_x are normal to x and faces_y are normal to y)
            faces_x_shape = [self.mesh_shape[0] + 1, self.mesh_shape[1]]
            faces_y_shape = [self.mesh_shape[0], self.mesh_shape[1] + 1]
            # Calculate total amount of faces_x
            n_faces_x = faces_x_shape[0] * faces_x_shape[1]
            # Get indices of faces_x
            faces_x_indices = [
                [self.index_unraveled[0] + delta, self.index_unraveled[1]]
                for delta in (0, 1)
            ]
            faces_x_indices = [
                np.ravel_multi_index(index, dims=faces_x_shape, order="F")
                for index in faces_x_indices
            ]
            # Get indices of faces_y
            faces_y_indices = [
                [self.index_unraveled[0], self.index_unraveled[1] + delta]
                for delta in (0, 1)
            ]
            faces_y_indices = [
                n_faces_x + np.ravel_multi_index(index, dims=faces_y_shape, order="F")
                for index in faces_y_indices
            ]
            faces_indices = faces_x_indices + faces_y_indices
        elif self.dim == 3:
            # Get shape of faces grids
            faces_x_shape = [
                n + 1 if i == 0 else n for i, n in enumerate(self.mesh_shape)
            ]
            faces_y_shape = [
                n + 1 if i == 1 else n for i, n in enumerate(self.mesh_shape)
            ]
            faces_z_shape = [
                n + 1 if i == 2 else n for i, n in enumerate(self.mesh_shape)
            ]
            # Calculate total amount of faces_x and faces_y
            n_faces_x = faces_x_shape[0] * faces_x_shape[1] * faces_x_shape[2]
            n_faces_y = faces_y_shape[0] * faces_y_shape[1] * faces_y_shape[2]
            # Get indices of faces_x
            faces_x_indices = [
                [
                    self.index_unraveled[0] + delta,
                    self.index_unraveled[1],
                    self.index_unraveled[2],
                ]
                for delta in (0, 1)
            ]
            faces_x_indices = [
                np.ravel_multi_index(index, dims=faces_x_shape, order="F")
                for index in faces_x_indices
            ]
            # Get indices of faces_y
            faces_y_indices = [
                [
                    self.index_unraveled[0],
                    self.index_unraveled[1] + delta,
                    self.index_unraveled[2],
                ]
                for delta in (0, 1)
            ]
            faces_y_indices = [
                n_faces_x + np.ravel_multi_index(index, dims=faces_y_shape, order="F")
                for index in faces_y_indices
            ]
            # Get indices of faces_z
            faces_z_indices = [
                [
                    self.index_unraveled[0],
                    self.index_unraveled[1],
                    self.index_unraveled[2] + delta,
                ]
                for delta in (0, 1)
            ]
            faces_z_indices = [
                n_faces_x
                + n_faces_y
                + np.ravel_multi_index(index, dims=faces_z_shape, order="F")
                for index in faces_z_indices
            ]
            faces_indices = faces_x_indices + faces_y_indices + faces_z_indices
        return faces_indices

    def get_neighbors(self, mesh):
        """
        Return the neighboring cells in the mesh.

        Parameters
        ----------
        mesh : TensorMesh
            TensorMesh where the current cell lives.

        Returns
        -------
        list of TensorCell
        """
        return [mesh[index] for index in self.neighbors]
