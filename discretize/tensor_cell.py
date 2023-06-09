"""
Cell class for TensorMesh
"""
import itertools
import numpy as np


class TensorCell:
    """
    Representation of a cell in a TensorMesh

    Parameters
    ----------
    h : (dim) array_like
        Array with the cell widths along each direction. For a 2D mesh, it
        must have two elements (``hx``, ``hy``). For a 3D mesh it must have
        three elements (``hx``, ``hy``, ``hz``).
    origin : (dim) array_like
        Array with the coordinates of the origin of the cell, i.e. the
        bottom-left-frontmost corner.
    index_unraveled : (dim) array_like
        Array with the unraveled indices of the cell in its parent mesh.
    mesh_shape : (dim) array_like
        Shape of the parent mesh.
    """

    def __init__(self, h, origin, index_unraveled, mesh_shape):
        self._h = h
        self._origin = origin
        self._index_unraveled = index_unraveled
        self._mesh_shape = mesh_shape

    def __repr__(self):
        repr = ", ".join(
            [
                f"{attr}={getattr(self, attr)}"
                for attr in ("h", "origin", "index", "mesh_shape")
            ]
        )
        return f"TensorCell({repr})"

    def __eq__(self, other):
        if not isinstance(other, TensorCell):
            raise TypeError(
                f"Cannot compare an object of type '{other.__class__.__name__}' "
                "with a TensorCell"
            )
        are_equal = (
            self.h == other.h
            and self.origin == other.origin
            and self.index == other.index
            and self.mesh_shape == other.mesh_shape
        )
        return are_equal

    @property
    def h(self):
        """
        Cell widths
        """
        return self._h

    @property
    def origin(self):
        """
        Coordinates of the origin of the cell
        """
        return self._origin

    @property
    def index(self):
        """
        Index of the cell in a TensorMesh
        """
        return np.ravel_multi_index(
            self.index_unraveled, dims=self.mesh_shape, order="F"
        )

    @property
    def index_unraveled(self):
        """
        Unraveled index of the cell in a TensorMesh
        """
        return self._index_unraveled

    @property
    def mesh_shape(self):
        """
        Shape of the parent mesh.
        """
        return self._mesh_shape

    @property
    def dim(self):
        """
        Dimensions of the cell (1, 2 or 3)
        """
        return len(self.h)

    @property
    def center(self):
        """
        Coordinates of the cell center

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
        Bounds of the cell

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
            edges_indices = self.index_unraveled
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
        return edges_indices

    def get_neighbors(self, mesh):
        """
        Return the neighboring cells in the mesh

        Parameters
        ----------
        mesh : TensorMesh
            TensorMesh where the current cell lives.

        Returns
        -------
        list of TensorCell
        """
        return [mesh[index] for index in self.neighbors]
