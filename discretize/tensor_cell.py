"""
Cell class for TensorMesh
"""
from numbers import Number, Integral


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
    index : (dim) array_like
        Array with the indices of the cell in its parent mesh.
    mesh_shape : (dim) array_like
        Shape of the parent mesh.
    """

    def __init__(self, h, origin, index, mesh_shape):
        self._h = h
        self._origin = origin
        self._index = index
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
        return self._index

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
        center : (dim) tuple
            Tuple with the coordinates of the cell center.
        """
        center = tuple(origin_i + h_i / 2 for origin_i, h_i in zip(self.origin, self.h))
        return center

    @property
    def bounds(self):
        """
        Bounds of the cell

        Coordinates that define the bounds of the cell. Bounds are returned in
        the following order: ``x1``, ``x2``, ``y1``, ``y2``, ``z1``, ``z2``.

        Returns
        -------
        bounds : (2 * dim) tuple
            Tuple with the cell bounds.
        """
        bounds = tuple(
            origin_i + factor * h_i
            for origin_i, h_i in zip(self.origin, self.h)
            for factor in (0, 1)
        )
        return bounds

    @property
    def neighbors(self):
        """
        Indices for this cell's neighbors within its parent mesh.

        Returns
        -------
        list of int
        """
        neighbor_indices = []
        for dim in range(self.dim):
            for delta in (-1, 1):
                index = list(self.index)
                index[dim] += delta
                if 0 <= index[dim] < self._mesh_shape[dim]:
                    neighbor_indices.append(index)
        return neighbor_indices

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
        return [mesh[*i] for i in self.neighbors]
