"""
Cell class for TensorMesh
"""


def _check_inputs(h, origin):
    """
    Check inputs for the TensorCell
    """
    if len(h) != len(origin):
        raise ValueError(
            f"Invalid h and origin arguments with {len(h)} and {len(origin)} "
            "elements, respectively. They must have the same number of elements."
        )
    pass


class TensorCell:
    """
    Representation of a cell in a TensorMesh

    Parameters
    ----------
    h : (dim) array_like
        Iterable with the cell widths along each direction. For a 2D mesh, it
        must have two elements (``hx``, ``hy``). For a 3D mesh it must have
        three elements (``hx``, ``hy``, ``hz``).
    origin : (dim) array_like
        Iterable with the coordinates of the origin of the cell, i.e. the
        bottom-left-frontmost corner.
    """

    def __init__(self, h, origin):
        _check_inputs(h, origin)
        self._h = h
        self._origin = origin

    def __repr__(self):
        return f"TensorCell(h={self.h}, origin={self.origin})"

    def __eq__(self, other):
        if not isinstance(other, TensorCell):
            raise TypeError(
                f"Cannot compare an object of type '{other.__class__.__name__}' "
                "with a TensorCell"
            )
        return self.h == other.h and self.origin == other.origin

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
