"""Base class for tensor-product style meshes."""

import numpy as np
import scipy.sparse as sp

from discretize.base.base_regular_mesh import BaseRegularMesh
from discretize.utils import (
    is_scalar,
    as_array_n_by_dim,
    unpack_widths,
    mkvc,
    ndgrid,
    spzeros,
    sdiag,
    sdinv,
    TensorType,
    interpolation_matrix,
    make_boundary_bool,
)
from discretize.utils.code_utils import deprecate_method, deprecate_property
import warnings


class BaseTensorMesh(BaseRegularMesh):
    """Base class for tensor-product style meshes.

    This class contains properites and methods that are common to Cartesian
    and cylindrical meshes. That is, meshes whose cell centers, nodes, faces
    and edges can be constructed with tensor-products of vectors.

    Do not use this class directly! Practical tensor meshes supported in
    discretize will inherit this class; i.e. :class:`discretize.TensorMesh`
    and :class:`~discretize.CylindricalMesh`. Inherit this class if you plan
    to develop a new tensor-style mesh class (e.g. a spherical mesh).

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
        corner. By default, the mesh is anchored such that its origin is at
        ``[0, 0, 0]``.

        For each dimension (x, y or z), The user may set the origin 2 ways:

        - a ``scalar`` which explicitly defines origin along that dimension.
        - **{'0', 'C', 'N'}** a :class:`str` specifying whether the zero coordinate along
          each axis is the first node location ('0'), in the center ('C') or the last
          node location ('N').

    See Also
    --------
    utils.unpack_widths :
        The function used to expand a ``list`` or ``tuple`` to generate widths.
    """

    _meshType = "BASETENSOR"
    _aliases = {
        **BaseRegularMesh._aliases,
        **{
            "gridFx": "faces_x",
            "gridFy": "faces_y",
            "gridFz": "faces_z",
            "gridEx": "edges_x",
            "gridEy": "edges_y",
            "gridEz": "edges_z",
        },
    }

    _unitDimensions = [1, 1, 1]
    _items = {"h"} | BaseRegularMesh._items

    def __init__(self, h, origin=None, **kwargs):
        if "x0" in kwargs:
            origin = kwargs.pop("x0")

        try:
            h = list(h)  # ensure value is a list (and make a copy)
        except TypeError:
            raise TypeError("h must be an iterable object, not {}".format(type(h)))
        if len(h) == 0 or len(h) > 3:
            raise ValueError("h must be of dimension 1, 2, or 3 not {}".format(len(h)))
        # expand value
        for i, h_i in enumerate(h):
            if is_scalar(h_i) and not isinstance(h_i, np.ndarray):
                # This gives you something over the unit cube.
                h_i = self._unitDimensions[i] * np.ones(int(h_i)) / int(h_i)
            elif isinstance(h_i, (list, tuple)):
                h_i = unpack_widths(h_i)
            if not isinstance(h_i, np.ndarray):
                raise TypeError("h[{0:d}] is not a numpy array.".format(i))
            if len(h_i.shape) != 1:
                raise ValueError("h[{0:d}] must be a 1D numpy array.".format(i))
            h[i] = h_i[:]  # make a copy.
        self._h = tuple(h)

        shape_cells = tuple([len(h_i) for h_i in h])
        kwargs.pop("shape_cells", None)
        super().__init__(shape_cells=shape_cells, **kwargs)  # do not pass origin here
        if origin is not None:
            self.origin = origin

    @property
    def h(self):
        r"""Cell widths along each axis direction.

        The widths of the cells along each axis direction are returned
        as a tuple of 1D arrays; e.g. (hx, hy, hz) for a 3D mesh.
        The lengths of the 1D arrays in the tuple are given by
        :py:attr:`~discretize.base.BaseRegularMesh.shape_cells`. Ordering
        begins at the bottom southwest corner. These are the
        cell widths used when creating the mesh.

        Returns
        -------
        (dim) tuple of numpy.ndarray
            Cell widths along each axis direction. This depends on the mesh class:

            - :class:`~discretize.TensorMesh`: cell widths along the *x* , [*y* and *z* ] directions
            - :class:`~discretize.CylindricalMesh`: cell widths along the *r*, :math:`\phi` and *z* directions
            - :class:`~discretize.TreeMesh`: cells widths of the *underlying tensor mesh* along the *x* , *y* [and *z* ] directions
        """
        return self._h

    @BaseRegularMesh.origin.setter
    def origin(self, value):  # NOQA D102
        # ensure value is a 1D array at all times
        try:
            value = list(value)
        except TypeError:
            raise TypeError("origin must be iterable")
        if len(value) != self.dim:
            raise ValueError("Dimension mismatch. len(origin) != len(h)")
        for i, (val, h_i) in enumerate(zip(value, self.h)):
            if val == "C":
                value[i] = -h_i.sum() * 0.5
            elif val == "N":
                value[i] = -h_i.sum()
        value = np.asarray(value, dtype=np.float64)
        self._origin = value

    @property
    def nodes_x(self):
        """Return x-coordinates of the nodes along the x-direction.

        This property returns a vector containing the x-coordinate values of
        the nodes along the x-direction. For instances of
        :class:`~discretize.TensorMesh` or :class:`~discretize.CylindricalMesh`,
        this is equivalent to the node positions which define the tensor along
        the x-axis. For instances of :class:`~discretize.TreeMesh` however, this
        property returns the x-coordinate values of the nodes along the x-direction
        for the underlying tensor mesh.

        Returns
        -------
        (n_nodes_x) numpy.ndarray of float
            A 1D array containing the x-coordinates of the nodes along
            the x-direction.

        """
        return np.r_[self.origin[0], self.h[0]].cumsum()

    @property
    def nodes_y(self):
        """Return y-coordinates of the nodes along the y-direction.

        For 2D and 3D meshes, this property returns a vector
        containing the y-coordinate values of the nodes along the
        y-direction. For instances of :class:`~discretize.TensorMesh` or
        :class:`~discretize.CylindricalMesh`, this is equivalent to
        the node positions which define the tensor along the y-axis.
        For instances of :class:`~discretize.TreeMesh` however, this property
        returns the y-coordinate values of the nodes along the y-direction
        for the underlying tensor mesh.

        Returns
        -------
        (n_nodes_y) numpy.ndarray of float or None
            A 1D array containing the y-coordinates of the nodes along
            the y-direction. Returns *None* for 1D meshes.

        """
        return None if self.dim < 2 else np.r_[self.origin[1], self.h[1]].cumsum()

    @property
    def nodes_z(self):
        """Return z-coordinates of the nodes along the z-direction.

        For 3D meshes, this property returns a 1D vector
        containing the z-coordinate values of the nodes along the
        z-direction. For instances of :class:`~discretize.TensorMesh` or
        :class:`~discretize.CylindricalMesh`, this is equivalent to
        the node positions which define the tensor along the z-axis.
        For instances of :class:`~discretize.TreeMesh` however, this property
        returns the z-coordinate values of the nodes along the z-direction
        for the underlying tensor mesh.

        Returns
        -------
        (n_nodes_z) numpy.ndarray of float or None
            A 1D array containing the z-coordinates of the nodes along
            the z-direction. Returns *None* for 1D and 2D meshes.

        """
        return None if self.dim < 3 else np.r_[self.origin[2], self.h[2]].cumsum()

    @property
    def cell_centers_x(self):
        """Return x-coordinates of the cell centers along the x-direction.

        For 1D, 2D and 3D meshes, this property returns a 1D vector
        containing the x-coordinate values of the cell centers along the
        x-direction. For instances of :class:`~discretize.TensorMesh` or
        :class:`~discretize.CylindricalMesh`, this is equivalent to
        the cell center positions which define the tensor along the x-axis.
        For instances of :class:`~discretize.TreeMesh` however, this property
        returns the x-coordinate values of the cell centers along the x-direction
        for the underlying tensor mesh.

        Returns
        -------
        (n_cells_x) numpy.ndarray of float
            A 1D array containing the x-coordinates of the cell centers along
            the x-direction.
        """
        nodes = self.nodes_x
        return (nodes[1:] + nodes[:-1]) / 2

    @property
    def cell_centers_y(self):
        """Return y-coordinates of the cell centers along the y-direction.

        For 2D and 3D meshes, this property returns a 1D vector
        containing the y-coordinate values of the cell centers along the
        y-direction. For instances of :class:`~discretize.TensorMesh` or
        :class:`~discretize.CylindricalMesh`, this is equivalent to
        the cell center positions which define the tensor along the y-axis.
        For instances of :class:`~discretize.TreeMesh` however, this property
        returns the y-coordinate values of the cell centers along the y-direction
        for the underlying tensor mesh .

        Returns
        -------
        (n_cells_y) numpy.ndarray of float or None
            A 1D array containing the y-coordinates of the cell centers along
            the y-direction. Returns *None* for 1D meshes.

        """
        if self.dim < 2:
            return None
        nodes = self.nodes_y
        return (nodes[1:] + nodes[:-1]) / 2

    @property
    def cell_centers_z(self):
        """Return z-coordinates of the cell centers along the z-direction.

        For 3D meshes, this property returns a 1D vector
        containing the z-coordinate values of the cell centers along the
        z-direction. For instances of :class:`~discretize.TensorMesh` or
        :class:`~discretize.CylindricalMesh`, this is equivalent to
        the cell center positions which define the tensor along the z-axis.
        For instances of :class:`~discretize.TreeMesh` however, this property
        returns the z-coordinate values of the cell centers along the z-direction
        for the underlying tensor mesh .

        Returns
        -------
        (n_cells_z) numpy.ndarray of float or None
            A 1D array containing the z-coordinates of the cell centers along
            the z-direction. Returns *None* for 1D and 2D meshes.

        """
        if self.dim < 3:
            return None
        nodes = self.nodes_z
        return (nodes[1:] + nodes[:-1]) / 2

    @property
    def cell_centers(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        return self._getTensorGrid("cell_centers")

    @property
    def nodes(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        return self._getTensorGrid("nodes")

    @property
    def boundary_nodes(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        dim = self.dim
        if dim == 1:
            return self.nodes_x[[0, -1]]
        return self.nodes[make_boundary_bool(self.shape_nodes)]

    @property
    def h_gridded(self):
        """Return dimensions of all mesh cells as staggered grid.

        This property returns a numpy array of shape (n_cells, dim)
        containing gridded x, (y and z) dimensions for all cells in the mesh.
        The first row corresponds to the bottom-front-leftmost cell.
        The cells are ordered along the x, then y, then z directions.

        Returns
        -------
        (n_cells, dim) numpy.ndarray of float
            Dimensions of all mesh cells as staggered grid

        Examples
        --------
        The following is a 1D example.

        >>> from discretize import TensorMesh
        >>> hx = np.ones(5)
        >>> mesh_1D = TensorMesh([hx])
        >>> mesh_1D.h_gridded
        array([[1.],
               [1.],
               [1.],
               [1.],
               [1.]])

        The following is a 3D example.

        >>> hx, hy, hz = np.ones(2), 2*np.ones(2), 3*np.ones(2)
        >>> mesh_3D = TensorMesh([hx, hy, hz])
        >>> mesh_3D.h_gridded
        array([[1., 2., 3.],
               [1., 2., 3.],
               [1., 2., 3.],
               [1., 2., 3.],
               [1., 2., 3.],
               [1., 2., 3.],
               [1., 2., 3.],
               [1., 2., 3.]])

        """
        if self.dim == 1:
            return self.h[0][:, None]
        return ndgrid(*self.h)

    @property
    def faces_x(self):
        """Gridded x-face locations.

        This property returns a numpy array of shape (n_faces_x, dim)
        containing gridded locations for all x-faces in the
        mesh. The first row corresponds to the bottom-front-leftmost x-face.
        The x-faces are ordered along the x, then y, then z directions.

        Returns
        -------
        (n_faces_x, dim) numpy.ndarray of float
            Gridded x-face locations
        """
        if self.nFx == 0:
            return
        return self._getTensorGrid("faces_x")

    @property
    def faces_y(self):
        """Gridded y-face locations.

        This property returns a numpy array of shape (n_faces_y, dim)
        containing gridded locations for all y-faces in the
        mesh. The first row corresponds to the bottom-front-leftmost y-face.
        The y-faces are ordered along the x, then y, then z directions.

        Returns
        -------
        n_faces_y, dim) numpy.ndarray of float or None
            Gridded y-face locations for 2D and 3D mesh. Returns *None* for 1D meshes.
        """
        if self.nFy == 0 or self.dim < 2:
            return
        return self._getTensorGrid("faces_y")

    @property
    def faces_z(self):
        """Gridded z-face locations.

        This property returns a numpy array of shape (n_faces_z, dim)
        containing gridded locations for all z-faces in the
        mesh. The first row corresponds to the bottom-front-leftmost z-face.
        The z-faces are ordered along the x, then y, then z directions.

        Returns
        -------
        (n_faces_z, dim) numpy.ndarray of float or None
            Gridded z-face locations for 3D mesh. Returns *None* for 1D and 2D meshes.
        """
        if self.nFz == 0 or self.dim < 3:
            return
        return self._getTensorGrid("faces_z")

    @property
    def faces(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        if self.faces_x is not None:
            faces = self.faces_x
        else:
            faces = np.empty((0, self.dim))
        if self.dim > 1 and self.faces_y is not None:
            faces = np.r_[faces, self.faces_y]
        if self.dim > 2 and self.faces_z is not None:
            faces = np.r_[faces, self.faces_z]
        return faces

    @property
    def boundary_faces(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        dim = self.dim
        if dim == 1:
            return self.nodes_x[[0, -1]]
        if dim == 2:
            fx = ndgrid(self.nodes_x[[0, -1]], self.cell_centers_y)
            fy = ndgrid(self.cell_centers_x, self.nodes_y[[0, -1]])
            return np.r_[fx, fy]
        if dim == 3:
            fx = ndgrid(self.nodes_x[[0, -1]], self.cell_centers_y, self.cell_centers_z)
            fy = ndgrid(self.cell_centers_x, self.nodes_y[[0, -1]], self.cell_centers_z)
            fz = ndgrid(self.cell_centers_x, self.cell_centers_y, self.nodes_z[[0, -1]])
            return np.r_[fx, fy, fz]

    @property
    def boundary_face_outward_normals(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        dim = self.dim
        if dim == 1:
            return np.array([-1, 1])
        if dim == 2:
            nx = ndgrid(np.r_[-1, 1], np.zeros(self.shape_cells[1]))
            ny = ndgrid(np.zeros(self.shape_cells[0]), np.r_[-1, 1])
            return np.r_[nx, ny]
        if dim == 3:
            nx = ndgrid(
                np.r_[-1, 1],
                np.zeros(self.shape_cells[1]),
                np.zeros(self.shape_cells[2]),
            )
            ny = ndgrid(
                np.zeros(self.shape_cells[0]),
                np.r_[-1, 1],
                np.zeros(self.shape_cells[2]),
            )
            nz = ndgrid(
                np.zeros(self.shape_cells[0]),
                np.zeros(self.shape_cells[1]),
                np.r_[-1, 1],
            )
            return np.r_[nx, ny, nz]

    @property
    def edges_x(self):
        """Gridded x-edge locations.

        This property returns a numpy array of shape (n_edges_x, dim)
        containing gridded locations for all x-edges in the mesh.
        The first row corresponds to the bottom-front-leftmost x-edge.
        The x-edges are ordered along the x, then y, then z directions.

        Returns
        -------
        (n_edges_x, dim) numpy.ndarray of float or None
            Gridded x-edge locations. Returns *None* if `shape_edges_x[0]` is 0.
        """
        if self.nEx == 0:
            return
        return self._getTensorGrid("edges_x")

    @property
    def edges_y(self):
        """Gridded y-edge locations.

        This property returns a numpy array of shape (n_edges_y, dim)
        containing gridded locations for all y-edges in the mesh.
        The first row corresponds to the bottom-front-leftmost y-edge.
        The y-edges are ordered along the x, then y, then z directions.

        Returns
        -------
        (n_edges_y, dim) numpy.ndarray of float
            Gridded y-edge locations. Returns *None* for 1D meshes.
        """
        if self.nEy == 0 or self.dim < 2:
            return
        return self._getTensorGrid("edges_y")

    @property
    def edges_z(self):
        """Gridded z-edge locations.

        This property returns a numpy array of shape (n_edges_z, dim)
        containing gridded locations for all z-edges in the mesh.
        The first row corresponds to the bottom-front-leftmost z-edge.
        The z-edges are ordered along the x, then y, then z directions.

        Returns
        -------
        (n_edges_z, dim) numpy.ndarray of float
            Gridded z-edge locations. Returns *None* for 1D and 2D meshes.
        """
        if self.nEz == 0 or self.dim < 3:
            return
        return self._getTensorGrid("edges_z")

    @property
    def edges(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        if self.edges_x is not None:
            edges = self.edges_x
        else:
            edges = np.empty((0, self.dim))
        if self.dim > 1 and self.edges_y is not None:
            edges = np.r_[edges, self.edges_y]
        if self.dim > 2 and self.edges_z is not None:
            edges = np.r_[edges, self.edges_z]
        return edges

    @property
    def boundary_edges(self):
        """Boundary edge locations.

        This property returns the locations of the edges on
        the boundary of the mesh as a numpy array. The shape
        of the numpy array is the number of boundary edges by
        the dimension of the mesh.

        Returns
        -------
        (n_boundary_edges, dim) numpy.ndarray of float
            Boundary edge locations
        """
        dim = self.dim
        if dim == 1:
            return None  # no boundary edges in 1D
        if dim == 2:
            ex = ndgrid(self.cell_centers_x, self.nodes_y[[0, -1]])
            ey = ndgrid(self.nodes_x[[0, -1]], self.cell_centers_y)
            return np.r_[ex, ey]
        if dim == 3:
            ex = self.edges_x[make_boundary_bool(self.shape_edges_x, dir="yz")]
            ey = self.edges_y[make_boundary_bool(self.shape_edges_y, dir="xz")]
            ez = self.edges_z[make_boundary_bool(self.shape_edges_z, dir="xy")]
            return np.r_[ex, ey, ez]

    def _getTensorGrid(self, key):
        if getattr(self, "_" + key, None) is None:
            setattr(self, "_" + key, ndgrid(self.get_tensor(key)))
        return getattr(self, "_" + key)

    def get_tensor(self, key):
        """Return the base 1D arrays for a specified mesh tensor.

        The cell-centers, nodes, x-faces, z-edges, etc... of a tensor mesh
        can be constructed by applying tensor products to the set of base
        1D arrays; i.e. (vx, vy, vz). These 1D arrays define the gridded
        locations for the mesh tensor along each axis. For a given mesh tensor
        (i.e. cell centers, nodes, x/y/z faces or x/y/z edges),
        **get_tensor** returns a list containing the base 1D arrays.

        Parameters
        ----------
        key : str
            Specifies the tensor being returned. Please choose from::

                'CC', 'cell_centers' -> location of cell centers
                'N', 'nodes'         -> location of nodes
                'Fx', 'faces_x'      -> location of faces with an x normal
                'Fy', 'faces_y'      -> location of faces with an y normal
                'Fz', 'faces_z'      -> location of faces with an z normal
                'Ex', 'edges_x'      -> location of edges with an x tangent
                'Ey', 'edges_y'      -> location of edges with an y tangent
                'Ez', 'edges_z'      -> location of edges with an z tangent

        Returns
        -------
        (dim) list of 1D numpy.ndarray
            list of base 1D arrays for the tensor.

        """
        key = self._parse_location_type(key)

        if key == "faces_x":
            ten = [
                self.nodes_x,
                self.cell_centers_y,
                self.cell_centers_z,
            ]
        elif key == "faces_y":
            ten = [
                self.cell_centers_x,
                self.nodes_y,
                self.cell_centers_z,
            ]
        elif key == "faces_z":
            ten = [
                self.cell_centers_x,
                self.cell_centers_y,
                self.nodes_z,
            ]
        elif key == "edges_x":
            ten = [self.cell_centers_x, self.nodes_y, self.nodes_z]
        elif key == "edges_y":
            ten = [self.nodes_x, self.cell_centers_y, self.nodes_z]
        elif key == "edges_z":
            ten = [self.nodes_x, self.nodes_y, self.cell_centers_z]
        elif key == "cell_centers":
            ten = [
                self.cell_centers_x,
                self.cell_centers_y,
                self.cell_centers_z,
            ]
        elif key == "nodes":
            ten = [self.nodes_x, self.nodes_y, self.nodes_z]
        else:
            raise KeyError(r"Unrecognized key {key}")

        return [t for t in ten if t is not None]

    # --------------- Methods ---------------------

    def is_inside(self, pts, location_type="nodes", **kwargs):
        """Determine which points lie within the mesh.

        For an arbitrary set of points, **is_indside** returns a
        boolean array identifying which points lie within the mesh.

        Parameters
        ----------
        pts : (n_pts, dim) numpy.ndarray
            Locations of input points. Must have same dimension as the mesh.
        location_type : str, optional
            Use *N* to determine points lying within the cluster of mesh
            nodes. Use *CC* to determine points lying within the cluster
            of mesh cell centers.

        Returns
        -------
        (n_pts) numpy.ndarray of bool
            Boolean array identifying points which lie within the mesh

        """
        if "locType" in kwargs:
            warnings.warn(
                "The locType keyword argument has been deprecated, please use location_type. "
                "This will be removed in discretize 1.0.0",
                FutureWarning,
            )
            location_type = kwargs["locType"]
        pts = as_array_n_by_dim(pts, self.dim)

        tensors = self.get_tensor(location_type)

        if location_type[0].lower() == "n" and self._meshType == "CYL":
            # NOTE: for a CYL mesh we add a node to check if we are inside in
            # the radial direction!
            tensors[0] = np.r_[0.0, tensors[0]]
            tensors[1] = np.r_[tensors[1], 2.0 * np.pi]

        inside = np.ones(pts.shape[0], dtype=bool)
        for i, tensor in enumerate(tensors):
            TOL = np.diff(tensor).min() * 1.0e-10
            inside = (
                inside
                & (pts[:, i] >= tensor.min() - TOL)
                & (pts[:, i] <= tensor.max() + TOL)
            )
        return inside

    def _getInterpolationMat(
        self, loc, location_type="cell_centers", zeros_outside=False
    ):
        """Produce an interpolation matrix.

        Parameters
        ----------
        loc : numpy.ndarray
            Location of points to interpolate to

        location_type: str, optional
            What to interpolate

            location_type can be::

                'Ex', 'edges_x'           -> x-component of field defined on x edges
                'Ey', 'edges_y'           -> y-component of field defined on y edges
                'Ez', 'edges_z'           -> z-component of field defined on z edges
                'Fx', 'faces_x'           -> x-component of field defined on x faces
                'Fy', 'faces_y'           -> y-component of field defined on y faces
                'Fz', 'faces_z'           -> z-component of field defined on z faces
                'N', 'nodes'              -> scalar field defined on nodes
                'CC', 'cell_centers'      -> scalar field defined on cell centers
                'CCVx', 'cell_centers_x'  -> x-component of vector field defined on cell centers
                'CCVy', 'cell_centers_y'  -> y-component of vector field defined on cell centers
                'CCVz', 'cell_centers_z'  -> z-component of vector field defined on cell centers

        Returns
        -------
        scipy.sparse.csr_matrix
            M, the interpolation matrix

        """
        loc = as_array_n_by_dim(loc, self.dim)

        if not zeros_outside:
            if not np.all(self.is_inside(loc)):
                raise ValueError("Points outside of mesh")
        else:
            indZeros = np.logical_not(self.is_inside(loc))
            loc[indZeros, :] = np.array([v.mean() for v in self.get_tensor("CC")])

        location_type = self._parse_location_type(location_type)

        if location_type in [
            "faces_x",
            "faces_y",
            "faces_z",
            "edges_x",
            "edges_y",
            "edges_z",
        ]:
            ind = {"x": 0, "y": 1, "z": 2}[location_type[-1]]
            if self.dim < ind:
                raise ValueError("mesh is not high enough dimension.")
            if "f" in location_type.lower():
                items = (self.nFx, self.nFy, self.nFz)[: self.dim]
            else:
                items = (self.nEx, self.nEy, self.nEz)[: self.dim]
            components = [spzeros(loc.shape[0], n) for n in items]
            components[ind] = interpolation_matrix(loc, *self.get_tensor(location_type))
            # remove any zero blocks (hstack complains)
            components = [comp for comp in components if comp.shape[1] > 0]
            Q = sp.hstack(components)

        elif location_type in ["cell_centers", "nodes"]:
            Q = interpolation_matrix(loc, *self.get_tensor(location_type))

        elif location_type in ["cell_centers_x", "cell_centers_y", "cell_centers_z"]:
            Q = interpolation_matrix(loc, *self.get_tensor("CC"))
            Z = spzeros(loc.shape[0], self.nC)
            if location_type[-1] == "x":
                Q = sp.hstack([Q, Z, Z])
            elif location_type[-1] == "y":
                Q = sp.hstack([Z, Q, Z])
            elif location_type[-1] == "z":
                Q = sp.hstack([Z, Z, Q])

        else:
            raise NotImplementedError(
                "getInterpolationMat: location_type=="
                + location_type
                + " and mesh.dim=="
                + str(self.dim)
            )

        if zeros_outside:
            Q[indZeros, :] = 0

        return Q.tocsr()

    def get_interpolation_matrix(  # NOQA D102
        self, loc, location_type="cell_centers", zeros_outside=False, **kwargs
    ):
        # Documentation inherited from discretize.base.BaseMesh
        if "locType" in kwargs:
            warnings.warn(
                "The locType keyword argument has been deprecated, please use location_type. "
                "This will be removed in discretize 1.0.0",
                FutureWarning,
            )
            location_type = kwargs["locType"]
        if "zerosOutside" in kwargs:
            warnings.warn(
                "The zerosOutside keyword argument has been deprecated, please use zeros_outside. "
                "This will be removed in discretize 1.0.0",
                FutureWarning,
            )
            zeros_outside = kwargs["zerosOutside"]
        return self._getInterpolationMat(loc, location_type, zeros_outside)

    def _fastInnerProduct(
        self, projection_type, model=None, invert_model=False, invert_matrix=False
    ):
        """Fast version of getFaceInnerProduct.

        This does not handle the case of a full tensor property.

        Parameters
        ----------
        model : numpy.ndarray
            material property (tensor properties are possible) at each cell center (nC, (1, 3, or 6))

        projection_type : str
            'edges' or 'faces'

        returnP : bool
            returns the projection matrices

        invert_model : bool
            inverts the material property

        invert_matrix : bool
            inverts the matrix

        Returns
        -------
        (n_faces, n_faces) scipy.sparse.csr_matrix
            M, the inner product matrix

        """
        projection_type = projection_type[0].upper()
        if projection_type not in ["F", "E"]:
            raise ValueError("projection_type must be 'F' for faces or 'E' for edges")

        if model is None:
            model = np.ones(self.nC)

        if invert_model:
            model = 1.0 / model

        if is_scalar(model):
            model = model * np.ones(self.nC)

        # number of elements we are averaging (equals dim for regular
        # meshes, but for cyl, where we use symmetry, it is 1 for edge
        # variables and 2 for face variables)
        if self._meshType == "CYL":
            shape = getattr(self, "vn" + projection_type)
            n_elements = sum([1 if x != 0 else 0 for x in shape])
        else:
            n_elements = self.dim

        # Isotropic? or anisotropic?
        if model.size == self.nC:
            Av = getattr(self, "ave" + projection_type + "2CC")
            Vprop = self.cell_volumes * mkvc(model)
            M = n_elements * sdiag(Av.T * Vprop)

        elif model.size == self.nC * self.dim:
            Av = getattr(self, "ave" + projection_type + "2CCV")

            # if cyl, then only certain components are relevant due to symmetry
            # for faces, x, z matters, for edges, y (which is theta) matters
            if self._meshType == "CYL":
                if projection_type == "E":
                    model = model[:, 1]  # this is the action of a projection mat
                elif projection_type == "F":
                    model = model[:, [0, 2]]

            V = sp.kron(sp.identity(n_elements), sdiag(self.cell_volumes))
            M = sdiag(Av.T * V * mkvc(model))
        else:
            return None

        if invert_matrix:
            return sdinv(M)
        else:
            return M

    def _fastInnerProductDeriv(
        self, projection_type, model, invert_model=False, invert_matrix=False
    ):
        """Faster function for inner product derivatives on tensor meshes.

        Parameters
        ----------
        projection_type : str
            'E' or 'F'
        tensorType : TensorType
            type of the tensor
        invert_model : bool
            inverts the material property
        invert_matrix : bool
            inverts the matrix

        Returns
        -------
        function
            dMdmu, the derivative of the inner product matrix
        """
        projection_type = projection_type[0].upper()
        if projection_type not in ["F", "E"]:
            raise ValueError("projection_type must be 'F' for faces or 'E' for edges")

        tensorType = TensorType(self, model)

        dMdprop = None

        if invert_matrix or invert_model:
            MI = self._fastInnerProduct(
                projection_type,
                model,
                invert_model=invert_model,
                invert_matrix=invert_matrix,
            )

        # number of elements we are averaging (equals dim for regular
        # meshes, but for cyl, where we use symmetry, it is 1 for edge
        # variables and 2 for face variables)
        if self._meshType == "CYL":
            shape = getattr(self, "vn" + projection_type)
            n_elements = sum([1 if x != 0 else 0 for x in shape])
        else:
            n_elements = self.dim

        if tensorType == 0:  # isotropic, constant
            Av = getattr(self, "ave" + projection_type + "2CC")
            V = sdiag(self.cell_volumes)
            ones = sp.csr_matrix(
                (np.ones(self.nC), (range(self.nC), np.zeros(self.nC))),
                shape=(self.nC, 1),
            )
            if not invert_matrix and not invert_model:
                dMdprop = n_elements * Av.T * V * ones
            elif invert_matrix and invert_model:
                dMdprop = n_elements * (
                    sdiag(MI.diagonal() ** 2)
                    * Av.T
                    * V
                    * ones
                    * sdiag(1.0 / model**2)
                )
            elif invert_model:
                dMdprop = n_elements * Av.T * V * sdiag(-1.0 / model**2)
            elif invert_matrix:
                dMdprop = n_elements * (sdiag(-MI.diagonal() ** 2) * Av.T * V)

        elif tensorType == 1:  # isotropic, variable in space
            Av = getattr(self, "ave" + projection_type + "2CC")
            V = sdiag(self.cell_volumes)
            if not invert_matrix and not invert_model:
                dMdprop = n_elements * Av.T * V
            elif invert_matrix and invert_model:
                dMdprop = n_elements * (
                    sdiag(MI.diagonal() ** 2) * Av.T * V * sdiag(1.0 / model**2)
                )
            elif invert_model:
                dMdprop = n_elements * Av.T * V * sdiag(-1.0 / model**2)
            elif invert_matrix:
                dMdprop = n_elements * (sdiag(-MI.diagonal() ** 2) * Av.T * V)

        elif tensorType == 2:  # anisotropic
            Av = getattr(self, "ave" + projection_type + "2CCV")
            V = sp.kron(sp.identity(self.dim), sdiag(self.cell_volumes))

            if self._meshType == "CYL":
                Zero = sp.csr_matrix((self.nC, self.nC))
                Eye = sp.eye(self.nC)
                if projection_type == "E":
                    P = sp.hstack([Zero, Eye, Zero])
                    # print(P.todense())
                elif projection_type == "F":
                    P = sp.vstack(
                        [sp.hstack([Eye, Zero, Zero]), sp.hstack([Zero, Zero, Eye])]
                    )
                    # print(P.todense())
            else:
                P = sp.eye(self.nC * self.dim)

            if not invert_matrix and not invert_model:
                dMdprop = Av.T * P * V
            elif invert_matrix and invert_model:
                dMdprop = (
                    sdiag(MI.diagonal() ** 2) * Av.T * P * V * sdiag(1.0 / model**2)
                )
            elif invert_model:
                dMdprop = Av.T * P * V * sdiag(-1.0 / model**2)
            elif invert_matrix:
                dMdprop = sdiag(-MI.diagonal() ** 2) * Av.T * P * V

        if dMdprop is not None:

            def innerProductDeriv(v=None):
                if v is None:
                    warnings.warn(
                        "Depreciation Warning: TensorMesh.innerProductDeriv."
                        " You should be supplying a vector. "
                        "Use: sdiag(u)*dMdprop",
                        FutureWarning,
                    )
                    return dMdprop
                return sdiag(v) * dMdprop

            return innerProductDeriv
        else:
            return None

    # DEPRECATED
    @property
    def hx(self):
        """Width of cells in the x direction.

        Returns
        -------
        numpy.ndarray

        .. deprecated:: 0.5.0
          `hx` will be removed in discretize 1.0.0 to reduce namespace clutter,
          please use `mesh.h[0]`.
        """
        warnings.warn(
            "hx has been deprecated, please access as mesh.h[0]", FutureWarning
        )
        return self.h[0]

    @property
    def hy(self):
        """Width of cells in the y direction.

        Returns
        -------
        numpy.ndarray or None

        .. deprecated:: 0.5.0
          `hy` will be removed in discretize 1.0.0 to reduce namespace clutter,
          please use `mesh.h[1]`.
        """
        warnings.warn(
            "hy has been deprecated, please access as mesh.h[1]", FutureWarning
        )
        return None if self.dim < 2 else self.h[1]

    @property
    def hz(self):
        """Width of cells in the z direction.

        Returns
        -------
        numpy.ndarray or None

        .. deprecated:: 0.5.0
          `hz` will be removed in discretize 1.0.0 to reduce namespace clutter,
          please use `mesh.h[2]`.
        """
        warnings.warn(
            "hz has been deprecated, please access as mesh.h[2]", FutureWarning
        )
        return None if self.dim < 3 else self.h[2]

    vectorNx = deprecate_property(
        "nodes_x", "vectorNx", removal_version="1.0.0", future_warn=True
    )
    vectorNy = deprecate_property(
        "nodes_y", "vectorNy", removal_version="1.0.0", future_warn=True
    )
    vectorNz = deprecate_property(
        "nodes_z", "vectorNz", removal_version="1.0.0", future_warn=True
    )
    vectorCCx = deprecate_property(
        "cell_centers_x", "vectorCCx", removal_version="1.0.0", future_warn=True
    )
    vectorCCy = deprecate_property(
        "cell_centers_y", "vectorCCy", removal_version="1.0.0", future_warn=True
    )
    vectorCCz = deprecate_property(
        "cell_centers_z", "vectorCCz", removal_version="1.0.0", future_warn=True
    )
    isInside = deprecate_method(
        "is_inside", "isInside", removal_version="1.0.0", future_warn=True
    )
    getTensor = deprecate_method(
        "get_tensor", "getTensor", removal_version="1.0.0", future_warn=True
    )
