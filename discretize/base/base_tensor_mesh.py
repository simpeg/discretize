"""
Base class for tensor-product style meshes
"""

import numpy as np
import scipy.sparse as sp

from discretize.base.base_mesh import BaseMesh
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


class BaseTensorMesh(BaseMesh):
    """
    Base class for tensor-product style meshes

    This class contains properites and methods that are common to Cartesian
    and cylindrical meshes. That is, meshes whose cell centers, nodes, faces
    and edges can be constructed with tensor-products of vectors.

    Do not use this class directly! Practical tensor meshes supported in
    discretize will inherit this class; i.e. :class:`discretize.TensorMesh`
    and :class:`~discretize.CylindricalMesh`. Inherit this class if you plan
    to develop a new tensor-style mesh class (e.g. a spherical mesh).

    Parameters
    ----------
    h : list of numpy.ndarray or list
        Defines the cell widths along each axis. The length of the list is equal to the dimension
        of the mesh (1, 2 or 3). For a 3D mesh, the list would have the form *[hx, hy, hz]* .
        Along each axis, the user has 3 choices for defining the cells widths:
            - the widths are defined as a 1D :class:`numpy.ndarray`
            - the widths are defined as a *list* of *tuple* of the form *(dh, nc, [npad])* where *dh* is the cell width, *nc* is the number of cells, and *npad* (optional) is a padding factor denoting exponential increase/decrease in the cell width for each cell; e.g. *[(2., 10, -1.3), (2., 50), (2., 10, 1.3)]*
            - the widths are defined as a mixed *list* of *tuples* and 1D *numpy arrays* .

    origin : numpy.ndarray (dim,) or str of length dim; optional
        Define the origin or 'anchor point' of the mesh; i.e. the bottom-left-frontmost corner.
        By default, the mesh is anchored such that its origin is at *[0, 0, 0]* . The user
        may set the origin 2 ways which instantiating the tensor mesh:
            - a :class:`numpy.ndarray` of shape (*dim* ,) which explicitly defines the x, (y, z) location of the origin.
            - a str of length *dim* specifying whether the zero coordinate along each axis is the first node location ('0'), in the center ('C') or the last node location ('N').

    """

    _meshType = "BASETENSOR"
    _aliases = {
        **BaseMesh._aliases,
        **{
            "gridCC": "cell_centers",
            "gridN": "nodes",
            "gridFx": "faces_x",
            "gridFy": "faces_y",
            "gridFz": "faces_z",
            "gridEx": "edges_x",
            "gridEy": "edges_y",
            "gridEz": "edges_z",
        },
    }

    _unitDimensions = [1, 1, 1]
    _items = {"h"} | BaseMesh._items

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
        """Cell widths along each axis direction

        The widths of the cells along each axis direction are returned
        as a tuple of 1D arrays; e.g. (hx, hy, hz) for a 3D mesh.
        The lengths of the 1D arrays in the tuple are given by
        :py:attr:`~discretize.base.BaseMesh.shape_cells`. Ordering
        begins at the bottom southwest corner. These are the
        cell widths used when creating the mesh.

        Returns
        -------
        tuple of numpy.ndarray
            Cell widths along each axis direction. This depends on the mesh class:

                - :class:`~discretize.TensorMesh`: cell widths along the *x* , [*y* and *z* ] directions
                - :class:`~discretize.CylindricalMesh`: cell widths along the *r*, :math:`\\phi` and *z* directions
                - :class:`~discretize.TreeMesh`: cells widths of the *underlying tensor mesh* along the *x* , *y* [and *z* ] directions

        """
        return self._h

    @BaseMesh.origin.setter
    def origin(self, value):
        # ensure value is a 1D array at all times
        try:
            value = list(value)
        except:
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
        """
        Return x-coordinates of the nodes along the x-direction

        For 1D, 2D and 3D meshes, this property returns a 1D vector
        containing the x-coordinate values of the nodes along the
        x-direction. For instances of :class:`~discretize.TensorMesh` or
        :class:`~discretize.CylindricalMesh`, this is equivalent to
        the node positions which define the tensor along the x-axis.
        For instances of :class:`~discretize.TreeMesh` however, this property
        returns the x-coordinate values of the nodes along the x-direction
        for the underlying tensor mesh .

        Returns
        -------
        np.ndarray of float (n_nodes_x,)
            A 1D array containing the x-coordinates of the nodes along
            the x-direction.

        """
        return np.r_[self.origin[0], self.h[0]].cumsum()

    @property
    def nodes_y(self):
        """
        Return y-coordinates of the nodes along the y-direction

        For 2D and 3D meshes, this property returns a 1D vector
        containing the y-coordinate values of the nodes along the
        y-direction. For instances of :class:`~discretize.TensorMesh` or
        :class:`~discretize.CylindricalMesh`, this is equivalent to
        the node positions which define the tensor along the y-axis.
        For instances of :class:`~discretize.TreeMesh` however, this property
        returns the y-coordinate values of the nodes along the y-direction
        for the underlying tensor mesh .

        Returns
        -------
        np.ndarray of float (n_nodes_y,) or None
            A 1D array containing the y-coordinates of the nodes along
            the y-direction. Returns *None* for 1D meshes.

        """
        return None if self.dim < 2 else np.r_[self.origin[1], self.h[1]].cumsum()

    @property
    def nodes_z(self):
        """
        Return z-coordinates of the nodes along the z-direction

        For 3D meshes, this property returns a 1D vector
        containing the z-coordinate values of the nodes along the
        z-direction. For instances of :class:`~discretize.TensorMesh` or
        :class:`~discretize.CylindricalMesh`, this is equivalent to
        the node positions which define the tensor along the z-axis.
        For instances of :class:`~discretize.TreeMesh` however, this property
        returns the z-coordinate values of the nodes along the z-direction
        for the underlying tensor mesh .

        Returns
        -------
        np.ndarray of float (n_nodes_z,)
            A 1D array containing the z-coordinates of the nodes along
            the z-direction. Returns *None* for 1D and 2D meshes.

        """
        return None if self.dim < 3 else np.r_[self.origin[2], self.h[2]].cumsum()

    @property
    def cell_centers_x(self):
        """
        Return x-coordinates of the cell centers along the x-direction

        For 1D, 2D and 3D meshes, this property returns a 1D vector
        containing the x-coordinate values of the cell centers along the
        x-direction. For instances of :class:`~discretize.TensorMesh` or
        :class:`~discretize.CylindricalMesh`, this is equivalent to
        the cell center positions which define the tensor along the x-axis.
        For instances of :class:`~discretize.TreeMesh` however, this property
        returns the x-coordinate values of the cell centers along the x-direction
        for the underlying tensor mesh .

        Returns
        -------
        np.ndarray of float (n_cells_x,)
            A 1D array containing the x-coordinates of the cell centers along
            the x-direction.

        """
        nodes = self.nodes_x
        return (nodes[1:] + nodes[:-1]) / 2

    @property
    def cell_centers_y(self):
        """
        Return y-coordinates of the cell centers along the y-direction

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
        np.ndarray of float (n_cells_y,)
            A 1D array containing the y-coordinates of the cell centers along
            the y-direction. Returns *None* for 1D meshes.

        """
        if self.dim < 2:
            return None
        nodes = self.nodes_y
        return (nodes[1:] + nodes[:-1]) / 2

    @property
    def cell_centers_z(self):
        """
        Return z-coordinates of the cell centers along the z-direction

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
        np.ndarray of float (n_cells_z,)
            A 1D array containing the z-coordinates of the cell centers along
            the z-direction. Returns *None* for 1D and 2D meshes.

        """
        if self.dim < 3:
            return None
        nodes = self.nodes_z
        return (nodes[1:] + nodes[:-1]) / 2

    @property
    def cell_centers(self):
        """Return gridded cell center locations

        This property returns a numpy array of shape (n_cells, dim)
        containing gridded cell center locations for all cells in the
        mesh. The first row corresponds to the bottom-front-leftmost cell. 
        The cells are ordered along the x, then y, then z directions.

        Returns
        -------
        np.ndarray of float (n_cells, dim)
            Gridded cell center locations

        Examples
        --------
        The following is a 1D example.

        >>> from discretize import TensorMesh
        >>> hx = np.ones(5)
        >>> mesh = TensorMesh([hx], '0')
        >>> print(mesh.grid_cell_centers)

        The following is a 3D example.

        >>> from discretize import TensorMesh
        >>> hx, hy, hz = np.ones(2), 2*np.ones(2), 3*np.ones(2)
        >>> mesh = TensorMesh([hx, hy, hz], '000')
        >>> print(mesh.grid_cell_centers)

        """
        return self._getTensorGrid("cell_centers")

    @property
    def nodes(self):
        """Return gridded node locations

        This property returns a numpy array of shape (n_nodes, dim)
        containing gridded node locations for all nodes in the
        mesh. The first row corresponds to the bottom-front-leftmost node. 
        The nodes are ordered along the x, then y, then z directions.

        Returns
        -------
        np.ndarray of float (n_nodes, dim)
            Gridded node locations

        Examples
        --------
        The following is a 1D example.

        >>> from discretize import TensorMesh
        >>> hx = np.ones(5)
        >>> mesh = TensorMesh([hx], '0')
        >>> print(mesh.grid_nodes)

        The following is a 3D example.

        >>> from discretize import TensorMesh
        >>> hx, hy, hz = np.ones(2), 2*np.ones(2), 3*np.ones(2)
        >>> mesh = TensorMesh([hx, hy, hz], '000')
        >>> print(mesh.grid_nodes)

        """
        return self._getTensorGrid("nodes")

    @property
    def boundary_nodes(self):
        """Boundary node locations

        This property returns the locations of the nodes on
        the boundary of the mesh as a numpy array. The shape
        of the numpy array is the number of boundary nodes by
        the dimension of the mesh.

        Returns
        -------
        np.ndarray of float (n_boundary_nodes, dim)
            Boundary node locations 
        """
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
        np.ndarray of float (n_cells, dim)
            Dimensions of all mesh cells as staggered grid

        Examples
        --------
        The following is a 1D example.

        >>> from discretize import TensorMesh
        >>> hx = np.ones(5)
        >>> mesh = TensorMesh([hx])
        >>> print(mesh.h_gridded)

        The following is a 3D example.

        >>> from discretize import TensorMesh
        >>> hx, hy, hz = np.ones(2), 2*np.ones(2), 3*np.ones(2)
        >>> mesh = TensorMesh([hx, hy, hz])
        >>> print(mesh.h_gridded)

        """
        if self.dim == 1:
            return self.h[0][:, None]
        return ndgrid(*self.h)

    @property
    def faces_x(self):
        """Gridded x-face locations

        This property returns a numpy array of shape (n_faces_x, dim)
        containing gridded locations for all x-faces in the
        mesh. The first row corresponds to the bottom-front-leftmost x-face. 
        The x-faces are ordered along the x, then y, then z directions.

        Returns
        -------
        np.ndarray of float (n_faces_x, dim)
            Gridded x-face locations
        """
        if self.nFx == 0:
            return
        return self._getTensorGrid("faces_x")

    @property
    def faces_y(self):
        """Gridded y-face locations

        This property returns a numpy array of shape (n_faces_y, dim)
        containing gridded locations for all y-faces in the
        mesh. The first row corresponds to the bottom-front-leftmost y-face. 
        The y-faces are ordered along the x, then y, then z directions.

        Returns
        -------
        np.ndarray of float (n_faces_y, dim) or None
            Gridded y-face locations for 2D and 3D mesh. Returns *None* for 1D meshes.
        """
        if self.nFy == 0 or self.dim < 2:
            return
        return self._getTensorGrid("faces_y")

    @property
    def faces_z(self):
        """Gridded z-face locations

        This property returns a numpy array of shape (n_faces_z, dim)
        containing gridded locations for all z-faces in the
        mesh. The first row corresponds to the bottom-front-leftmost z-face. 
        The z-faces are ordered along the x, then y, then z directions.

        Returns
        -------
        np.ndarray of float (n_faces_z, dim) or None
            Gridded z-face locations for 3D mesh. Returns *None* for 1D and 2D meshes.
        """
        if self.nFz == 0 or self.dim < 3:
            return
        return self._getTensorGrid("faces_z")

    @property
    def faces(self):
        """Gridded face locations

        This property returns a numpy array of shape (n_faces, dim)
        containing gridded locations for all faces in the mesh.
        The first row corresponds to the bottom-front-leftmost x-face.
        The output array returns the x-faces, then the y-faces, then
        the z-faces; i.e. *mesh.faces* is equivalent to *np.r_[mesh.faces_x, mesh.faces_y, mesh.face_z]* .
        For each face type, the locations are ordered along the x, then y, then z directions.

        Returns
        -------
        numpy.ndarray of float (n_faces, dim)
            Gridded face locations

        """
        faces = self.faces_x
        if self.dim > 1:
            faces = np.r_[faces, self.faces_y]
        if self.dim > 2:
            faces = np.r_[faces, self.faces_z]
        return faces

    @property
    def boundary_faces(self):
        """Boundary face locations

        This property returns the locations of the faces on
        the boundary of the mesh as a numpy array. The shape
        of the numpy array is the number of boundary faces by
        the dimension of the mesh.

        Returns
        -------
        np.ndarray of float (n_boundary_faces, dim)
            Boundary faces locations 
        """
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
    def boundary_face_outward_normals(self):
        """Outward normal vectors of boundary faces

        This property returns the outward normal vectors of faces
        the boundary of the mesh as a numpy array. The shape
        of the numpy array is the number of boundary faces by
        the dimension of the mesh.

        Returns
        -------
        np.ndarray of float (n_boundary_faces, dim)
            Outward normal vectors of boundary faces 
        """
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
        """Gridded x-edge locations

        This property returns a numpy array of shape (n_edges_x, dim)
        containing gridded locations for all x-edges in the mesh.
        The first row corresponds to the bottom-front-leftmost x-edge. 
        The x-edges are ordered along the x, then y, then z directions.

        Returns
        -------
        np.ndarray of float (n_edges_x, dim)
            Gridded x-edge locations
        """
        if self.nEx == 0:
            return
        return self._getTensorGrid("edges_x")

    @property
    def edges_y(self):
        """Gridded y-edge locations

        This property returns a numpy array of shape (n_edges_y, dim)
        containing gridded locations for all y-edges in the mesh.
        The first row corresponds to the bottom-front-leftmost y-edge. 
        The y-edges are ordered along the x, then y, then z directions.

        Returns
        -------
        np.ndarray of float (n_edges_y, dim)
            Gridded y-edge locations. Returns *None* for 1D meshes.
        """
        if self.nEy == 0 or self.dim < 2:
            return
        return self._getTensorGrid("edges_y")

    @property
    def edges_z(self):
        """Gridded z-edge locations

        This property returns a numpy array of shape (n_edges_z, dim)
        containing gridded locations for all z-edges in the mesh.
        The first row corresponds to the bottom-front-leftmost z-edge. 
        The z-edges are ordered along the x, then y, then z directions.

        Returns
        -------
        np.ndarray of float (n_edges_z, dim)
            Gridded z-edge locations. Returns *None* for 1D and 2D meshes.
        """
        if self.nEz == 0 or self.dim < 3:
            return
        return self._getTensorGrid("edges_z")

    @property
    def edges(self):
        """Gridded edge locations

        This property returns a numpy array of shape (n_edges, dim)
        containing gridded locations for all edges in the mesh.
        The first row corresponds to the bottom-front-leftmost x-edge.
        The output array returns the x-edges, then the y-edges, then
        the z-edges; i.e. *mesh.edges* is equivalent to *np.r_[mesh.edges_x, mesh.edges_y, mesh.edges_z]* .
        For each edge type, the locations are ordered along the x, then y, then z directions.

        Returns
        -------
        numpy.ndarray of float (n_edges, dim)
            Gridded edge locations

        """
        edges = self.edges_x
        if self.dim > 1:
            edges = np.r_[edges, self.edges_y]
        if self.dim > 2:
            edges = np.r_[edges, self.edges_z]
        return edges

    @property
    def boundary_edges(self):
        """Boundary edge locations

        This property returns the locations of the edges on
        the boundary of the mesh as a numpy array. The shape
        of the numpy array is the number of boundary edges by
        the dimension of the mesh.

        Returns
        -------
        np.ndarray of float (n_boundary_edges, dim)
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
        """Returns the base 1D arrays for a specified mesh tensor.

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
        list of 1D numpy.ndarray
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
        """Determine which points lie within the mesh

        For an arbitrary set of points, **is_indside** returns a
        boolean array identifying which points lie within the mesh.

        Parameters
        ----------
        pts : np.ndarray (n_pts, dim)
            Locations of input points. Must have same dimension as the mesh.
        location_type : str
            Use *N* to determine points lying within the cluster of mesh
            nodes. Use *CC* to determine points lying within the cluster
            of mesh cell centers.

        Returns
        -------
        np.ndarray of bool (n_pts,)
            Boolean array identifying points which lie within the mesh

        """
        if "locType" in kwargs:
            warnings.warn(
                "The locType keyword argument has been deprecated, please use location_type. "
                "This will be removed in discretize 1.0.0",
                DeprecationWarning,
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
        """Produces interpolation matrix

        Parameters
        ----------
        loc : numpy.ndarray
            Location of points to interpolate to

        location_type: stc
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

    def get_interpolation_matrix(
        self, loc, location_type="cell_centers", zeros_outside=False, **kwargs
    ):
        """Construct linear interpolation matrix from mesh

        This method constructs a linear interpolation matrix from tensor locations
        (nodes, cell-centers, faces, etc...) on the mesh to a set of arbitrary locations.

        Parameters
        ----------
        loc : numpy.ndarray (*, dim)
            Location of points being to interpolate to. Must have same dimensions as the mesh.

        location_type : str
            Tensor locations on the mesh being interpolated from. *location_type* must be one of::

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

        zeros_outside : bool (default = False )
            If *False*, nearest neighbour is used to compute the interpolate value
            at locations outside the mesh. If *True* , values at locations outside
            the mesh will be zero.

        Returns
        -------

        scipy.sparse.csr_matrix
            A sparse matrix which interpolates the specified tensor quantity on mesh to
            the set of specified locations.


        Examples
        --------

        Here is a 1D example where a function evaluated on the nodes
        is interpolated to a set of random locations. To compare the accuracy, the
        function is evaluated at the set of random locations.

        >>> from discretize import TensorMesh
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> 
        >>> np.random.seed(14)
        >>> locs = np.random.rand(50)*0.8+0.1
        >>> dense = np.linspace(0, 1, 200)
        >>> fun = lambda x: np.cos(2*np.pi*x)
        >>> 
        >>> hx = 0.125 * np.ones(8)
        >>> mesh1D = TensorMesh([hx])
        >>> Q = mesh1D.get_interpolation_matrix(locs, 'nodes')
        >>> 
        >>> fig1 = plt.figure(figsize=(5, 3))
        >>> ax = fig1.add_axes([0.05, 0.05, 0.9, 0.9])
        >>> ax.plot(dense, fun(dense), 'k:', lw=3)
        >>> ax.plot(mesh1D.nodes, fun(mesh1D.nodes), 'ks', markersize=8)
        >>> ax.plot(locs, Q*fun(mesh1D.nodes), 'go', markersize=4)
        >>> ax.plot(locs, fun(locs), 'rs', markersize=4)
        >>> ax.legend(
        >>>     [
        >>>         'True Function',
        >>>         'True (discrete loc.)',
        >>>         'Interpolated (computed)',
        >>>         'True (interp. loc.)'
        >>>     ],
        >>>     loc='upper center'
        >>> )
        >>> plt.show()

        Here, demonstrate a similar example on a 2D mesh using a 2D Gaussian distribution.
        We interpolate the Gaussian from the nodes to cell centers and examine the relative
        error.

        >>> hx = np.ones(10)
        >>> hy = np.ones(10)
        >>> mesh2D = TensorMesh([hx, hy], x0='CC')
        >>> 
        >>> def fun(x, y):
        >>>     return np.exp(-(x**2 + y**2)/2**2)
        >>> 
        >>> nodes = mesh2D.nodes
        >>> val_nodes = fun(nodes[:, 0], nodes[:, 1])
        >>> centers = mesh2D.cell_centers
        >>> val_centers = fun(centers[:, 0], centers[:, 1])
        >>> 
        >>> A = mesh2D.get_interpolation_matrix(centers, 'nodes')
        >>> val_interp = A.dot(val_nodes)
        >>> fig = plt.figure(figsize=(11,3.3))
        >>> clim = (0., 1.)
        >>> ax1 = fig.add_subplot(131)
        >>> ax2 = fig.add_subplot(132)
        >>> ax3 = fig.add_subplot(133)
        >>> mesh2D.plot_image(val_centers, ax=ax1, clim=clim)
        >>> mesh2D.plot_image(val_interp, ax=ax2, clim=clim)
        >>> mesh2D.plot_image(val_centers-val_interp, ax=ax3, clim=clim)
        >>> ax1.set_title('Analytic at Centers')
        >>> ax2.set_title('Interpolated from Nodes')
        >>> ax3.set_title('Relative Error')
        >>> plt.show()

        """
        if "locType" in kwargs:
            warnings.warn(
                "The locType keyword argument has been deprecated, please use location_type. "
                "This will be removed in discretize 1.0.0",
                DeprecationWarning,
            )
            location_type = kwargs["locType"]
        if "zerosOutside" in kwargs:
            warnings.warn(
                "The zerosOutside keyword argument has been deprecated, please use zeros_outside. "
                "This will be removed in discretize 1.0.0",
                DeprecationWarning,
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
        scipy.sparse.csr_matrix
            M, the inner product matrix (nF, nF)

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
        """

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
                    * sdiag(1.0 / model ** 2)
                )
            elif invert_model:
                dMdprop = n_elements * Av.T * V * sdiag(-1.0 / model ** 2)
            elif invert_matrix:
                dMdprop = n_elements * (sdiag(-MI.diagonal() ** 2) * Av.T * V)

        elif tensorType == 1:  # isotropic, variable in space
            Av = getattr(self, "ave" + projection_type + "2CC")
            V = sdiag(self.cell_volumes)
            if not invert_matrix and not invert_model:
                dMdprop = n_elements * Av.T * V
            elif invert_matrix and invert_model:
                dMdprop = n_elements * (
                    sdiag(MI.diagonal() ** 2) * Av.T * V * sdiag(1.0 / model ** 2)
                )
            elif invert_model:
                dMdprop = n_elements * Av.T * V * sdiag(-1.0 / model ** 2)
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
                    sdiag(MI.diagonal() ** 2) * Av.T * P * V * sdiag(1.0 / model ** 2)
                )
            elif invert_model:
                dMdprop = Av.T * P * V * sdiag(-1.0 / model ** 2)
            elif invert_matrix:
                dMdprop = sdiag(-MI.diagonal() ** 2) * Av.T * P * V

        if dMdprop is not None:

            def innerProductDeriv(v=None):
                if v is None:
                    warnings.warn(
                        "Depreciation Warning: TensorMesh.innerProductDeriv."
                        " You should be supplying a vector. "
                        "Use: sdiag(u)*dMdprop",
                        DeprecationWarning,
                    )
                    return dMdprop
                return sdiag(v) * dMdprop

            return innerProductDeriv
        else:
            return None

    # DEPRECATED
    @property
    def hx(self):
        """Width of cells in the x direction

        Returns
        -------
        numpy.ndarray

        .. deprecated:: 0.5.0
          `hx` will be removed in discretize 1.0.0 to reduce namespace clutter,
          please use `mesh.h[0]`.
        """
        warnings.warn(
            "hx has been deprecated, please access as mesh.h[0]", DeprecationWarning
        )
        return self.h[0]

    @property
    def hy(self):
        """Width of cells in the y direction

        Returns
        -------
        numpy.ndarray or None

        .. deprecated:: 0.5.0
          `hy` will be removed in discretize 1.0.0 to reduce namespace clutter,
          please use `mesh.h[1]`.
        """
        warnings.warn(
            "hy has been deprecated, please access as mesh.h[1]", DeprecationWarning
        )
        return None if self.dim < 2 else self.h[1]

    @property
    def hz(self):
        """Width of cells in the z direction

        Returns
        -------
        numpy.ndarray or None

        .. deprecated:: 0.5.0
          `hz` will be removed in discretize 1.0.0 to reduce namespace clutter,
          please use `mesh.h[2]`.
        """
        warnings.warn(
            "hz has been deprecated, please access as mesh.h[2]", DeprecationWarning
        )
        return None if self.dim < 3 else self.h[2]

    vectorNx = deprecate_property("nodes_x", "vectorNx", removal_version="1.0.0")
    vectorNy = deprecate_property("nodes_y", "vectorNy", removal_version="1.0.0")
    vectorNz = deprecate_property("nodes_z", "vectorNz", removal_version="1.0.0")
    vectorCCx = deprecate_property(
        "cell_centers_x", "vectorCCx", removal_version="1.0.0"
    )
    vectorCCy = deprecate_property(
        "cell_centers_y", "vectorCCy", removal_version="1.0.0"
    )
    vectorCCz = deprecate_property(
        "cell_centers_z", "vectorCCz", removal_version="1.0.0"
    )
    getInterpolationMat = deprecate_method(
        "get_interpolation_matrix", "getInterpolationMat", removal_version="1.0.0"
    )
    isInside = deprecate_method("is_inside", "isInside", removal_version="1.0.0")
    getTensor = deprecate_method("get_tensor", "getTensor", removal_version="1.0.0")


BaseTensorMesh.__module__ = "discretize.base"
