"""Base classes for all regular shaped meshes supported in ``discretize``."""

import numpy as np
from discretize.utils import mkvc, Identity
from discretize.base.base_mesh import BaseMesh
from discretize.utils.code_utils import deprecate_method
import warnings


class BaseRegularMesh(BaseMesh):
    """Base Regular mesh class for the ``discretize`` package.

    The ``BaseRegularMesh`` class does all the basic counting and organizing
    you wouldn't want to do manually. ``BaseRegularMesh`` is a class that should
    always be inherited by meshes with a regular structure; e.g.
    :class:`~discretize.TensorMesh`, :class:`~discretize.CylindricalMesh`,
    :class:`~discretize.TreeMesh` or :class:`~discretize.CurvilinearMesh`.

    Parameters
    ----------
    shape_cells : array_like of int
        number of cells in each dimension
    origin : array_like of float, optional
        origin of the bottom south west corner of the mesh, defaults to 0.
    orientation : discretize.utils.Identity or array_like of float, optional
        Orientation of the three major axes of the mesh; defaults to :class:`~discretize.utils.Identity`.
        If provided, this must be an orthogonal matrix with the correct dimension.
    reference_system : {'cartesian', 'cylindrical', 'spherical'}
        Can also be a shorthand version of these, e.g. {'car[t]', 'cy[l]', 'sph'}
    """

    _aliases = {
        **BaseMesh._aliases,
        "nEx": "n_edges_x",
        "nEy": "n_edges_y",
        "nEz": "n_edges_z",
        "vnE": "n_edges_per_direction",
        "nFx": "n_faces_x",
        "nFy": "n_faces_y",
        "nFz": "n_faces_z",
        "vnF": "n_faces_per_direction",
        "vnC": "shape_cells",
    }

    _items = {"shape_cells", "origin", "orientation", "reference_system"}

    # Instantiate the class
    def __init__(
        self,
        shape_cells,
        origin=None,
        orientation=None,
        reference_system=None,
        **kwargs,
    ):
        if "n" in kwargs:
            shape_cells = kwargs.pop("n")
        if "x0" in kwargs:
            origin = kwargs.pop("x0")
        axis_u = kwargs.pop("axis_u", None)
        axis_v = kwargs.pop("axis_v", None)
        axis_w = kwargs.pop("axis_w", None)
        if axis_u is not None and axis_v is not None and axis_w is not None:
            orientation = np.array([axis_u, axis_v, axis_w])

        shape_cells = tuple((int(val) for val in shape_cells))
        self._shape_cells = shape_cells
        # some default values
        if origin is None:
            origin = np.zeros(self.dim)
        self.origin = origin

        if orientation is None:
            orientation = Identity()

        self.orientation = orientation
        if reference_system is None:
            reference_system = "cartesian"
        self.reference_system = reference_system
        super().__init__(**kwargs)

    @property
    def origin(self):
        """Origin or 'anchor point' of the mesh.

        For a mesh defined in Cartesian coordinates (e.g.
        :class:`~discretize.TensorMesh`, :class:`~discretize.CylindricalMesh`,
        :class:`~discretize.TreeMesh`), *origin* is the
        bottom southwest corner. For a :class:`~discretize.CylindricalMesh`,
        *origin* is the bottom of the axis of rotational symmetry
        for the mesh (i.e. bottom of z-axis).

        Returns
        -------
        (dim) numpy.ndarray of float
            origin location
        """
        return self._origin

    @origin.setter
    def origin(self, value):
        # ensure the value is a numpy array
        value = np.asarray(value, dtype=np.float64)
        value = np.atleast_1d(value)
        if len(value) != self.dim:
            raise ValueError(
                f"origin and shape must be the same length, got {len(value)} and {self.dim}"
            )
        self._origin = value

    @property
    def shape_cells(self):
        """Number of cells in each coordinate direction.

        For meshes of class :class:`~discretize.TensorMesh`,
        :class:`~discretize.CylindricalMesh` or :class:`~discretize.CurvilinearMesh`,
        **shape_cells** returns the number of cells along each coordinate axis direction.
        For mesh of class :class:`~discretize.TreeMesh`, *shape_cells* returns
        the number of underlying tensor mesh cells along each coordinate direction.

        Returns
        -------
        (dim) tuple of int
            the number of cells in each coordinate direcion

        Notes
        -----
        Property also accessible as using the shorthand **vnC**
        """
        return self._shape_cells

    @property
    def orientation(self):
        """Rotation matrix defining mesh axes relative to Cartesian.

        This property returns a rotation matrix between the local coordinate
        axes of the mesh and the standard Cartesian axes. For a 3D mesh, this
        would define the x, y and z axes of the mesh relative to the Easting,
        Northing and elevation directions. The *orientation* property can
        be used to transform locations from a local coordinate
        system to a conventional Cartesian system. By default, *orientation*
        is an identity matrix of shape (mesh.dim, mesh.dim).

        Returns
        -------
        (dim, dim) numpy.ndarray of float
            Square rotation matrix defining orientation

        Examples
        --------
        For a visual example of this, please see the figure in the
        docs for :class:`~discretize.mixins.InterfaceVTK`.
        """
        return self._orientation

    @orientation.setter
    def orientation(self, value):
        if isinstance(value, Identity):
            self._orientation = np.identity(self.dim)
        else:
            R = np.atleast_2d(np.asarray(value, dtype=np.float64))
            dim = self.dim
            if R.shape != (dim, dim):
                raise ValueError(
                    f"Orientation matrix must be square and of shape {(dim, dim)}, got {R.shape}"
                )
            # Ensure each row is unitary
            R = R / np.linalg.norm(R, axis=1)[:, None]
            # Check if matrix is orthogonal
            if not np.allclose(R @ R.T, np.identity(self.dim), rtol=1.0e-5, atol=1e-6):
                raise ValueError("Orientation matrix is not orthogonal")
            self._orientation = R

    @property
    def reference_system(self):
        """Coordinate reference system.

        The type of coordinate reference frame. Will be one of the values "cartesian",
        "cylindrical", or "spherical".

        Returns
        -------
        str {'cartesian', 'cylindrical', 'spherical'}
            The coordinate system associated with the mesh.
        """
        return self._reference_system

    @reference_system.setter
    def reference_system(self, value):
        """Check if the reference system is of a known type."""
        choices = ["cartesian", "cylindrical", "spherical"]
        # Here are a few abbreviations that users can harnes
        abrevs = {
            "car": choices[0],
            "cart": choices[0],
            "cy": choices[1],
            "cyl": choices[1],
            "sph": choices[2],
        }
        # Get the name and fix it if it is abbreviated
        value = value.lower()
        value = abrevs.get(value, value)
        if value not in choices:
            raise ValueError(
                "Coordinate system ({}) unknown.".format(self.reference_system)
            )
        self._reference_system = value

    @property
    def x0(self):
        """Alias for the :py:attr:`~.BaseRegularMesh.origin`.

        See Also
        --------
        origin
        """
        return self.origin

    @x0.setter
    def x0(self, val):
        self.origin = val

    @property
    def dim(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        return len(self.shape_cells)

    @property
    def n_cells(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        return int(np.prod(self.shape_cells))

    @property
    def n_nodes(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        return int(np.prod([x + 1 for x in self.shape_cells]))

    @property
    def n_edges_x(self):
        """Number of x-edges in the mesh.

        This property returns the number of edges that
        are parallel to the x-axis; i.e. x-edges.

        Returns
        -------
        int
            Number of x-edges in the mesh

        Notes
        -----
        Property also accessible as using the shorthand **nEx**
        """
        return int(np.prod([x + y for x, y in zip(self.shape_cells, (0, 1, 1))]))

    @property
    def n_edges_y(self):
        """Number of y-edges in the mesh.

        This property returns the number of edges that
        are parallel to the y-axis; i.e. y-edges.

        Returns
        -------
        int
            Number of y-edges in the mesh

        Notes
        -----
        Property also accessible as using the shorthand **nEy**
        """
        if self.dim < 2:
            return None
        return int(np.prod([x + y for x, y in zip(self.shape_cells, (1, 0, 1))]))

    @property
    def n_edges_z(self):
        """Number of z-edges in the mesh.

        This property returns the number of edges that
        are parallel to the z-axis; i.e. z-edges.

        Returns
        -------
        int
            Number of z-edges in the mesh

        Notes
        -----
        Property also accessible as using the shorthand **nEz**
        """
        if self.dim < 3:
            return None
        return int(np.prod([x + y for x, y in zip(self.shape_cells, (1, 1, 0))]))

    @property
    def n_edges_per_direction(self):
        """The number of edges in each direction.

        This property returns a tuple with the number of edges
        in each axis direction of the mesh. For a 3D mesh,
        *n_edges_per_direction* would return a tuple of the form
        (nEx, nEy, nEz). Thus the length of the
        tuple depends on the dimension of the mesh.

        Returns
        -------
        (dim) tuple of int
            Number of edges in each direction

        Notes
        -----
        Property also accessible as using the shorthand **vnE**

        Examples
        --------
        >>> import discretize
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> M = discretize.TensorMesh([np.ones(n) for n in [2,3]])
        >>> M.plot_grid(edges=True)
        >>> plt.show()
        """
        return tuple(
            x for x in [self.n_edges_x, self.n_edges_y, self.n_edges_z] if x is not None
        )

    @property
    def n_edges(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        n = self.n_edges_x
        if self.dim > 1:
            n += self.n_edges_y
        if self.dim > 2:
            n += self.n_edges_z
        return n

    @property
    def n_faces_x(self):
        """Number of x-faces in the mesh.

        This property returns the number of faces whose normal
        vector is parallel to the x-axis; i.e. x-faces.

        Returns
        -------
        int
            Number of x-faces in the mesh

        Notes
        -----
        Property also accessible as using the shorthand **nFx**
        """
        return int(np.prod([x + y for x, y in zip(self.shape_cells, (1, 0, 0))]))

    @property
    def n_faces_y(self):
        """Number of y-faces in the mesh.

        This property returns the number of faces whose normal
        vector is parallel to the y-axis; i.e. y-faces.

        Returns
        -------
        int
            Number of y-faces in the mesh

        Notes
        -----
        Property also accessible as using the shorthand **nFy**
        """
        if self.dim < 2:
            return None
        return int(np.prod([x + y for x, y in zip(self.shape_cells, (0, 1, 0))]))

    @property
    def n_faces_z(self):
        """Number of z-faces in the mesh.

        This property returns the number of faces whose normal
        vector is parallel to the z-axis; i.e. z-faces.

        Returns
        -------
        int
            Number of z-faces in the mesh

        Notes
        -----
        Property also accessible as using the shorthand **nFz**
        """
        if self.dim < 3:
            return None
        return int(np.prod([x + y for x, y in zip(self.shape_cells, (0, 0, 1))]))

    @property
    def n_faces_per_direction(self):
        """The number of faces in each axis direction.

        This property returns a tuple with the number of faces
        in each axis direction of the mesh. For a 3D mesh,
        *n_faces_per_direction* would return a tuple of the form
        (nFx, nFy, nFz). Thus the length of the
        tuple depends on the dimension of the mesh.

        Returns
        -------
        (dim) tuple of int
            Number of faces in each axis direction

        Notes
        -----
        Property also accessible as using the shorthand **vnF**

        Examples
        --------
        >>> import discretize
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> M = discretize.TensorMesh([np.ones(n) for n in [2,3]])
        >>> M.plot_grid(faces=True)
        >>> plt.show()
        """
        return tuple(
            x for x in [self.n_faces_x, self.n_faces_y, self.n_faces_z] if x is not None
        )

    @property
    def n_faces(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        n = self.n_faces_x
        if self.dim > 1:
            n += self.n_faces_y
        if self.dim > 2:
            n += self.n_faces_z
        return n

    @property
    def face_normals(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        if self.dim == 2:
            nX = np.c_[np.ones(self.n_faces_x), np.zeros(self.n_faces_x)]
            nY = np.c_[np.zeros(self.n_faces_y), np.ones(self.n_faces_y)]
            return np.r_[nX, nY]
        elif self.dim == 3:
            nX = np.c_[
                np.ones(self.n_faces_x),
                np.zeros(self.n_faces_x),
                np.zeros(self.n_faces_x),
            ]
            nY = np.c_[
                np.zeros(self.n_faces_y),
                np.ones(self.n_faces_y),
                np.zeros(self.n_faces_y),
            ]
            nZ = np.c_[
                np.zeros(self.n_faces_z),
                np.zeros(self.n_faces_z),
                np.ones(self.n_faces_z),
            ]
            return np.r_[nX, nY, nZ]

    @property
    def edge_tangents(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        if self.dim == 2:
            tX = np.c_[np.ones(self.n_edges_x), np.zeros(self.n_edges_x)]
            tY = np.c_[np.zeros(self.n_edges_y), np.ones(self.n_edges_y)]
            return np.r_[tX, tY]
        elif self.dim == 3:
            tX = np.c_[
                np.ones(self.n_edges_x),
                np.zeros(self.n_edges_x),
                np.zeros(self.n_edges_x),
            ]
            tY = np.c_[
                np.zeros(self.n_edges_y),
                np.ones(self.n_edges_y),
                np.zeros(self.n_edges_y),
            ]
            tZ = np.c_[
                np.zeros(self.n_edges_z),
                np.zeros(self.n_edges_z),
                np.ones(self.n_edges_z),
            ]
            return np.r_[tX, tY, tZ]

    @property
    def reference_is_rotated(self):
        """Indicate whether mesh uses standard coordinate axes.

        The standard basis vectors defining the x, y, and z axes of
        a mesh are :math:`(1,0,0)`, :math:`(0,1,0)` and :math:`(0,0,1)`,
        respectively. However, the :py:attr:`~BaseRegularMesh.orientation` property
        can be used to define rotated coordinate axes for our mesh.

        The *reference_is_rotated* property determines
        whether the mesh is using standard coordinate axes.
        If the coordinate axes are standard, *mesh.orientation* is
        the identity matrix and *reference_is_rotated* returns a value of *False*.
        Otherwise, *reference_is_rotated* returns a value of *True*.

        Returns
        -------
        bool
            *False* is the mesh uses the standard coordinate axes and *True* otherwise.
        """
        return not np.allclose(self.orientation, np.identity(self.dim))

    @property
    def rotation_matrix(self):
        """Alias for :py:attr:`~.BaseRegularMesh.orientation`.

        See Also
        --------
        orientation
        """
        return self.orientation  # np.array([self.axis_u, self.axis_v, self.axis_w])

    @property
    def axis_u(self):
        """Orientation of the first axis.

        .. deprecated:: 0.7.0
          `axis_u` will be removed in discretize 1.0.0. This functionality was replaced
          by the :py:attr:`~.BaseRegularMesh.orientation`.
        """
        warnings.warn(
            "The axis_u property is deprecated, please access as self.orientation[0]. "
            "This will be removed in discretize 1.0.0.",
            FutureWarning,
        )
        return self.orientation[0]

    @axis_u.setter
    def axis_u(self, value):
        warnings.warn(
            "Setting the axis_u property is deprecated, and now unchecked, please "
            "directly set the self.orientation property. This will be removed in "
            "discretize 1.0.0.",
            FutureWarning,
        )
        self.orientation[0] = value

    @property
    def axis_v(self):
        """Orientation of the second axis.

        .. deprecated:: 0.7.0
          `axis_v` will be removed in discretize 1.0.0. This functionality was replaced
          by the :py:attr:`~.BaseRegularMesh.orientation`.
        """
        warnings.warn(
            "The axis_v property is deprecated, please access as self.orientation[1]. "
            "This will be removed in discretize 1.0.0.",
            FutureWarning,
        )
        return self.orientation[1]

    @axis_v.setter
    def axis_v(self, value):
        warnings.warn(
            "Setting the axis_v property is deprecated, and now unchecked, please "
            "directly set the self.orientation property. This will be removed in "
            "discretize 1.0.0.",
            FutureWarning,
        )
        value = value / np.linalg.norm(value)
        self.orientation[1] = value

    @property
    def axis_w(self):
        """Orientation of the third axis.

        .. deprecated:: 0.7.0
          `axis_w` will be removed in discretize 1.0.0. This functionality was replaced
          by the :py:attr:`~.BaseRegularMesh.orientation`.
        """
        warnings.warn(
            "The axis_w property is deprecated, please access as self.orientation[2]. "
            "This will be removed in discretize 1.0.0.",
            FutureWarning,
        )
        return self.orientation[2]

    @axis_w.setter
    def axis_w(self, value):
        warnings.warn(
            "Setting the axis_v property is deprecated, and now unchecked, please "
            "directly set the self.orientation property. This will be removed in "
            "discretize 1.0.0.",
            FutureWarning,
        )
        value = value / np.linalg.norm(value)
        self.orientation[2] = value


class BaseRectangularMesh(BaseRegularMesh):
    """Base rectangular mesh class for the ``discretize`` package.

    The ``BaseRectangularMesh`` class acts as an extension of the
    :class:`~discretize.base.BaseRegularMesh` classes with a regular structure.
    """

    _aliases = {
        **BaseRegularMesh._aliases,
        **{
            "vnN": "shape_nodes",
            "vnEx": "shape_edges_x",
            "vnEy": "shape_edges_y",
            "vnEz": "shape_edges_z",
            "vnFx": "shape_faces_x",
            "vnFy": "shape_faces_y",
            "vnFz": "shape_faces_z",
        },
    }

    @property
    def shape_nodes(self):
        """The number of nodes along each axis direction.

        This property returns a tuple containing the number of nodes along
        each axis direction. The length of the tuple is equal to the
        dimension of the mesh; i.e. 1, 2 or 3.

        Returns
        -------
        (dim) tuple of int
            Number of nodes along each axis direction

        Notes
        -----
        Property also accessible as using the shorthand **vnN**
        """
        return tuple(x + 1 for x in self.shape_cells)

    @property
    def shape_edges_x(self):
        """Number of x-edges along each axis direction.

        This property returns a tuple containing the number of x-edges
        along each axis direction. The length of the tuple is equal to the
        dimension of the mesh; i.e. 1, 2 or 3.

        Returns
        -------
        (dim) tuple of int
            Number of x-edges along each axis direction

            - *1D mesh:* `(n_cells_x)`
            - *2D mesh:* `(n_cells_x, n_nodes_y)`
            - *3D mesh:* `(n_cells_x, n_nodes_y, n_nodes_z)`

        Notes
        -----
        Property also accessible as using the shorthand **vnEx**
        """
        return self.shape_cells[:1] + self.shape_nodes[1:]

    @property
    def shape_edges_y(self):
        """Number of y-edges along each axis direction.

        This property returns a tuple containing the number of y-edges
        along each axis direction. If `dim` is 1, there are no y-edges.

        Returns
        -------
        None or (dim) tuple of int
            Number of y-edges along each axis direction

            - *1D mesh: None*
            - *2D mesh:* `(n_nodes_x, n_cells_y)`
            - *3D mesh:* `(n_nodes_x, n_cells_y, n_nodes_z)`

        Notes
        -----
        Property also accessible as using the shorthand **vnEy**
        """
        if self.dim < 2:
            return None
        sc = self.shape_cells
        sn = self.shape_nodes
        return (sn[0], sc[1]) + sn[2:]  # conditionally added if dim == 3!

    @property
    def shape_edges_z(self):
        """Number of z-edges along each axis direction.

        This property returns a tuple containing the number of z-edges
        along each axis direction. There are only z-edges if `dim` is 3.

        Returns
        -------
        None or (dim) tuple of int
            Number of z-edges along each axis direction.

            - *1D mesh: None*
            - *2D mesh: None*
            - *3D mesh:* `(n_nodes_x, n_nodes_y, n_cells_z)`

        Notes
        -----
        Property also accessible as using the shorthand **vnEz**
        """
        if self.dim < 3:
            return None
        return self.shape_nodes[:2] + self.shape_cells[2:]

    @property
    def shape_faces_x(self):
        """Number of x-faces along each axis direction.

        This property returns a tuple containing the number of x-faces
        along each axis direction.

        Returns
        -------
        (dim) tuple of int
            Number of x-faces along each axis direction

            - *1D mesh:* `(n_nodes_x)`
            - *2D mesh:* `(n_nodes_x, n_cells_y)`
            - *3D mesh:* `(n_nodes_x, n_cells_y, n_cells_z)`

        Notes
        -----
        Property also accessible as using the shorthand **vnFx**
        """
        return self.shape_nodes[:1] + self.shape_cells[1:]

    @property
    def shape_faces_y(self):
        """Number of y-faces along each axis direction.

        This property returns a tuple containing the number of y-faces
        along each axis direction. If `dim` is 1, there are no y-edges.

        Returns
        -------
        None or (dim) tuple of int
            Number of y-faces along each axis direction

            - *1D mesh: None*
            - *2D mesh:* `(n_cells_x, n_nodes_y)`
            - *3D mesh:* `(n_cells_x, n_nodes_y, n_cells_z)`

        Notes
        -----
        Property also accessible as using the shorthand **vnFy**
        """
        if self.dim < 2:
            return None
        sc = self.shape_cells
        sn = self.shape_nodes
        return (sc[0], sn[1]) + sc[2:]

    @property
    def shape_faces_z(self):
        """Number of z-faces along each axis direction.

        This property returns a tuple containing the number of z-faces
        along each axis direction. There are only z-faces if `dim` is 3.

        Returns
        -------
        None or (dim) tuple of int
            Number of z-faces along each axis direction.

                - *1D mesh: None*
                - *2D mesh: None*
                - *3D mesh:* (n_cells_x, n_cells_y, n_nodes_z)

        Notes
        -----
        Property also accessible as using the shorthand **vnFz**
        """
        if self.dim < 3:
            return None
        return self.shape_cells[:2] + self.shape_nodes[2:]

    ##################################
    # Redo the numbering so they are dependent of the shape tuples
    # these should all inherit the parent's docstrings
    ##################################

    @property
    def n_cells(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        return int(np.prod(self.shape_cells))

    @property
    def n_nodes(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseMesh
        return int(np.prod(self.shape_nodes))

    @property
    def n_edges_x(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseRegularMesh
        return int(np.prod(self.shape_edges_x))

    @property
    def n_edges_y(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseRegularMesh
        if self.dim < 2:
            return
        return int(np.prod(self.shape_edges_y))

    @property
    def n_edges_z(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseRegularMesh
        if self.dim < 3:
            return
        return int(np.prod(self.shape_edges_z))

    @property
    def n_faces_x(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseRegularMesh
        return int(np.prod(self.shape_faces_x))

    @property
    def n_faces_y(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseRegularMesh
        if self.dim < 2:
            return
        return int(np.prod(self.shape_faces_y))

    @property
    def n_faces_z(self):  # NOQA D102
        # Documentation inherited from discretize.base.BaseRegularMesh
        if self.dim < 3:
            return
        return int(np.prod(self.shape_faces_z))

    def reshape(
        self,
        x,
        x_type="cell_centers",
        out_type="cell_centers",
        return_format="V",
        **kwargs,
    ):
        """Reshape tensor quantities.

        **Reshape** is a quick command that will do its best to reshape discrete
        quantities living on meshes than inherit the :class:`discretize.base_mesh.RectangularMesh`
        class. For example, you may have a 1D array defining a vector on mesh faces, and you would
        like to extract the x-component and reshaped it to a 3D matrix.

        Parameters
        ----------
        x : numpy.ndarray or list of numpy.ndarray
            The input quantity. , ndarray (tensor) or a list
        x_type : {'CC', 'N', 'F', 'Fx', 'Fy', 'Fz', 'E', 'Ex', 'Ey', 'Ez'}
            Defines the locations on the mesh where input parameter *x* lives.
        out_type : str
            Defines the output quantity. Choice depends on your input for *x_type*:

            - *x_type* = 'CC' ---> *out_type* = 'CC'
            - *x_type* = 'N' ---> *out_type* = 'N'
            - *x_type* = 'F' ---> *out_type* = {'F', 'Fx', 'Fy', 'Fz'}
            - *x_type* = 'E' ---> *out_type* = {'E', 'Ex', 'Ey', 'Ez'}
        return_format : str
            The dimensions of quantity being returned

            - *V:* return a vector (1D array) or a list of vectors
            - *M:* return matrix (nD array) or a list of matrices

        """
        if "xType" in kwargs:
            warnings.warn(
                "The xType keyword argument has been deprecated, please use x_type. "
                "This will be removed in discretize 1.0.0",
                FutureWarning,
            )
            x_type = kwargs["xType"]
        if "outType" in kwargs:
            warnings.warn(
                "The outType keyword argument has been deprecated, please use out_type. "
                "This will be removed in discretize 1.0.0",
                FutureWarning,
            )
            out_type = kwargs["outType"]
        if "format" in kwargs:
            warnings.warn(
                "The format keyword argument has been deprecated, please use return_format. "
                "This will be removed in discretize 1.0.0",
                FutureWarning,
            )
            return_format = kwargs["format"]

        x_type = self._parse_location_type(x_type)
        out_type = self._parse_location_type(out_type)

        allowed_x_type = [
            "cell_centers",
            "nodes",
            "faces",
            "faces_x",
            "faces_y",
            "faces_z",
            "edges",
            "edges_x",
            "edges_y",
            "edges_z",
        ]
        if not (isinstance(x, list) or isinstance(x, np.ndarray)):
            raise Exception("x must be either a list or a ndarray")
        if x_type not in allowed_x_type:
            raise Exception(
                "x_type must be either '" + "', '".join(allowed_x_type) + "'"
            )
        if out_type not in allowed_x_type:
            raise Exception(
                "out_type must be either '" + "', '".join(allowed_x_type) + "'"
            )
        if return_format not in ["M", "V"]:
            raise Exception("return_format must be either 'M' or 'V'")
        if out_type[: len(x_type)] != x_type:
            raise Exception("You cannot change types when reshaping.")
        if x_type not in out_type:
            raise Exception("You cannot change type of components.")

        if isinstance(x, list):
            for i, xi in enumerate(x):
                if not isinstance(x, np.ndarray):
                    raise Exception("x[{0:d}] must be a numpy array".format(i))
                if xi.size != x[0].size:
                    raise Exception("Number of elements in list must not change.")

            x_array = np.ones((x.size, len(x)))
            # Unwrap it and put it in a np array
            for i, xi in enumerate(x):
                x_array[:, i] = mkvc(xi)
            x = x_array

        if not isinstance(x, np.ndarray):
            raise Exception("x must be a numpy array")

        x = x[:]  # make a copy.
        x_type_is_FE_xyz = (
            len(x_type) > 1
            and x_type[0] in ["f", "e"]
            and x_type[-1] in ["x", "y", "z"]
        )

        def outKernal(xx, nn):
            """Return xx as either a matrix (shape == nn) or a vector."""
            if return_format == "M":
                return xx.reshape(nn, order="F")
            elif return_format == "V":
                return mkvc(xx)

        def switchKernal(xx):
            """Switch over the different options."""
            if x_type in ["cell_centers", "nodes"]:
                nn = self.shape_cells if x_type == "cell_centers" else self.shape_nodes
                if xx.size != np.prod(nn):
                    raise Exception("Number of elements must not change.")
                return outKernal(xx, nn)
            elif x_type in ["faces", "edges"]:
                # This will only deal with components of fields,
                # not full 'F' or 'E'
                xx = mkvc(xx)  # unwrap it in case it is a matrix
                if x_type == "faces":
                    nn = (self.nFx, self.nFy, self.nFz)[: self.dim]
                else:
                    nn = (self.nEx, self.nEy, self.nEz)[: self.dim]
                nn = np.r_[0, nn]

                nx = [0, 0, 0]
                nx[0] = self.shape_faces_x if x_type == "faces" else self.shape_edges_x
                nx[1] = self.shape_faces_y if x_type == "faces" else self.shape_edges_y
                nx[2] = self.shape_faces_z if x_type == "faces" else self.shape_edges_z

                for dim, dimName in enumerate(["x", "y", "z"]):
                    if dimName in out_type:
                        if self.dim <= dim:
                            raise Exception(
                                "Dimensions of mesh not great enough for "
                                "{}_{}".format(x_type, dimName)
                            )
                        if xx.size != np.sum(nn):
                            raise Exception("Vector is not the right size.")
                        start = np.sum(nn[: dim + 1])
                        end = np.sum(nn[: dim + 2])
                        return outKernal(xx[start:end], nx[dim])

            elif x_type_is_FE_xyz:
                # This will deal with partial components (x, y or z)
                # lying on edges or faces
                if "x" in x_type:
                    nn = self.shape_faces_x if "f" in x_type else self.shape_edges_x
                elif "y" in x_type:
                    nn = self.shape_faces_y if "f" in x_type else self.shape_edges_y
                elif "z" in x_type:
                    nn = self.shape_faces_z if "f" in x_type else self.shape_edges_z
                if xx.size != np.prod(nn):
                    raise Exception(
                        f"Vector is not the right size. Expected {np.prod(nn)}, got {xx.size}"
                    )
                return outKernal(xx, nn)

        # Check if we are dealing with a vector quantity
        isVectorQuantity = len(x.shape) == 2 and x.shape[1] == self.dim

        if out_type in ["faces", "edges"]:
            if isVectorQuantity:
                raise Exception("Not sure what to do with a vector vector quantity..")
            outTypeCopy = out_type
            out = ()
            for dirName in ["x", "y", "z"][: self.dim]:
                out_type = outTypeCopy + "_" + dirName
                out += (switchKernal(x),)
            return out
        elif isVectorQuantity:
            out = ()
            for ii in range(x.shape[1]):
                out += (switchKernal(x[:, ii]),)
            return out
        else:
            return switchKernal(x)

    # DEPRECATED
    r = deprecate_method("reshape", "r", removal_version="1.0.0", future_warn=True)

    @property
    def nCx(self):
        """Number of cells in the x direction.

        Returns
        -------
        int

        .. deprecated:: 0.5.0
          `nCx` will be removed in discretize 1.0.0, it is replaced by
          `mesh.shape_cells[0]` to reduce namespace clutter.
        """
        warnings.warn(
            "nCx has been deprecated, please access as mesh.shape_cells[0]",
            FutureWarning,
        )
        return self.shape_cells[0]

    @property
    def nCy(self):
        """Number of cells in the y direction.

        Returns
        -------
        int or None
            None if dim < 2

        .. deprecated:: 0.5.0
          `nCy` will be removed in discretize 1.0.0, it is replaced by
          `mesh.shape_cells[1]` to reduce namespace clutter.
        """
        warnings.warn(
            "nCy has been deprecated, please access as mesh.shape_cells[1]",
            FutureWarning,
        )
        if self.dim < 2:
            return None
        return self.shape_cells[1]

    @property
    def nCz(self):
        """Number of cells in the z direction.

        Returns
        -------
        int or None
            None if dim < 3

        .. deprecated:: 0.5.0
          `nCz` will be removed in discretize 1.0.0, it is replaced by
          `mesh.shape_cells[2]` to reduce namespace clutter.
        """
        warnings.warn(
            "nCz has been deprecated, please access as mesh.shape_cells[2]",
            FutureWarning,
        )
        if self.dim < 3:
            return None
        return self.shape_cells[2]

    @property
    def nNx(self):
        """Number of nodes in the x-direction.

        Returns
        -------
        int

        .. deprecated:: 0.5.0
          `nNx` will be removed in discretize 1.0.0, it is replaced by
          `mesh.shape_nodes[0]` to reduce namespace clutter.
        """
        warnings.warn(
            "nNx has been deprecated, please access as mesh.shape_nodes[0]",
            FutureWarning,
        )
        return self.shape_nodes[0]

    @property
    def nNy(self):
        """Number of nodes in the y-direction.

        Returns
        -------
        int or None
            None if dim < 2

        .. deprecated:: 0.5.0
          `nNy` will be removed in discretize 1.0.0, it is replaced by
          `mesh.shape_nodes[1]` to reduce namespace clutter.
        """
        warnings.warn(
            "nNy has been deprecated, please access as mesh.shape_nodes[1]",
            FutureWarning,
        )
        if self.dim < 2:
            return None
        return self.shape_nodes[1]

    @property
    def nNz(self):
        """Number of nodes in the z-direction.

        Returns
        -------
        int or None
            None if dim < 3

        .. deprecated:: 0.5.0
          `nNz` will be removed in discretize 1.0.0, it is replaced by
          `mesh.shape_nodes[2]` to reduce namespace clutter.
        """
        warnings.warn(
            "nNz has been deprecated, please access as mesh.shape_nodes[2]",
            FutureWarning,
        )
        if self.dim < 3:
            return None
        return self.shape_nodes[2]
