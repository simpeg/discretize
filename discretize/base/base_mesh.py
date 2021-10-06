"""
Base classes for all meshes supported in ``discretize``
"""

import numpy as np
import os
import json

from discretize.utils import mkvc, Identity
from discretize.utils.code_utils import deprecate_property, deprecate_method
import warnings


class BaseMesh:
    """
    Base mesh class for the ``discretize`` package

    The ``BaseMesh`` class does all the basic counting and organizing
    you wouldn't want to do manually. ``BaseMesh`` is a class that should
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

    _REGISTRY = {}
    _aliases = {
        "nC": "n_cells",
        "nN": "n_nodes",
        "nEx": "n_edges_x",
        "nEy": "n_edges_y",
        "nEz": "n_edges_z",
        "nE": "n_edges",
        "vnE": "n_edges_per_direction",
        "nFx": "n_faces_x",
        "nFy": "n_faces_y",
        "nFz": "n_faces_z",
        "nF": "n_faces",
        "vnF": "n_faces_per_direction",
        "vnC": "shape_cells",
        "serialize": "to_dict",
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

    def __getattr__(self, name):
        if name == "_aliases":
            raise AttributeError
        name = self._aliases.get(name, name)
        return super().__getattribute__(name)

    @property
    def origin(self):
        """Origin or 'anchor point' of the mesh

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
        """The number of cells in each coordinate direction.

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
        """Rotation matrix defining mesh axes relative to Cartesian

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
        """Coordinate reference system

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

    def to_dict(self):
        """Representation of the mesh's attributes as a dictionary

        The dictionary representation of the mesh class necessary to reconstruct the
        object. This is useful for serialization. All of the attributes returned in this
        dictionary will be JSON serializable.

        The mesh class is also stored in the dictionary as strings under the
        `__module__` and `__class__` keys.

        Returns
        -------
        dict
            Dictionary of {attribute: value} for the attributes of this mesh.
        """
        cls = type(self)
        out = {
            "__module__": cls.__module__,
            "__class__": cls.__name__,
        }
        for item in self._items:
            attr = getattr(self, item, None)
            if attr is not None:
                if isinstance(attr, np.ndarray):
                    attr = attr.tolist()
                elif isinstance(attr, tuple):
                    # change to a list and make sure inner items are not numpy arrays
                    attr = list(attr)
                    for i, thing in enumerate(attr):
                        if isinstance(thing, np.ndarray):
                            attr[i] = thing.tolist()
                out[item] = attr
        return out

    def equals(self, other_mesh):
        """Compares current mesh with another mesh to determine if they are identical

        This method compares all the properties of the current mesh to *other_mesh*
        and determines if both meshes are identical. If so, this function returns
        a boolean value of *True* . Otherwise, the function returns *False* .

        Parameters
        ----------
        other_mesh : discretize.base.BaseMesh
            An instance of any discretize mesh class.

        Returns
        -------
        bool
            *True* if meshes are identical and *False* otherwise.
        """
        if type(self) != type(other_mesh):
            return False
        for item in self._items:
            my_attr = getattr(self, item, None)
            other_mesh_attr = getattr(other_mesh, item, None)
            if isinstance(my_attr, np.ndarray):
                is_equal = np.allclose(my_attr, other_mesh_attr, rtol=0, atol=0)
            elif isinstance(my_attr, tuple):
                is_equal = len(my_attr) == len(other_mesh_attr)
                if is_equal:
                    for thing1, thing2 in zip(my_attr, other_mesh_attr):
                        if isinstance(thing1, np.ndarray):
                            is_equal = np.allclose(thing1, thing2, rtol=0, atol=0)
                        else:
                            try:
                                is_equal = thing1 == thing2
                            except Exception:
                                is_equal = False
                    if not is_equal:
                        return is_equal
            else:
                try:
                    is_equal = my_attr == other_mesh_attr
                except Exception:
                    is_equal = False
            if not is_equal:
                return is_equal
        return is_equal

    def serialize(self):
        """
        An alias for :py:meth:`~.BaseMesh.to_dict`
        """
        return self.to_dict()

    @classmethod
    def deserialize(cls, items, **kwargs):
        """Create this mesh from a dictionary of attributes

        Parameters
        ----------
        items : dict
            dictionary of {attribute : value} pairs that will be passed to this class's
            initialization method as keyword arguments.
        **kwargs
            This is used to catch (and ignore) keyword arguments that used to be used.
        """
        items.pop("__module__", None)
        items.pop("__class__", None)
        return cls(**items)

    @property
    def x0(self):
        """
        An alias for the :py:attr:`~.BaseMesh.origin`
        """
        return self.origin

    @x0.setter
    def x0(self, val):
        self.origin = val

    @property
    def dim(self):
        """The dimension of the mesh (1, 2, or 3).

        The dimension is an integer denoting whether the mesh
        is 1D, 2D or 3D.

        Returns
        -------
        int
            Dimension of the mesh; i.e. 1, 2 or 3
        """
        return len(self.shape_cells)

    @property
    def n_cells(self):
        """Total number of cells in the mesh.

        Returns
        -------
        int
            Number of cells in the mesh

        Notes
        -----
        Property also accessible as using the shorthand **nC**
        """
        return int(np.prod(self.shape_cells))

    def __len__(self):
        """Total number of cells in the mesh.

        Essentially this is an alias for :py:attr:`~.BaseMesh.n_cells`.

        Returns
        -------
        int
            Number of cells in the mesh
        """
        return self.n_cells

    @property
    def n_nodes(self):
        """Total number of nodes in the mesh

        Returns
        -------
        int
            Number of nodes in the mesh

        Notes
        -----
        Property also accessible as using the shorthand **nN**
        """
        return int(np.prod([x + 1 for x in self.shape_cells]))

    @property
    def n_edges_x(self):
        """Number of x-edges in the mesh

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
        """Number of y-edges in the mesh

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
        """Number of z-edges in the mesh

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
        """The number of edges in each direction

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
    def n_edges(self):
        """Total number of edges in the mesh

        Returns
        -------
        int
            Total number of edges in the mesh

        Notes
        -----
        Property also accessible as using the shorthand **nE**
        """
        n = self.n_edges_x
        if self.dim > 1:
            n += self.n_edges_y
        if self.dim > 2:
            n += self.n_edges_z
        return n

    @property
    def n_faces_x(self):
        """Number of x-faces in the mesh

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
        """Number of y-faces in the mesh

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
        """Number of z-faces in the mesh

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
        """The number of faces in each axis direction

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
    def n_faces(self):
        """Total number of faces in the mesh

        Returns
        -------
        int
            Total number of faces in the mesh

        Notes
        -----
        Property also accessible as using the shorthand **nF**
        """
        n = self.n_faces_x
        if self.dim > 1:
            n += self.n_faces_y
        if self.dim > 2:
            n += self.n_faces_z
        return n

    @property
    def face_normals(self):
        """Unit normal vectors for all mesh faces

        The unit normal vector defines the direction that
        is perpendicular to a surface. Calling
        *face_normals* returns a numpy.ndarray containing
        the unit normal vectors for all faces in the mesh.
        For a 3D mesh, the array would have shape (n_faces, dim).
        The rows of the output are organized by x-faces,
        then y-faces, then z-faces vectors.

        Returns
        -------
        (n_faces, dim) numpy.ndarray of float
            Unit normal vectors for all mesh faces
        """
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
    def edge_tangents(self):
        """Unit tangent vectors for all mesh edges

        For a given edge, the unit tangent vector defines the
        path direction one would take if traveling along that edge.
        Calling *edge_tangents* returns a numpy.ndarray containing
        the unit tangent vectors for all edges in the mesh.
        For a 3D mesh, the array would have shape (n_edges, dim).
        The rows of the output are organized by x-edges,
        then y-edges, then z-edges vectors.

        Returns
        -------
        (n_edges, dim) numpy.ndarray of float
            Unit tangent vectors for all mesh edges
        """
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

    def project_face_vector(self, face_vectors):
        """Project vectors onto the faces of the mesh.

        Consider a numpy array *face_vectors* whose rows provide
        a vector for each face in the mesh. For each face,
        *project_face_vector* computes the dot product between
        a vector and the corresponding face's unit normal vector.
        That is, *project_face_vector* projects the vectors
        in *face_vectors* to the faces of the mesh.

        Parameters
        ----------
        face_vectors : (n_faces, dim) numpy.ndarray
            Numpy array containing the vectors that will be projected to the mesh faces

        Returns
        -------
        (n_faces) numpy.ndarray of float
            Dot product between each vector and the unit normal vector of the corresponding face
        """
        if not isinstance(face_vectors, np.ndarray):
            raise Exception("face_vectors must be an ndarray")
        if not (
            len(face_vectors.shape) == 2
            and face_vectors.shape[0] == self.n_faces
            and face_vectors.shape[1] == self.dim
        ):
            raise Exception("face_vectors must be an ndarray of shape (n_faces, dim)")
        return np.sum(face_vectors * self.face_normals, 1)

    def project_edge_vector(self, edge_vectors):
        """Project vectors to the edges of the mesh.

        Consider a numpy array *edge_vectors* whose rows provide
        a vector for each edge in the mesh. For each edge,
        *project_edge_vector* computes the dot product between
        a vector and the corresponding edge's unit tangent vector.
        That is, *project_edge_vector* projects the vectors
        in *edge_vectors* to the edges of the mesh.

        Parameters
        ----------
        edge_vectors : (n_edges, dim) numpy.ndarray
            Numpy array containing the vectors that will be projected to the mesh edges

        Returns
        -------
        (n_edges) numpy.ndarray of float
            Dot product between each vector and the unit tangent vector of the corresponding edge
        """
        if not isinstance(edge_vectors, np.ndarray):
            raise Exception("edge_vectors must be an ndarray")
        if not (
            len(edge_vectors.shape) == 2
            and edge_vectors.shape[0] == self.n_edges
            and edge_vectors.shape[1] == self.dim
        ):
            raise Exception("edge_vectors must be an ndarray of shape (nE, dim)")
        return np.sum(edge_vectors * self.edge_tangents, 1)

    def save(self, file_name="mesh.json", verbose=False, **kwargs):
        """Save the mesh to json

        This method is used to save a mesh by writing
        its properties to a .json file. To load a mesh you have
        previously saved, see :py:func:`~discretize.utils.load_mesh`.

        Parameters
        ----------
        file_name : str, optional
            File name for saving the mesh properties
        verbose : bool, optional
            If *True*, the path of the json file is printed
        """

        if "filename" in kwargs:
            file_name = kwargs["filename"]
            warnings.warn(
                "The filename keyword argument has been deprecated, please use file_name. "
                "This will be removed in discretize 1.0.0",
                DeprecationWarning,
            )
        f = os.path.abspath(file_name)  # make sure we are working with abs path
        with open(f, "w") as outfile:
            json.dump(self.to_dict(), outfile)

        if verbose:
            print("Saved {}".format(f))

        return f

    def copy(self):
        """Make a copy of the current mesh

        Returns
        -------
        type(mesh)
            A copy of this mesh.
        """
        cls = type(self)
        items = self.to_dict()
        items.pop("__module__", None)
        items.pop("__class__", None)
        return cls(**items)

    @property
    def reference_is_rotated(self):
        """Indicates whether mesh uses standard coordinate axes

        The standard basis vectors defining the x, y, and z axes of
        a mesh are :math:`(1,0,0)`, :math:`(0,1,0)` and :math:`(0,0,1)`,
        respectively. However, the :py:attr:`~BaseMesh.orientation` property
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
        """
        Alias for :py:attr:`~.BaseMesh.orientation`
        """
        return self.orientation  # np.array([self.axis_u, self.axis_v, self.axis_w])

    def _parse_location_type(self, location_type):
        if len(location_type) == 0:
            return location_type
        elif location_type[0] == "F":
            if len(location_type) > 1:
                return "faces_" + location_type[-1]
            else:
                return "faces"
        elif location_type[0] == "E":
            if len(location_type) > 1:
                return "edges_" + location_type[-1]
            else:
                return "edges"
        elif location_type[0] == "N":
            return "nodes"
        elif location_type[0] == "C":
            if len(location_type) > 2:
                return "cell_centers_" + location_type[-1]
            else:
                return "cell_centers"
        else:
            return location_type

    def validate(self):
        """Return the validation state of the mesh

        This mesh is valid immediately upon initialization

        Returns
        -------
        bool : True
        """
        return True

    # DEPRECATED
    normals = deprecate_property("face_normals", "normals", removal_version="1.0.0")
    tangents = deprecate_property("edge_tangents", "tangents", removal_version="1.0.0")
    projectEdgeVector = deprecate_method(
        "project_edge_vector", "projectEdgeVector", removal_version="1.0.0"
    )
    projectFaceVector = deprecate_method(
        "project_face_vector", "projectFaceVector", removal_version="1.0.0"
    )

    @property
    def axis_u(self):
        """
        .. deprecated:: 0.7.0
          `axis_u` will be removed in discretize 1.0.0. This functionality was replaced
          by the :py:attr:`~.BaseMesh.orientation`.
        """
        warnings.warn(
            "The axis_u property is deprecated, please access as self.orientation[0]. "
            "This will be removed in discretize 1.0.0.",
            DeprecationWarning,
        )
        return self.orientation[0]

    @axis_u.setter
    def axis_u(self, value):
        warnings.warn(
            "Setting the axis_u property is deprecated, and now unchecked, please "
            "directly set the self.orientation property. This will be removed in "
            "discretize 1.0.0.",
            DeprecationWarning,
        )
        self.orientation[0] = value

    @property
    def axis_v(self):
        """
        .. deprecated:: 0.7.0
          `axis_v` will be removed in discretize 1.0.0. This functionality was replaced
          by the :py:attr:`~.BaseMesh.orientation`.
        """
        warnings.warn(
            "The axis_v property is deprecated, please access as self.orientation[1]. "
            "This will be removed in discretize 1.0.0.",
            DeprecationWarning,
        )
        return self.orientation[1]

    @axis_v.setter
    def axis_v(self, value):
        warnings.warn(
            "Setting the axis_v property is deprecated, and now unchecked, please "
            "directly set the self.orientation property. This will be removed in "
            "discretize 1.0.0.",
            DeprecationWarning,
        )
        value = value / np.linalg.norm(value)
        self.orientation[1] = value

    @property
    def axis_w(self):
        """
        .. deprecated:: 0.7.0
          `axis_w` will be removed in discretize 1.0.0. This functionality was replaced
          by the :py:attr:`~.BaseMesh.orientation`.
        """
        warnings.warn(
            "The axis_w property is deprecated, please access as self.orientation[2]. "
            "This will be removed in discretize 1.0.0.",
            DeprecationWarning,
        )
        return self.orientation[2]

    @axis_w.setter
    def axis_w(self, value):
        warnings.warn(
            "Setting the axis_v property is deprecated, and now unchecked, please "
            "directly set the self.orientation property. This will be removed in "
            "discretize 1.0.0.",
            DeprecationWarning,
        )
        value = value / np.linalg.norm(value)
        self.orientation[2] = value


class BaseRectangularMesh(BaseMesh):
    """
    Base rectangular mesh class for the ``discretize`` package.

    The ``BaseRectangularMesh`` class acts as an extension of the
    :class:`~discretize.base.BaseMesh` classes with a regular structure.
    """

    _aliases = {
        **BaseMesh._aliases,
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
        """Returns the number of nodes along each axis direction

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
        """Number of x-edges along each axis direction

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
        """Number of y-edges along each axis direction

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
        """Number of z-edges along each axis direction

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
        """Number of x-faces along each axis direction

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
        """Number of y-faces along each axis direction

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
        """Number of z-faces along each axis direction

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
    def n_cells(self):
        return int(np.prod(self.shape_cells))

    @property
    def n_nodes(self):
        return int(np.prod(self.shape_nodes))

    @property
    def n_edges_x(self):
        return int(np.prod(self.shape_edges_x))

    @property
    def n_edges_y(self):
        if self.dim < 2:
            return
        return int(np.prod(self.shape_edges_y))

    @property
    def n_edges_z(self):
        if self.dim < 3:
            return
        return int(np.prod(self.shape_edges_z))

    @property
    def n_faces_x(self):
        return int(np.prod(self.shape_faces_x))

    @property
    def n_faces_y(self):
        if self.dim < 2:
            return
        return int(np.prod(self.shape_faces_y))

    @property
    def n_faces_z(self):
        if self.dim < 3:
            return
        return int(np.prod(self.shape_faces_z))

    def reshape(
        self, x, x_type="cell_centers", out_type="cell_centers", format="V", **kwargs
    ):
        """General reshape method for tensor quantities

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
        format : str
            The dimensions of quantity being returned

                - *V:* return a vector (1D array) or a list of vectors
                - *M:* return matrix (nD array) or a list of matrices

        """
        if "xType" in kwargs:
            warnings.warn(
                "The xType keyword argument has been deprecated, please use x_type. "
                "This will be removed in discretize 1.0.0",
                DeprecationWarning,
            )
            x_type = kwargs["xType"]
        if "outType" in kwargs:
            warnings.warn(
                "The outType keyword argument has been deprecated, please use out_type. "
                "This will be removed in discretize 1.0.0",
                DeprecationWarning,
            )
            out_type = kwargs["outType"]

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
        if format not in ["M", "V"]:
            raise Exception("format must be either 'M' or 'V'")
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
            """Returns xx as either a matrix (shape == nn) or a vector."""
            if format == "M":
                return xx.reshape(nn, order="F")
            elif format == "V":
                return mkvc(xx)

        def switchKernal(xx):
            """Switches over the different options."""
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
            for ii, dirName in enumerate(["x", "y", "z"][: self.dim]):
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
    r = deprecate_method("reshape", "r", removal_version="1.0.0")

    @property
    def nCx(self):
        """Number of cells in the x direction

        Returns
        -------
        int

        .. deprecated:: 0.5.0
          `nCx` will be removed in discretize 1.0.0, it is replaced by
          `mesh.shape_cells[0]` to reduce namespace clutter.
        """

        warnings.warn(
            "nCx has been deprecated, please access as mesh.shape_cells[0]",
            DeprecationWarning,
        )
        return self.shape_cells[0]

    @property
    def nCy(self):
        """Number of cells in the y direction

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
            DeprecationWarning,
        )
        if self.dim < 2:
            return None
        return self.shape_cells[1]

    @property
    def nCz(self):
        """Number of cells in the z direction

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
            DeprecationWarning,
        )
        if self.dim < 3:
            return None
        return self.shape_cells[2]

    @property
    def nNx(self):
        """Number of nodes in the x-direction

        Returns
        -------
        int

        .. deprecated:: 0.5.0
          `nNx` will be removed in discretize 1.0.0, it is replaced by
          `mesh.shape_nodes[0]` to reduce namespace clutter.
        """

        warnings.warn(
            "nNx has been deprecated, please access as mesh.shape_nodes[0]",
            DeprecationWarning,
        )
        return self.shape_nodes[0]

    @property
    def nNy(self):
        """Number of nodes in the y-direction

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
            DeprecationWarning,
        )
        if self.dim < 2:
            return None
        return self.shape_nodes[1]

    @property
    def nNz(self):
        """Number of nodes in the z-direction

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
            DeprecationWarning,
        )
        if self.dim < 3:
            return None
        return self.shape_nodes[2]
