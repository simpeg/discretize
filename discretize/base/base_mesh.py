"""
Base classes for all discretize meshes
"""

import numpy as np
import os
import json

from discretize.utils import mkvc, Identity
from discretize.utils.code_utils import deprecate_property, deprecate_method
import warnings


class BaseMesh:
    """
    BaseMesh does all the counting you don't want to do.
    BaseMesh should be inherited by meshes with a regular structure.
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
        """Origin of the mesh"""
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
        """The number of cells in each direction

        Returns
        -------
        tuple of ints

        Notes
        -----
        Also accessible as `vnC`.
        """
        return self._shape_cells

    @property
    def orientation(self):
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
        "The type of coordinate reference frame. Can take on the values "
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

    def equals(self, other):
        if type(self) != type(other):
            return False
        for item in self._items:
            my_attr = getattr(self, item, None)
            other_attr = getattr(other, item, None)
            if isinstance(my_attr, np.ndarray):
                is_equal = np.allclose(my_attr, other_attr, rtol=0, atol=0)
            elif isinstance(my_attr, tuple):
                is_equal = len(my_attr) == len(other_attr)
                if is_equal:
                    for thing1, thing2 in zip(my_attr, other_attr):
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
                    is_equal = my_attr == other_attr
                except Exception:
                    is_equal = False
            if not is_equal:
                return is_equal
        return is_equal

    def serialize(self):
        return self.to_dict()

    @classmethod
    def deserialize(cls, items, **kwargs):
        items.pop("__module__", None)
        items.pop("__class__", None)
        return cls(**items)

    @property
    def x0(self):
        return self.origin

    @x0.setter
    def x0(self, val):
        self.origin = val

    @property
    def dim(self):
        """The dimension of the mesh (1, 2, or 3).

        Returns
        -------
        int
            dimension of the mesh
        """
        return len(self.shape_cells)

    @property
    def n_cells(self):
        """Total number of cells in the mesh.

        Returns
        -------
        int
            number of cells in the mesh

        Notes
        -----
        Also accessible as `nC`.

        Examples
        --------
        >>> import discretize
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> mesh = discretize.TensorMesh([np.ones(n) for n in [2,3]])
        >>> mesh.plot_grid(centers=True, show_it=True)
        >>> print(mesh.n_cells)
        """
        return int(np.prod(self.shape_cells))

    def __len__(self):
        """The number of cells on the mesh."""
        return self.n_cells

    @property
    def n_nodes(self):
        """Total number of nodes

        Returns
        -------
        int
            number of nodes in the mesh

        Notes
        -----
        Also accessible as `nN`.

        Examples
        --------
        >>> import discretize
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> mesh = discretize.TensorMesh([np.ones(n) for n in [2,3]])
        >>> mesh.plot_grid(nodes=True, show_it=True)
        >>> print(mesh.n_nodes)
        """
        return int(np.prod([x + 1 for x in self.shape_cells]))

    @property
    def n_edges_x(self):
        """Number of x-edges

        Returns
        -------
        int

        Notes
        -----
        Also accessible as `nEx`.

        """
        return int(np.prod([x + y for x, y in zip(self.shape_cells, (0, 1, 1))]))

    @property
    def n_edges_y(self):
        """Number of y-edges

        Returns
        -------
        int

        Notes
        -----
        Also accessible as `nEy`.

        """
        if self.dim < 2:
            return None
        return int(np.prod([x + y for x, y in zip(self.shape_cells, (1, 0, 1))]))

    @property
    def n_edges_z(self):
        """Number of z-edges

        Returns
        -------
        int

        Notes
        -----
        Also accessible as `nEz`.

        """
        if self.dim < 3:
            return None
        return int(np.prod([x + y for x, y in zip(self.shape_cells, (1, 1, 0))]))

    @property
    def n_edges_per_direction(self):
        """The number of edges in each direction

        Returns
        -------
        n_edges_per_direction : tuple
            [n_edges_x, n_edges_y, n_edges_z], (dim, )

        Notes
        -----
        Also accessible as `vnE`.

        Examples
        --------
        >>> import discretize
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> M = discretize.TensorMesh([np.ones(n) for n in [2,3]])
        >>> M.plot_grid(edges=True, show_it=True)
        """
        return tuple(
            x for x in [self.n_edges_x, self.n_edges_y, self.n_edges_z] if x is not None
        )

    @property
    def n_edges(self):
        """Total number of edges.

        Returns
        -------
        int
            sum([n_edges_x, n_edges_y, n_edges_z])

        Notes
        -----
        Also accessible as `nE`.

        """
        n = self.n_edges_x
        if self.dim > 1:
            n += self.n_edges_y
        if self.dim > 2:
            n += self.n_edges_z
        return n

    @property
    def n_faces_x(self):
        """Number of x-faces

        Returns
        -------
        int

        Notes
        -----
        Also accessible as `nFx`.
        """
        return int(np.prod([x + y for x, y in zip(self.shape_cells, (1, 0, 0))]))

    @property
    def n_faces_y(self):
        """Number of y-faces

        Returns
        -------
        int

        Notes
        -----
        Also accessible as `nFy`.
        """
        if self.dim < 2:
            return None
        return int(np.prod([x + y for x, y in zip(self.shape_cells, (0, 1, 0))]))

    @property
    def n_faces_z(self):
        """Number of z-faces

        Returns
        -------
        int

        Notes
        -----
        Also accessible as `nFz`.
        """
        if self.dim < 3:
            return None
        return int(np.prod([x + y for x, y in zip(self.shape_cells, (0, 0, 1))]))

    @property
    def n_faces_per_direction(self):
        """The number of faces in each direction

        Returns
        -------
        n_faces_per_direction : tuple
            [n_faces_x, n_faces_y, n_faces_z], (dim, )

        Notes
        -----
        Also accessible as `vnF`.

        Examples
        --------
        >>> import discretize
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> M = discretize.TensorMesh([np.ones(n) for n in [2,3]])
        >>> M.plot_grid(faces=True, show_it=True)
        """
        return tuple(
            x for x in [self.n_faces_x, self.n_faces_y, self.n_faces_z] if x is not None
        )

    @property
    def n_faces(self):
        """Total number of faces.

        Returns
        -------
        int
            sum([n_faces_x, n_faces_y, n_faces_z])

        Notes
        -----
        Also accessible as `nF`.

        """
        n = self.n_faces_x
        if self.dim > 1:
            n += self.n_faces_y
        if self.dim > 2:
            n += self.n_faces_z
        return n

    @property
    def face_normals(self):
        """Face Normals

        Returns
        -------
        numpy.ndarray
            normals, (n_faces, dim)
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
        """Edge Tangents

        Returns
        -------
        numpy.ndarray
            normals, (n_edges, dim)
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

    def project_face_vector(self, face_vector):
        """Project vectors onto the faces of the mesh.

        Given a vector, face_vector, in cartesian coordinates, this will project
        it onto the mesh using the normals

        Parameters
        ----------
        face_vector : numpy.ndarray
            face vector with shape (n_faces, dim)

        Returns
        -------
        numpy.ndarray
            projected face vector, (n_faces, )

        """
        if not isinstance(face_vector, np.ndarray):
            raise Exception("face_vector must be an ndarray")
        if not (
            len(face_vector.shape) == 2
            and face_vector.shape[0] == self.n_faces
            and face_vector.shape[1] == self.dim
        ):
            raise Exception("face_vector must be an ndarray of shape (n_faces x dim)")
        return np.sum(face_vector * self.face_normals, 1)

    def project_edge_vector(self, edge_vector):
        """Project vectors onto the edges of the mesh

        Given a vector, edge_vector, in cartesian coordinates, this will project
        it onto the mesh using the tangents

        Parameters
        ----------
        edge_vector : numpy.ndarray
            edge vector with shape (n_edges, dim)

        Returns
        -------
        numpy.ndarray
            projected edge vector, (n_edges, )

        """
        if not isinstance(edge_vector, np.ndarray):
            raise Exception("edge_vector must be an ndarray")
        if not (
            len(edge_vector.shape) == 2
            and edge_vector.shape[0] == self.n_edges
            and edge_vector.shape[1] == self.dim
        ):
            raise Exception("edge_vector must be an ndarray of shape (nE x dim)")
        return np.sum(edge_vector * self.edge_tangents, 1)

    def save(self, file_name="mesh.json", verbose=False, **kwargs):
        """
        Save the mesh to json
        :param str file: file_name for saving the casing properties
        :param str directory: working directory for saving the file
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
        """
        Make a copy of the current mesh
        """
        cls = type(self)
        items = self.to_dict()
        items.pop("__module__", None)
        items.pop("__class__", None)
        return cls(**items)

    @property
    def reference_is_rotated(self):
        """True if the axes are rotated from the traditional <X,Y,Z> system
        with vectors of :math:`(1,0,0)`, :math:`(0,1,0)`, and :math:`(0,0,1)`
        """
        return not np.allclose(self.orientation, np.identity(self.dim))

    @property
    def rotation_matrix(self):
        """Builds a rotation matrix to transform coordinates from their coordinate
        system into a conventional cartesian system. This is built off of the
        three `axis_u`, `axis_v`, and `axis_w` properties; these mapping
        coordinates use the letters U, V, and W (the three letters preceding X,
        Y, and Z in the alphabet) to define the projection of the X, Y, and Z
        durections. These UVW vectors describe the placement and transformation
        of the mesh's coordinate sytem assuming at most 3 directions.

        Why would you want to use these UVW mapping vectors the this
        `rotation_matrix` property? They allow us to define the relationship
        between local and global coordinate systems and provide a tool for
        switching between the two while still maintaing the connectivity of the
        mesh's cells. For a visual example of this, please see the figure in the
        docs for the :class:`~discretize.mixins.vtk_mod.InterfaceVTK`.
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
        """Every object will be valid upon initialization"""
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
          `axis_u` will be removed in discretize 1.0.0, it is replaced by
          `mesh.orientation` for better mesh orientation validation.

        See Also
        --------
        orientation
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
          `axis_v` will be removed in discretize 1.0.0, it is replaced by
          `mesh.orientation` for better mesh orientation validation.

        See Also
        --------
        orientation
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
          `axis_w` will be removed in discretize 1.0.0, it is replaced by
          `mesh.orientation` for better mesh orientation validation.

        See Also
        --------
        orientation
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


BaseMesh.__module__ = "discretize.base"


class BaseRectangularMesh(BaseMesh):
    """
    BaseRectangularMesh
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
        """Number of nodes in each direction

        Returns
        -------
        tuple of int

        Notes
        -----
        Also accessible as `vnN`.
        """
        return tuple(x + 1 for x in self.shape_cells)

    @property
    def shape_edges_x(self):
        """Number of x-edges in each direction

        Returns
        -------
        tuple of int
            (nx_cells, ny_nodes, nz_nodes)

        Notes
        -----
        Also accessible as `vnEx`.
        """
        return self.shape_cells[:1] + self.shape_nodes[1:]

    @property
    def shape_edges_y(self):
        """Number of y-edges in each direction

        Returns
        -------
        tuple of int or None
            (nx_nodes, ny_cells, nz_nodes), None if dim < 2

        Notes
        -----
        Also accessible as `vnEy`.
        """
        if self.dim < 2:
            return None
        sc = self.shape_cells
        sn = self.shape_nodes
        return (sn[0], sc[1]) + sn[2:]  # conditionally added if dim == 3!

    @property
    def shape_edges_z(self):
        """Number of z-edges in each direction

        Returns
        -------
        tuple of int or None
            (nx_nodes, ny_nodes, nz_cells), None if dim < 3

        Notes
        -----
        Also accessible as `vnEz`.
        """
        if self.dim < 3:
            return None
        return self.shape_nodes[:2] + self.shape_cells[2:]

    @property
    def shape_faces_x(self):
        """Number of x-faces in each direction

        Returns
        -------
        tuple of int
            (nx_nodes, ny_cells, nz_cells)

        Notes
        -----
        Also accessible as `vnFx`.
        """
        return self.shape_nodes[:1] + self.shape_cells[1:]

    @property
    def shape_faces_y(self):
        """Number of y-faces in each direction

        Returns
        -------
        tuple of int or None
            (nx_cells, ny_nodes, nz_cells), None if dim < 2

        Notes
        -----
        Also accessible as `vnFy`.
        """
        if self.dim < 2:
            return None
        sc = self.shape_cells
        sn = self.shape_nodes
        return (sc[0], sn[1]) + sc[2:]

    @property
    def shape_faces_z(self):
        """Number of z-faces in each direction

        Returns
        -------
        tuple of int or None
            (nx_cells, ny_cells, nz_nodes), None if dim < 3

        Notes
        -----
        Also accessible as `vnFz`.
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
        """A quick reshape command that will do the best it
        can at giving you what you want.

        For example, you have a face variable, and you want the x
        component of it reshaped to a 3D matrix.

        `reshape` can fulfil your dreams::

            mesh.reshape(V, 'F', 'Fx', 'M')
                         |   |     |    |
                         |   |     |    {
                         |   |     |      How: 'M' or ['V'] for a matrix
                         |   |     |      (ndgrid style) or a vector (n x dim)
                         |   |     |    }
                         |   |     {
                         |   |       What you want: ['CC'], 'N',
                         |   |                       'F', 'Fx', 'Fy', 'Fz',
                         |   |                       'E', 'Ex', 'Ey', or 'Ez'
                         |   |     }
                         |   {
                         |     What is it: ['CC'], 'N',
                         |                  'F', 'Fx', 'Fy', 'Fz',
                         |                  'E', 'Ex', 'Ey', or 'Ez'
                         |   }
                         {
                           The input: as a list or ndarray
                         }


        For example::

            # Separates each component of the Ex grid into 3 matrices
            Xex, Yex, Zex = r(mesh.gridEx, 'Ex', 'Ex', 'M')

            # Given an edge vector, return just the x edges as a vector
            XedgeVector = r(edgeVector, 'E', 'Ex', 'V')

            # Separates each component of the edgeVector into 3 vectors
            eX, eY, eZ = r(edgeVector, 'E', 'E', 'V')
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


BaseRectangularMesh.__module__ = "discretize.base"
