"""Module for the base ``discretize`` mesh."""
import numpy as np
import warnings
import os
import json
from scipy.spatial import KDTree
from discretize.utils.code_utils import (
    deprecate_property,
    deprecate_method,
    as_array_n_by_dim,
)


class BaseMesh:
    """
    Base mesh class for the ``discretize`` package.

    This class contains the basic structure of properties and methods
    that should be supported on all discretize meshes.
    """

    _aliases = {
        "nC": "n_cells",
        "nN": "n_nodes",
        "nE": "n_edges",
        "nF": "n_faces",
        "serialize": "to_dict",
        "gridCC": "cell_centers",
        "gridN": "nodes",
        "aveF2CC": "average_face_to_cell",
        "aveF2CCV": "average_face_to_cell_vector",
        "aveCC2F": "average_cell_to_face",
        "aveCCV2F": "average_cell_vector_to_face",
        "aveE2CC": "average_edge_to_cell",
        "aveE2CCV": "average_edge_to_cell_vector",
        "aveN2CC": "average_node_to_cell",
        "aveN2E": "average_node_to_edge",
        "aveN2F": "average_node_to_face",
    }

    def __getattr__(self, name):
        """Reimplement get attribute to allow for aliases."""
        if name == "_aliases":
            raise AttributeError
        name = self._aliases.get(name, name)
        return super().__getattribute__(name)

    def to_dict(self):
        """Represent the mesh's attributes as a dictionary.

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
        """Compare the current mesh with another mesh to determine if they are identical.

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
        """Represent the mesh's attributes as a dictionary.

        An alias for :py:meth:`~.BaseMesh.to_dict`

        See Also
        --------
        to_dict
        """
        return self.to_dict()

    @classmethod
    def deserialize(cls, items, **kwargs):
        """Create this mesh from a dictionary of attributes.

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

    def save(self, file_name="mesh.json", verbose=False, **kwargs):
        """Save the mesh to json.

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
                FutureWarning,
            )
        f = os.path.abspath(file_name)  # make sure we are working with abs path
        with open(f, "w") as outfile:
            json.dump(self.to_dict(), outfile)

        if verbose:
            print("Saved {}".format(f))

        return f

    def copy(self):
        """Make a copy of the current mesh.

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

    def validate(self):
        """Return the validation state of the mesh.

        This mesh is valid immediately upon initialization

        Returns
        -------
        bool : True
        """
        return True

    # Counting dim, n_cells, n_nodes, n_edges, n_faces
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
        raise NotImplementedError(f"dim not implemented for {type(self)}")

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
        raise NotImplementedError(f"n_cells not implemented for {type(self)}")

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
        """Total number of nodes in the mesh.

        Returns
        -------
        int
            Number of nodes in the mesh

        Notes
        -----
        Property also accessible as using the shorthand **nN**
        """
        raise NotImplementedError(f"n_nodes not implemented for {type(self)}")

    @property
    def n_edges(self):
        """Total number of edges in the mesh.

        Returns
        -------
        int
            Total number of edges in the mesh

        Notes
        -----
        Property also accessible as using the shorthand **nE**
        """
        raise NotImplementedError(f"n_edges not implemented for {type(self)}")

    @property
    def n_faces(self):
        """Total number of faces in the mesh.

        Returns
        -------
        int
            Total number of faces in the mesh

        Notes
        -----
        Property also accessible as using the shorthand **nF**
        """
        raise NotImplementedError(f"n_faces not implemented for {type(self)}")

    # grid locations
    @property
    def cell_centers(self):
        """Return gridded cell center locations.

        This property returns a numpy array of shape (n_cells, dim)
        containing gridded cell center locations for all cells in the
        mesh. The cells are ordered along the x, then y, then z directions.

        Returns
        -------
        (n_cells, dim) numpy.ndarray of float
            Gridded cell center locations

        Examples
        --------
        The following is a 1D example.

        >>> from discretize import TensorMesh
        >>> hx = np.ones(5)
        >>> mesh_1D = TensorMesh([hx], '0')
        >>> mesh_1D.cell_centers
        array([0.5, 1.5, 2.5, 3.5, 4.5])

        The following is a 3D example.

        >>> hx, hy, hz = np.ones(2), 2*np.ones(2), 3*np.ones(2)
        >>> mesh_3D = TensorMesh([hx, hy, hz], '000')
        >>> mesh_3D.cell_centers
        array([[0.5, 1. , 1.5],
               [1.5, 1. , 1.5],
               [0.5, 3. , 1.5],
               [1.5, 3. , 1.5],
               [0.5, 1. , 4.5],
               [1.5, 1. , 4.5],
               [0.5, 3. , 4.5],
               [1.5, 3. , 4.5]])

        """
        raise NotImplementedError(f"cell_centers not implemented for {type(self)}")

    @property
    def nodes(self):
        """Return gridded node locations.

        This property returns a numpy array of shape (n_nodes, dim)
        containing gridded node locations for all nodes in the
        mesh. The nodes are ordered along the x, then y, then z directions.

        Returns
        -------
        (n_nodes, dim) numpy.ndarray of float
            Gridded node locations

        Examples
        --------
        The following is a 1D example.

        >>> from discretize import TensorMesh
        >>> hx = np.ones(5)
        >>> mesh_1D = TensorMesh([hx], '0')
        >>> mesh_1D.nodes
        array([0., 1., 2., 3., 4., 5.])

        The following is a 3D example.

        >>> hx, hy, hz = np.ones(2), 2*np.ones(2), 3*np.ones(2)
        >>> mesh_3D = TensorMesh([hx, hy, hz], '000')
        >>> mesh_3D.nodes
        array([[0., 0., 0.],
               [1., 0., 0.],
               [2., 0., 0.],
               [0., 2., 0.],
               [1., 2., 0.],
               [2., 2., 0.],
               [0., 4., 0.],
               [1., 4., 0.],
               [2., 4., 0.],
               [0., 0., 3.],
               [1., 0., 3.],
               [2., 0., 3.],
               [0., 2., 3.],
               [1., 2., 3.],
               [2., 2., 3.],
               [0., 4., 3.],
               [1., 4., 3.],
               [2., 4., 3.],
               [0., 0., 6.],
               [1., 0., 6.],
               [2., 0., 6.],
               [0., 2., 6.],
               [1., 2., 6.],
               [2., 2., 6.],
               [0., 4., 6.],
               [1., 4., 6.],
               [2., 4., 6.]])

        """
        raise NotImplementedError(f"nodes not implemented for {type(self)}")

    @property
    def boundary_nodes(self):
        """Boundary node locations.

        This property returns the locations of the nodes on
        the boundary of the mesh as a numpy array. The shape
        of the numpy array is the number of boundary nodes by
        the dimension of the mesh.

        Returns
        -------
        (n_boundary_nodes, dim) numpy.ndarray of float
            Boundary node locations
        """
        raise NotImplementedError(f"boundary_nodes not implemented for {type(self)}")

    @property
    def faces(self):
        """Gridded face locations.

        This property returns a numpy array of shape (n_faces, dim)
        containing gridded locations for all faces in the mesh.

        For structued meshes, the first row corresponds to the bottom-front-leftmost x-face.
        The output array returns the x-faces, then the y-faces, then
        the z-faces; i.e. *mesh.faces* is equivalent to *np.r_[mesh.faces_x, mesh.faces_y, mesh.face_z]* .
        For each face type, the locations are ordered along the x, then y, then z directions.

        Returns
        -------
        (n_faces, dim) numpy.ndarray of float
            Gridded face locations

        Examples
        --------
        Here, we provide an example of a minimally staggered curvilinear mesh.
        In this case, the x and y-faces have normal vectors that are
        primarily along the x and y-directions, respectively.

        >>> from discretize import CurvilinearMesh
        >>> from discretize.utils import example_curvilinear_grid, mkvc
        >>> from matplotlib import pyplot as plt

        >>> x, y = example_curvilinear_grid([10, 10], "rotate")
        >>> mesh1 = CurvilinearMesh([x, y])
        >>> faces = mesh1.faces
        >>> x_faces = faces[:mesh1.n_faces_x]
        >>> y_faces = faces[mesh1.n_faces_x:]

        >>> fig1 = plt.figure(figsize=(5, 5))
        >>> ax1 = fig1.add_subplot(111)
        >>> mesh1.plot_grid(ax=ax1)
        >>> ax1.scatter(x_faces[:, 0], x_faces[:, 1], 30, 'r')
        >>> ax1.scatter(y_faces[:, 0], y_faces[:, 1], 30, 'g')
        >>> ax1.legend(['Mesh', 'X-faces', 'Y-faces'], fontsize=16)
        >>> plt.plot()

        Here, we provide an example of a highly irregular curvilinear mesh.
        In this case, the y-faces are not defined by normal vectors along
        a particular direction.

        >>> x, y = example_curvilinear_grid([10, 10], "sphere")
        >>> mesh2 = CurvilinearMesh([x, y])
        >>> faces = mesh2.faces
        >>> x_faces = faces[:mesh2.n_faces_x]
        >>> y_faces = faces[mesh2.n_faces_x:]

        >>> fig2 = plt.figure(figsize=(5, 5))
        >>> ax2 = fig2.add_subplot(111)
        >>> mesh2.plot_grid(ax=ax2)
        >>> ax2.scatter(x_faces[:, 0], x_faces[:, 1], 30, 'r')
        >>> ax2.scatter(y_faces[:, 0], y_faces[:, 1], 30, 'g')
        >>> ax2.legend(['Mesh', 'X-faces', 'Y-faces'], fontsize=16)
        >>> plt.plot()

        """
        raise NotImplementedError(f"faces not implemented for {type(self)}")

    @property
    def boundary_faces(self):
        """Boundary face locations.

        This property returns the locations of the faces on
        the boundary of the mesh as a numpy array. The shape
        of the numpy array is the number of boundary faces by
        the dimension of the mesh.

        Returns
        -------
        (n_boundary_faces, dim) numpy.ndarray of float
            Boundary faces locations
        """
        raise NotImplementedError(f"boundary_faces not implemented for {type(self)}")

    @property
    def edges(self):
        """Gridded edge locations.

        This property returns a numpy array of shape (n_edges, dim)
        containing gridded locations for all edges in the mesh.

        For structured meshes, the first row corresponds to the bottom-front-leftmost x-edge.
        The output array returns the x-edges, then the y-edges, then
        the z-edges; i.e. *mesh.edges* is equivalent to *np.r_[mesh.edges_x, mesh.edges_y, mesh.edges_z]* .
        For each edge type, the locations are ordered along the x, then y, then z directions.

        Returns
        -------
        (n_edges, dim) numpy.ndarray of float
            Gridded edge locations

        Examples
        --------
        Here, we provide an example of a minimally staggered curvilinear mesh.
        In this case, the x and y-edges have normal vectors that are
        primarily along the x and y-directions, respectively.

        >>> from discretize import CurvilinearMesh
        >>> from discretize.utils import example_curvilinear_grid, mkvc
        >>> from matplotlib import pyplot as plt

        >>> x, y = example_curvilinear_grid([10, 10], "rotate")
        >>> mesh1 = CurvilinearMesh([x, y])
        >>> edges = mesh1.edges
        >>> x_edges = edges[:mesh1.n_edges_x]
        >>> y_edges = edges[mesh1.n_edges_x:]

        >>> fig1 = plt.figure(figsize=(5, 5))
        >>> ax1 = fig1.add_subplot(111)
        >>> mesh1.plot_grid(ax=ax1)
        >>> ax1.scatter(x_edges[:, 0], x_edges[:, 1], 30, 'r')
        >>> ax1.scatter(y_edges[:, 0], y_edges[:, 1], 30, 'g')
        >>> ax1.legend(['Mesh', 'X-edges', 'Y-edges'], fontsize=16)
        >>> plt.plot()

        Here, we provide an example of a highly irregular curvilinear mesh.
        In this case, the y-edges are not defined by normal vectors along
        a particular direction.

        >>> x, y = example_curvilinear_grid([10, 10], "sphere")
        >>> mesh2 = CurvilinearMesh([x, y])
        >>> edges = mesh2.edges
        >>> x_edges = edges[:mesh2.n_edges_x]
        >>> y_edges = edges[mesh2.n_edges_x:]

        >>> fig2 = plt.figure(figsize=(5, 5))
        >>> ax2 = fig2.add_subplot(111)
        >>> mesh2.plot_grid(ax=ax2)
        >>> ax2.scatter(x_edges[:, 0], x_edges[:, 1], 30, 'r')
        >>> ax2.scatter(y_edges[:, 0], y_edges[:, 1], 30, 'g')
        >>> ax2.legend(['Mesh', 'X-edges', 'Y-edges'], fontsize=16)
        >>> plt.show()

        """
        raise NotImplementedError(f"edges not implemented for {type(self)}")

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
        raise NotImplementedError(f"boundary_edges not implemented for {type(self)}")

    # unit directions

    @property
    def face_normals(self):
        """Unit normal vectors for all mesh faces.

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
        raise NotImplementedError(f"face_normals not implemented for {type(self)}")

    @property
    def edge_tangents(self):
        """Unit tangent vectors for all mesh edges.

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
        raise NotImplementedError(f"edge_tangents not implemented for {type(self)}")

    @property
    def boundary_face_outward_normals(self):
        """Outward normal vectors of boundary faces.

        This property returns the outward normal vectors of faces
        the boundary of the mesh as a numpy array. The shape
        of the numpy array is the number of boundary faces by
        the dimension of the mesh.

        Returns
        -------
        (n_boundary_faces, dim) numpy.ndarray of float
            Outward normal vectors of boundary faces
        """
        raise NotImplementedError(
            f"boundary_face_outward_normals not implemented for {type(self)}"
        )

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

    # Mesh properties
    @property
    def cell_volumes(self):
        """Return cell volumes.

        Calling this property will compute and return a 1D array
        containing the volumes of mesh cells.

        Returns
        -------
        (n_cells) numpy.ndarray
            The quantity returned depends on the dimensions of the mesh:

                - *1D:* Returns the cell widths
                - *2D:* Returns the cell areas
                - *3D:* Returns the cell volumes
        """
        raise NotImplementedError(f"cell_volumes not implemented for {type(self)}")

    @property
    def face_areas(self):
        """Return areas of all faces in the mesh.

        Calling this property will compute and return the areas of all
        faces as a 1D numpy array. For structured meshes, the returned quantity is
        ordered x-face areas, then y-face areas, then z-face areas.

        Returns
        -------
        (n_faces) numpy.ndarray
            The length of the quantity returned depends on the dimensions of the mesh:

            - *1D:* returns the x-face areas
            - *2D:* returns the x-face and y-face areas in order; i.e. y-edge
              and x-edge lengths, respectively
            - *3D:* returns the x, y and z-face areas in order
        """
        raise NotImplementedError(f"face_areas not implemented for {type(self)}")

    @property
    def edge_lengths(self):
        """Return lengths of all edges in the mesh.

        Calling this property will compute and return the lengths of all
        edges in the mesh. For structured meshes, the returned quantity is ordered
        x-edge lengths, then y-edge lengths, then z-edge lengths.

        Returns
        -------
        (n_edges) numpy.ndarray
            The length of the quantity returned depends on the dimensions of the mesh:

            - *1D:* returns the x-edge lengths
            - *2D:* returns the x-edge and y-edge lengths in order
            - *3D:* returns the x, y and z-edge lengths in order
        """
        raise NotImplementedError(f"edge_lengths not implemented for {type(self)}")

    # Differential Operators
    @property
    def face_divergence(self):
        r"""Face divergence operator (faces to cell-centres).

        This property constructs the 2nd order numerical divergence operator
        that maps from faces to cell centers. The operator is a sparse matrix
        :math:`\mathbf{D_f}` that can be applied as a matrix-vector product to
        a discrete vector :math:`\mathbf{u}` that lives on mesh faces; i.e.::

            div_u = Df @ u

        Once constructed, the operator is stored permanently as a property of the mesh.
        *See notes for additional details.*

        Returns
        -------
        (n_cells, n_faces) scipy.sparse.csr_matrix
            The numerical divergence operator from faces to cell centers

        Notes
        -----
        In continuous space, the divergence operator is defined as:

        .. math::
            \phi = \nabla \cdot \vec{u} = \frac{\partial u_x}{\partial x}
            + \frac{\partial u_y}{\partial y} + \frac{\partial u_z}{\partial z}

        Where :math:`\mathbf{u}` is the discrete representation of the continuous variable
        :math:`\vec{u}` on cell faces and :math:`\boldsymbol{\phi}` is the discrete
        representation of :math:`\phi` at cell centers, **face_divergence** constructs a
        discrete linear operator :math:`\mathbf{D_f}` such that:

        .. math::
            \boldsymbol{\phi} = \mathbf{D_f \, u}

        For each cell, the computation of the face divergence can be expressed
        according to the integral form below. For cell :math:`i` whose corresponding
        faces are indexed as a subset :math:`K` from the set of all mesh faces:

        .. math::
            \phi_i = \frac{1}{V_i} \sum_{k \in K} A_k \, \vec{u}_k \cdot \hat{n}_k

        where :math:`V_i` is the volume of cell :math:`i`, :math:`A_k` is
        the surface area of face *k*, :math:`\vec{u}_k` is the value of
        :math:`\vec{u}` on face *k*, and :math:`\hat{n}_k`
        represents the outward normal vector of face *k* for cell *i*.


        Examples
        --------
        Below, we demonstrate 1) how to apply the face divergence operator to
        a discrete vector and 2) the mapping of the face divergence operator and
        its sparsity. Our example is carried out on a 2D mesh but it can
        be done equivalently for a 3D mesh.

        We start by importing the necessary packages and modules.

        >>> from discretize import TensorMesh
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> import matplotlib as mpl

        Define a 2D mesh

        >>> h = np.ones(20)
        >>> mesh = TensorMesh([h, h], "CC")

        Create a discrete vector on mesh faces

        >>> faces_x = mesh.faces_x
        >>> faces_y = mesh.faces_y
        >>> ux = (faces_x[:, 0] / np.sqrt(np.sum(faces_x ** 2, axis=1))) * np.exp(
        ...     -(faces_x[:, 0] ** 2 + faces_x[:, 1] ** 2) / 6 ** 2
        ... )
        >>> uy = (faces_y[:, 1] / np.sqrt(np.sum(faces_y ** 2, axis=1))) * np.exp(
        ...     -(faces_y[:, 0] ** 2 + faces_y[:, 1] ** 2) / 6 ** 2
        ... )
        >>> u = np.r_[ux, uy]

        Construct the divergence operator and apply to face-vector

        >>> Df = mesh.face_divergence
        >>> div_u = Df @ u

        Plot the original face-vector and its divergence

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(13, 6))
            >>> ax1 = fig.add_subplot(121)
            >>> mesh.plot_image(
            ...     u, ax=ax1, v_type="F", view="vec", stream_opts={"color": "w", "density": 1.0}
            ... )
            >>> ax1.set_title("Vector at cell faces", fontsize=14)
            >>> ax2 = fig.add_subplot(122)
            >>> mesh.plot_image(div_u, ax=ax2)
            >>> ax2.set_yticks([])
            >>> ax2.set_ylabel("")
            >>> ax2.set_title("Divergence at cell centers", fontsize=14)
            >>> plt.show()

        The discrete divergence operator is a sparse matrix that maps
        from faces to cell centers. To demonstrate this, we construct
        a small 2D mesh. We then show the ordering of the elements in
        the original discrete quantity :math:`\mathbf{u}` and its
        discrete divergence :math:`\boldsymbol{\phi}` as well as a
        spy plot.

        .. collapse:: Expand to show scripting for plot

            >>> mesh = TensorMesh([[(1, 6)], [(1, 3)]])
            >>> fig = plt.figure(figsize=(10, 10))
            >>> ax1 = fig.add_subplot(211)
            >>> mesh.plot_grid(ax=ax1)
            >>> ax1.plot(
            ...     mesh.cell_centers[:, 0], mesh.cell_centers[:, 1], "ro", markersize=8
            ... )
            >>> for ii, loc in zip(range(mesh.nC), mesh.cell_centers):
            ...     ax1.text(loc[0]+0.05, loc[1]+0.02, "{0:d}".format(ii), color="r")
            >>> ax1.plot(
            ...     mesh.faces_x[:, 0], mesh.faces_x[:, 1], "g>", markersize=8
            ... )
            >>> for ii, loc in zip(range(mesh.nFx), mesh.faces_x):
            ...     ax1.text(loc[0]+0.05, loc[1]+0.02, "{0:d}".format(ii), color="g")
            >>> ax1.plot(
            ...     mesh.faces_y[:, 0], mesh.faces_y[:, 1], "g^", markersize=8
            ... )
            >>> for ii, loc in zip(range(mesh.nFy), mesh.faces_y):
            ...     ax1.text(loc[0] + 0.05, loc[1] + 0.1, "{0:d}".format((ii + mesh.nFx)), color="g")

            >>> ax1.set_xticks([])
            >>> ax1.set_yticks([])
            >>> ax1.spines['bottom'].set_color('white')
            >>> ax1.spines['top'].set_color('white')
            >>> ax1.spines['left'].set_color('white')
            >>> ax1.spines['right'].set_color('white')
            >>> ax1.set_xlabel('X', fontsize=16, labelpad=-5)
            >>> ax1.set_ylabel('Y', fontsize=16, labelpad=-15)
            >>> ax1.set_title("Mapping of Face Divergence", fontsize=14, pad=15)
            >>> ax1.legend(
            ...     ['Mesh', r'$\mathbf{\phi}$ (centers)', r'$\mathbf{u}$ (faces)'],
            ...     loc='upper right', fontsize=14
            ... )
            >>> ax2 = fig.add_subplot(212)
            >>> ax2.spy(mesh.face_divergence)
            >>> ax2.set_title("Spy Plot", fontsize=14, pad=5)
            >>> ax2.set_ylabel("Cell Index", fontsize=12)
            >>> ax2.set_xlabel("Face Index", fontsize=12)
            >>> plt.show()
        """
        raise NotImplementedError(f"face_divergence not implemented for {type(self)}")

    @property
    def nodal_gradient(self):
        r"""Nodal gradient operator (nodes to edges).

        This property constructs the 2nd order numerical gradient operator
        that maps from nodes to edges. The operator is a sparse matrix
        :math:`\mathbf{G_n}` that can be applied as a matrix-vector product
        to a discrete scalar quantity :math:`\boldsymbol{\phi}` that
        lives on the nodes, i.e.::

            grad_phi = Gn @ phi

        Once constructed, the operator is stored permanently as a property of the mesh.

        Returns
        -------
        (n_edges, n_nodes) scipy.sparse.csr_matrix
            The numerical gradient operator from nodes to edges

        Notes
        -----
        In continuous space, the gradient operator is defined as:

        .. math::
            \vec{u} = \nabla \phi = \frac{\partial \phi}{\partial x}\hat{x}
            + \frac{\partial \phi}{\partial y}\hat{y}
            + \frac{\partial \phi}{\partial z}\hat{z}

        Where :math:`\boldsymbol{\phi}` is the discrete representation of the continuous variable
        :math:`\phi` on the nodes and :math:`\mathbf{u}` is the discrete
        representation of :math:`\vec{u}` on the edges, **nodal_gradient** constructs a
        discrete linear operator :math:`\mathbf{G_n}` such that:

        .. math::
            \mathbf{u} = \mathbf{G_n} \, \boldsymbol{\phi}

        The Cartesian components of :math:`\vec{u}` are defined on their corresponding
        edges (x, y or z) as follows; e.g. the x-component of the gradient is defined
        on x-edges. For edge :math:`i` which defines a straight path
        of length :math:`h_i` between adjacent nodes :math:`n_1` and :math:`n_2`:

        .. math::
            u_i = \frac{\phi_{n_2} - \phi_{n_1}}{h_i}

        Note that :math:`u_i \in \mathbf{u}` may correspond to a value on an
        x, y or z edge. See the example below.

        Examples
        --------
        Below, we demonstrate 1) how to apply the nodal gradient operator to
        a discrete scalar quantity, and 2) the mapping of the nodal gradient
        operator and its sparsity. Our example is carried out on a 2D mesh
        but it can be done equivalently for a 3D mesh.

        We start by importing the necessary packages and modules.

        >>> from discretize import TensorMesh
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> import matplotlib as mpl

        For a discrete scalar quantity defined on the nodes, we take the
        gradient by constructing the gradient operator and multiplying
        as a matrix-vector product.

        Create a uniform grid

        >>> h = np.ones(20)
        >>> mesh = TensorMesh([h, h], "CC")

        Create a discrete scalar on nodes

        >>> nodes = mesh.nodes
        >>> phi = np.exp(-(nodes[:, 0] ** 2 + nodes[:, 1] ** 2) / 4 ** 2)

        Construct the gradient operator and apply to vector

        >>> Gn = mesh.nodal_gradient
        >>> grad_phi = Gn @ phi

        Plot the original function and the gradient

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(13, 6))
            >>> ax1 = fig.add_subplot(121)
            >>> mesh.plot_image(phi, v_type="N", ax=ax1)
            >>> ax1.set_title("Scalar at nodes", fontsize=14)
            >>> ax2 = fig.add_subplot(122)
            >>> mesh.plot_image(
            ...     grad_phi, ax=ax2, v_type="E", view="vec",
            ...     stream_opts={"color": "w", "density": 1.0}
            ... )
            >>> ax2.set_yticks([])
            >>> ax2.set_ylabel("")
            >>> ax2.set_title("Gradient at edges", fontsize=14)
            >>> plt.show()

        The nodal gradient operator is a sparse matrix that maps
        from nodes to edges. To demonstrate this, we construct
        a small 2D mesh. We then show the ordering of the elements in
        the original discrete quantity :math:`\boldsymbol{\phi}` and its
        discrete gradient as well as a spy plot.

        .. collapse:: Expand to show scripting for plot

            >>> mesh = TensorMesh([[(1, 3)], [(1, 6)]])
            >>> fig = plt.figure(figsize=(12, 10))
            >>> ax1 = fig.add_subplot(121)
            >>> mesh.plot_grid(ax=ax1)
            >>> ax1.set_title("Mapping of Gradient Operator", fontsize=14, pad=15)
            >>> ax1.plot(mesh.nodes[:, 0], mesh.nodes[:, 1], "ro", markersize=8)
            >>> for ii, loc in zip(range(mesh.nN), mesh.nodes):
            >>>     ax1.text(loc[0] + 0.05, loc[1] + 0.02, "{0:d}".format(ii), color="r")

            >>> ax1.plot(mesh.edges_x[:, 0], mesh.edges_x[:, 1], "g>", markersize=8)
            >>> for ii, loc in zip(range(mesh.nEx), mesh.edges_x):
            >>>     ax1.text(loc[0] + 0.05, loc[1] + 0.02, "{0:d}".format(ii), color="g")

            >>> ax1.plot(mesh.edges_y[:, 0], mesh.edges_y[:, 1], "g^", markersize=8)
            >>> for ii, loc in zip(range(mesh.nEy), mesh.edges_y):
            >>>     ax1.text(loc[0] + 0.05, loc[1] + 0.02, "{0:d}".format((ii + mesh.nEx)), color="g")

            >>> ax1.set_xticks([])
            >>> ax1.set_yticks([])
            >>> ax1.spines['bottom'].set_color('white')
            >>> ax1.spines['top'].set_color('white')
            >>> ax1.spines['left'].set_color('white')
            >>> ax1.spines['right'].set_color('white')
            >>> ax1.set_xlabel('X', fontsize=16, labelpad=-5)
            >>> ax1.set_ylabel('Y', fontsize=16, labelpad=-15)
            >>> ax1.legend(
            >>>     ['Mesh', r'$\mathbf{\phi}$ (nodes)', r'$\mathbf{u}$ (edges)'],
            >>>     loc='upper right', fontsize=14
            >>> )
            >>> ax2 = fig.add_subplot(122)
            >>> ax2.spy(mesh.nodal_gradient)
            >>> ax2.set_title("Spy Plot", fontsize=14, pad=5)
            >>> ax2.set_ylabel("Edge Index", fontsize=12)
            >>> ax2.set_xlabel("Node Index", fontsize=12)
            >>> plt.plot()
        """
        raise NotImplementedError(f"nodal_gradient not implemented for {type(self)}")

    @property
    def edge_curl(self):
        r"""Edge curl operator (edges to faces).

        This property constructs the 2nd order numerical curl operator
        that maps from edges to faces. The operator is a sparse matrix
        :math:`\mathbf{C_e}` that can be applied as a matrix-vector product
        to a discrete vector quantity **u** that lives
        on the edges; i.e.::

            curl_u = Ce @ u

        Once constructed, the operator is stored permanently as a property of the mesh.

        Returns
        -------
        (n_faces, n_edges) scipy.sparse.csr_matrix
            The numerical curl operator from edges to faces

        Notes
        -----
        In continuous space, the curl operator is defined as:

        .. math::
            \vec{w} = \nabla \times \vec{u} =
            \begin{vmatrix}
            \hat{x} & \hat{y} & \hat{z} \\
            \partial_x & \partial_y & \partial_z \\
            u_x & u_y & u_z
            \end{vmatrix}

        Where :math:`\mathbf{u}` is the discrete representation of the continuous variable
        :math:`\vec{u}` on cell edges and :math:`\mathbf{w}` is the discrete
        representation of the curl on the faces, **edge_curl** constructs a
        discrete linear operator :math:`\mathbf{C_e}` such that:

        .. math::
            \mathbf{w} = \mathbf{C_e \, u}

        The computation of the curl on mesh faces can be expressed
        according to the integral form below. For face :math:`i` bordered by
        a set of edges indexed by subset :math:`K`:

        .. math::
            w_i = \frac{1}{A_i} \sum_{k \in K} \vec{u}_k \cdot \vec{\ell}_k

        where :math:`A_i` is the surface area of face *i*,
        :math:`u_k` is the value of :math:`\vec{u}` on face *k*,
        and \vec{\ell}_k is the path along edge *k*.

        Examples
        --------
        Below, we demonstrate the mapping and sparsity of the edge curl
        for a 3D tensor mesh. We choose a the index for a single face,
        and illustrate which edges are used to compute the curl on that
        face.

        >>> from discretize import TensorMesh
        >>> from discretize.utils import mkvc
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import matplotlib as mpl
        >>> import mpl_toolkits.mplot3d as mp3d
        >>> mpl.rcParams.update({'font.size': 14})

        Create a simple tensor mesh, and grab the **edge_curl** operator:

        >>> mesh = TensorMesh([[(1, 2)], [(1, 2)], [(1, 2)]])
        >>> Ce = mesh.edge_curl

        Then we choose a *face* for illustration purposes:

        >>> face_ind = 2  # Index of a face in the mesh (could be x, y or z)
        >>> edge_ind = np.where(
        ...     np.sum((mesh.edges-mesh.faces[face_ind, :])**2, axis=1) <= 0.5 + 1e-6
        ... )[0]

        >>> face = mesh.faces[face_ind, :]
        >>> face_norm = mesh.face_normals[face_ind, :]
        >>> edges = mesh.edges[edge_ind, :]
        >>> edge_tan = mesh.edge_tangents[edge_ind, :]
        >>> node = np.min(edges, axis=0)

        >>> min_edges = np.min(edges, axis=0)
        >>> max_edges = np.max(edges, axis=0)
        >>> if face_norm[0] == 1:
        ...     k = (edges[:, 1] == min_edges[1]) | (edges[:, 2] == max_edges[2])
        ...     poly = node + np.c_[np.r_[0, 0, 0, 0], np.r_[0, 1, 1, 0], np.r_[0, 0, 1, 1]]
        ...     ds = [0.07, -0.07, -0.07]
        ... elif face_norm[1] == 1:
        ...     k = (edges[:, 0] == max_edges[0]) | (edges[:, 2] == min_edges[2])
        ...     poly = node + np.c_[np.r_[0, 1, 1, 0], np.r_[0, 0, 0, 0], np.r_[0, 0, 1, 1]]
        ...     ds = [0.07, -0.09, -0.07]
        ... elif face_norm[2] == 1:
        ...     k = (edges[:, 0] == min_edges[0]) | (edges[:, 1] == max_edges[1])
        ...     poly = node + np.c_[np.r_[0, 1, 1, 0], np.r_[0, 0, 1, 1], np.r_[0, 0, 0, 0]]
        ...     ds = [0.07, -0.09, -0.07]
        >>> edge_tan[k, :] *= -1

        Plot the curve and its mapping for a single face.

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(10, 15))
            >>> ax1 = fig.add_axes([0, 0.35, 1, 0.6], projection='3d', elev=25, azim=-60)
            >>> mesh.plot_grid(ax=ax1)
            >>> ax1.plot(
            ...     mesh.edges[edge_ind, 0], mesh.edges[edge_ind, 1], mesh.edges[edge_ind, 2],
            ...     "go", markersize=10
            ... )
            >>> ax1.plot(
            ...     mesh.faces[face_ind, 0], mesh.faces[face_ind, 1], mesh.faces[face_ind, 2],
            ...     "rs", markersize=10
            ... )
            >>> poly = mp3d.art3d.Poly3DCollection(
            ...     [poly], alpha=0.1, facecolor='r', linewidth=None
            ... )
            >>> ax1.add_collection(poly)
            >>> ax1.quiver(
            ...     edges[:, 0], edges[:, 1], edges[:, 2],
            ...     0.5*edge_tan[:, 0], 0.5*edge_tan[:, 1], 0.5*edge_tan[:, 2],
            ...     edgecolor='g', pivot='middle', linewidth=4, arrow_length_ratio=0.25
            ... )
            >>> ax1.text(face[0]+ds[0], face[1]+ds[1], face[2]+ds[2], "{0:d}".format(face_ind), color="r")
            >>> for ii, loc in zip(range(len(edge_ind)), edges):
            ...     ax1.text(loc[0]+ds[0], loc[1]+ds[1], loc[2]+ds[2], "{0:d}".format(edge_ind[ii]), color="g")
            >>> ax1.legend(
            ...     ['Mesh', r'$\mathbf{u}$ (edges)', r'$\mathbf{w}$ (face)'],
            ...     loc='upper right', fontsize=14
            ... )

            Manually make axis properties invisible

            >>> ax1.set_xticks([])
            >>> ax1.set_yticks([])
            >>> ax1.set_zticks([])
            >>> ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            >>> ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            >>> ax1.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
            >>> ax1.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            >>> ax1.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            >>> ax1.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
            >>> ax1.set_xlabel('X', labelpad=-15, fontsize=16)
            >>> ax1.set_ylabel('Y', labelpad=-20, fontsize=16)
            >>> ax1.set_zlabel('Z', labelpad=-20, fontsize=16)
            >>> ax1.set_title("Mapping for a Single Face", fontsize=16, pad=-15)

            >>> ax2 = fig.add_axes([0.05, 0.05, 0.9, 0.3])
            >>> ax2.spy(Ce)
            >>> ax2.set_title("Spy Plot", fontsize=16, pad=5)
            >>> ax2.set_ylabel("Face Index", fontsize=12)
            >>> ax2.set_xlabel("Edge Index", fontsize=12)
            >>> plt.show()
        """
        raise NotImplementedError(f"edge_curl not implemented for {type(self)}")

    @property
    def boundary_face_scalar_integral(self):
        r"""Represent the operation of integrating a scalar function on the boundary.

        This matrix represents the boundary surface integral of a scalar function
        multiplied with a finite volume test function on the mesh.

        Returns
        -------
        (n_faces, n_boundary_faces) scipy.sparse.csr_matrix

        Notes
        -----
        The integral we are representing on the boundary of the mesh is

        .. math:: \int_{\Omega} u\vec{w} \cdot \hat{n} \partial \Omega

        In discrete form this is:

        .. math:: w^T * P @ u_b

        where `w` is defined on all faces, and `u_b` is defined on boundary faces.
        """
        raise NotImplementedError(
            f"boundary_face_scalar_integral not implemented for {type(self)}"
        )

    @property
    def boundary_edge_vector_integral(self):
        r"""Represent the operation of integrating a vector function on the boundary.

        This matrix represents the boundary surface integral of a vector function
        multiplied with a finite volume test function on the mesh.

        In 1D and 2D, the operation assumes that the right array contains only a single
        component of the vector ``u``. In 3D, however, we must assume that ``u`` will
        contain each of the three vector components, and it must be ordered as,
        ``[edges_1_x, ... ,edge_N_x, edge_1_y, ..., edge_N_y, edge_1_z, ..., edge_N_z]``
        , where ``N`` is the number of boundary edges.

        Returns
        -------
        scipy.sparse.csr_matrix
            Sparse matrix of shape (n_edges, n_boundary_edges) for 1D or 2D mesh,
            (n_edges, 3*n_boundary_edges) for a 3D mesh.

        Notes
        -----
        The integral we are representing on the boundary of the mesh is

        .. math:: \int_{\Omega} \vec{w} \cdot (\vec{u} \times \hat{n}) \partial \Omega

        In discrete form this is:

        .. math:: w^T * P @ u_b

        where `w` is defined on all edges, and `u_b` is all three components defined on
        boundary edges.
        """
        raise NotImplementedError(
            f"boundary_edge_vector_integral not implemented for {type(self)}"
        )

    @property
    def boundary_node_vector_integral(self):
        r"""Represent the operation of integrating a vector function dotted with the boundary normal.

        This matrix represents the boundary surface integral of a vector function
        dotted with the boundary normal and multiplied with a scalar finite volume
        test function on the mesh.

        Returns
        -------
        (n_nodes, dim * n_boundary_nodes) scipy.sparse.csr_matrix
            Sparse matrix of shape.

        Notes
        -----
        The integral we are representing on the boundary of the mesh is

        .. math:: \int_{\Omega} (w \vec{u}) \cdot \hat{n} \partial \Omega

        In discrete form this is:

        .. math:: w^T * P @ u_b

        where `w` is defined on all nodes, and `u_b` is all three components defined on
        boundary nodes.
        """
        raise NotImplementedError(
            f"boundary_node_vector_integral not implemented for {type(self)}"
        )

    @property
    def nodal_laplacian(self):
        r"""Nodal scalar Laplacian operator (nodes to nodes).

        This property constructs the 2nd order scalar Laplacian operator
        that maps from nodes to nodes. The operator is a sparse matrix
        :math:`\mathbf{L_n}` that can be applied as a matrix-vector product to a
        discrete scalar quantity :math:`\boldsymbol{\phi}` that lives on the
        nodes, i.e.::

            laplace_phi = Ln @ phi

        The operator ``*`` assumes a zero Neumann boundary condition for the discrete
        scalar quantity. Once constructed, the operator is stored permanently as
        a property of the mesh.

        Returns
        -------
        (n_nodes, n_nodes) scipy.sparse.csr_matrix
            The numerical Laplacian operator from nodes to nodes

        Notes
        -----
        In continuous space, the scalar Laplacian operator is defined as:

        .. math::
            \psi = \nabla^2 \phi = \frac{\partial^2 \phi}{\partial x^2}
            + \frac{\partial^2 \phi}{\partial y^2}
            + \frac{\partial^2 \phi}{\partial z^2}

        Where :math:`\boldsymbol{\phi}` is the discrete representation of the continuous variable
        :math:`\phi` on the nodes, and :math:`\boldsymbol{\psi}` is the discrete representation
        of its scalar Laplacian on the nodes, **nodal_laplacian** constructs a
        discrete linear operator :math:`\mathbf{L_n}` such that:

        .. math::
            \boldsymbol{\psi} = \mathbf{L_n} \, \boldsymbol{\phi}
        """
        # EXAMPLE INACTIVE BECAUSE OPERATOR IS BROKEN
        #
        # Examples
        # --------

        # Below, we demonstrate how to apply the nodal Laplacian operator to
        # a discrete scalar quantity, the mapping of the nodal Laplacian operator and
        # its sparsity. Our example is carried out on a 2D mesh but it can
        # be done equivalently for a 3D mesh.

        # We start by importing the necessary packages and modules.

        # >>> from discretize import TensorMesh
        # >>> import numpy as np
        # >>> import matplotlib.pyplot as plt
        # >>> import matplotlib as mpl

        # For a discrete scalar quantity defined on the nodes, we take the
        # Laplacian by constructing the operator and multiplying
        # as a matrix-vector product.

        # >>> # Create a uniform grid
        # >>> h = np.ones(20)
        # >>> mesh = TensorMesh([h, h], "CC")
        # >>>
        # >>> # Create a discrete scalar on nodes. The scalar MUST
        # >>> # respect the zero Neumann boundary condition.
        # >>> nodes = mesh.nodes
        # >>> phi = np.exp(-(nodes[:, 0] ** 2 + nodes[:, 1] ** 2) / 4 ** 2)
        # >>>
        # >>> # Construct the Laplacian operator and apply to vector
        # >>> Ln = mesh.nodal_laplacian
        # >>> laplacian_phi = Ln @ phi
        # >>>
        # >>> # Plot
        # >>> fig = plt.figure(figsize=(13, 6))
        # >>> ax1 = fig.add_subplot(121)
        # >>> mesh.plot_image(phi, ax=ax1)
        # >>> ax1.set_title("Scalar at nodes", fontsize=14)
        # >>> ax2 = fig.add_subplot(122)
        # >>> mesh.plot_image(laplacian_phi, ax=ax1)
        # >>> ax2.set_yticks([])
        # >>> ax2.set_ylabel("")
        # >>> ax2.set_title("Laplacian at nodes", fontsize=14)
        # >>> plt.show()

        # The nodal Laplacian operator is a sparse matrix that maps
        # from nodes to nodes. To demonstrate this, we construct
        # a small 2D mesh. We then show the ordering of the nodes
        # and a spy plot illustrating the sparsity of the operator.

        # >>> mesh = TensorMesh([[(1, 4)], [(1, 4)]])
        # >>> fig = plt.figure(figsize=(12, 6))
        # >>> ax1 = fig.add_subplot(211)
        # >>> mesh.plot_grid(ax=ax1)
        # >>> ax1.set_title("Ordering of the Nodes", fontsize=14, pad=15)
        # >>> ax1.plot(mesh.nodes[:, 0], mesh.nodes[:, 1], "ro", markersize=8)
        # >>> for ii, loc in zip(range(mesh.nN), mesh.nodes):
        # ...     ax1.text(loc[0] + 0.05, loc[1] + 0.02, "{0:d}".format(ii), color="r")
        # >>> ax1.set_xticks([])
        # >>> ax1.set_yticks([])
        # >>> ax1.spines['bottom'].set_color('white')
        # >>> ax1.spines['top'].set_color('white')
        # >>> ax1.spines['left'].set_color('white')
        # >>> ax1.spines['right'].set_color('white')
        # >>> ax1.set_xlabel('X', fontsize=16, labelpad=-5)
        # >>> ax1.set_ylabel('Y', fontsize=16, labelpad=-15)
        # >>> ax1.legend(
        # ...     ['Mesh', r'$\mathbf{\phi}$ (nodes)'],
        # ...     loc='upper right', fontsize=14
        # ... )
        # >>> ax2 = fig.add_subplot(212)
        # >>> ax2.spy(mesh.nodal_laplacian)
        # >>> ax2.set_title("Spy Plot", fontsize=14, pad=5)
        # >>> ax2.set_ylabel("Node Index", fontsize=12)
        # >>> ax2.set_xlabel("Node Index", fontsize=12)
        raise NotImplementedError(f"nodal_laplacian not implemented for {type(self)}")

    @property
    def stencil_cell_gradient(self):
        r"""Stencil for cell gradient operator (cell centers to faces).

        This property constructs a differencing operator that acts on
        cell centered quantities. The operator takes the difference between
        the values at adjacent cell centers along each axis direction,
        and places the result on the shared face; e.g. differences
        along the x-axis are mapped to x-faces. The operator is a sparse
        matrix :math:`\mathbf{G}` that can be applied as a matrix-vector
        product to a cell centered quantity :math:`\boldsymbol{\phi}`, i.e.::

            diff_phi = G @ phi

        By default, the operator assumes zero-Neumann boundary conditions
        on the scalar quantity. Before calling **stencil_cell_gradient** however,
        the user can set a mix of zero Dirichlet and zero Neumann boundary
        conditions using :py:attr:`~discretize.operators.DiffOperators.set_cell_gradient_BC`.
        When **stencil_cell_gradient** is called, the boundary conditions are
        enforced for the differencing operator. Once constructed,
        the operator is stored as a property of the mesh.

        Returns
        -------
        (n_faces, n_cells) scipy.sparse.csr_matrix
            The stencil for the cell gradient

        Examples
        --------
        Below, we demonstrate how to set boundary conditions for the cell gradient
        stencil, construct the cell gradient stencil and apply it to a discrete
        scalar quantity. The mapping of the cell gradient operator and
        its sparsity is also illustrated. Our example is carried out on a 2D
        mesh but it can be done equivalently for a 3D mesh.

        We start by importing the necessary packages and modules.

        >>> from discretize import TensorMesh
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> import matplotlib as mpl

        We then construct a mesh and define a scalar function at cell
        centers. In this case, the scalar represents some block within
        a homogeneous medium.

        Create a uniform grid

        >>> h = np.ones(40)
        >>> mesh = TensorMesh([h, h], "CC")

        Create a discrete scalar at cell centers

        >>> centers = mesh.cell_centers
        >>> phi = np.zeros(mesh.nC)
        >>> k = (np.abs(mesh.cell_centers[:, 0]) < 10.) & (np.abs(mesh.cell_centers[:, 1]) < 10.)
        >>> phi[k] = 1.

        Before constructing the operator, we must define
        the boundary conditions; zero Neumann for our example. Once the
        operator is created, it is applied as a matrix-vector product.

        >>> mesh.set_cell_gradient_BC(['neumann', 'neumann'])
        >>> G = mesh.stencil_cell_gradient
        >>> diff_phi = G @ phi

        Now we plot the original scalar, and the differencing taken along the
        x and y axes.

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(15, 4.5))
            >>> ax1 = fig.add_subplot(131)
            >>> mesh.plot_image(phi, ax=ax1)
            >>> ax1.set_title("Scalar at cell centers", fontsize=14)

            >>> ax2 = fig.add_subplot(132)
            >>> mesh.plot_image(diff_phi, ax=ax2, v_type="Fx")
            >>> ax2.set_yticks([])
            >>> ax2.set_ylabel("")
            >>> ax2.set_title("Difference (x-axis)", fontsize=14)

            >>> ax3 = fig.add_subplot(133)
            >>> mesh.plot_image(diff_phi, ax=ax3, v_type="Fy")
            >>> ax3.set_yticks([])
            >>> ax3.set_ylabel("")
            >>> ax3.set_title("Difference (y-axis)", fontsize=14)
            >>> plt.show()

        The cell gradient stencil is a sparse differencing matrix that maps
        from cell centers to faces. To demonstrate this, we construct
        a small 2D mesh. We then show the ordering of the elements
        and a spy plot.

        >>> mesh = TensorMesh([[(1, 3)], [(1, 6)]])
        >>> mesh.set_cell_gradient_BC('neumann')

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(12, 10))
            >>> ax1 = fig.add_subplot(121)
            >>> mesh.plot_grid(ax=ax1)
            >>> ax1.set_title("Mapping of Stencil", fontsize=14, pad=15)
            >>> ax1.plot(mesh.cell_centers[:, 0], mesh.cell_centers[:, 1], "ro", markersize=8)
            >>> for ii, loc in zip(range(mesh.nC), mesh.cell_centers):
            ...     ax1.text(loc[0] + 0.05, loc[1] + 0.02, "{0:d}".format(ii), color="r")
            >>> ax1.plot(mesh.faces_x[:, 0], mesh.faces_x[:, 1], "g>", markersize=8)
            >>> for ii, loc in zip(range(mesh.nFx), mesh.faces_x):
            ...     ax1.text(loc[0] + 0.05, loc[1] + 0.02, "{0:d}".format(ii), color="g")
            >>> ax1.plot(mesh.faces_y[:, 0], mesh.faces_y[:, 1], "g^", markersize=8)
            >>> for ii, loc in zip(range(mesh.nFy), mesh.faces_y):
            ...     ax1.text(loc[0] + 0.05, loc[1] + 0.02, "{0:d}".format((ii + mesh.nFx)), color="g")
            >>> ax1.set_xticks([])
            >>> ax1.set_yticks([])
            >>> ax1.spines['bottom'].set_color('white')
            >>> ax1.spines['top'].set_color('white')
            >>> ax1.spines['left'].set_color('white')
            >>> ax1.spines['right'].set_color('white')
            >>> ax1.set_xlabel('X', fontsize=16, labelpad=-5)
            >>> ax1.set_ylabel('Y', fontsize=16, labelpad=-15)
            >>> ax1.legend(
            ...     ['Mesh', r'$\mathbf{\phi}$ (centers)', r'$\mathbf{G^\ast \phi}$ (faces)'],
            ...     loc='upper right', fontsize=14
            ... )

            >>> ax2 = fig.add_subplot(122)
            >>> ax2.spy(mesh.stencil_cell_gradient)
            >>> ax2.set_title("Spy Plot", fontsize=14, pad=5)
            >>> ax2.set_ylabel("Face Index", fontsize=12)
            >>> ax2.set_xlabel("Cell Index", fontsize=12)
            >>> plt.show()
        """
        raise NotImplementedError(
            f"stencil_cell_gradient not implemented for {type(self)}"
        )

    # Inner Products
    def get_face_inner_product(
        self,
        model=None,
        invert_model=False,
        invert_matrix=False,
        do_fast=True,
        **kwargs,
    ):
        r"""Generate the face inner product matrix or its inverse.

        This method generates the inner product matrix (or its inverse)
        when discrete variables are defined on mesh faces. It is also capable of
        constructing the inner product matrix when physical properties
        are defined in the form of constitutive relations. For a comprehensive
        description of the inner product matrices that can be constructed
        with **get_face_inner_product**, see *Notes*.

        Parameters
        ----------
        model : None or numpy.ndarray, optional
            Parameters defining the material properties for every cell in the mesh.
            Inner product matrices can be constructed for the following cases:

            - *None* : returns the basic inner product matrix
            - *(n_cells)* :class:`numpy.ndarray` : returns inner product matrix for an
              isotropic model. The array contains a scalar physical property value for
              each cell.
            - *(n_cells, dim)* :class:`numpy.ndarray` : returns inner product matrix for
              diagonal anisotropic case. Columns are ordered ``np.c_[σ_xx, σ_yy, σ_zz]``.
              This can also a be a 1D array with the same number of total elements in
              column major order.
            - *(n_cells, 3)* :class:`numpy.ndarray` (``dim`` is 2) or
              *(n_cells, 6)* :class:`numpy.ndarray` (``dim`` is 3) :
              returns inner product matrix for full tensor properties case. Columns are
              ordered ``np.c_[σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz]`` This can also be a
              1D array with the same number of total elements in column major order.

        invert_model : bool, optional
            The inverse of *model* is used as the physical property.
        invert_matrix : bool, optional
            Returns the inverse of the inner product matrix.
            The inverse not implemented for full tensor properties.
        do_fast : bool, optional
            Do a faster implementation (if available).

        Returns
        -------
        (n_faces, n_faces) scipy.sparse.csr_matrix
            inner product matrix

        Notes
        -----
        For continuous vector quantities :math:`\vec{u}` and :math:`\vec{w}`
        whose discrete representations :math:`\mathbf{u}` and :math:`\mathbf{w}`
        live on the faces, **get_face_inner_product** constructs the inner product matrix
        :math:`\mathbf{M_\ast}` (or its inverse :math:`\mathbf{M_\ast^{-1}}`) for the
        following cases:

        **Basic Inner Product:** the inner product between :math:`\vec{u}` and :math:`\vec{w}`

        .. math::
            \langle \vec{u}, \vec{w} \rangle = \mathbf{u^T \, M \, w}

        **Isotropic Case:** the inner product between :math:`\vec{u}` and :math:`\sigma \vec{w}`
        where :math:`\sigma` is a scalar function.

        .. math::
            \langle \vec{u}, \sigma \vec{w} \rangle = \mathbf{u^T \, M_\sigma \, w}

        **Tensor Case:** the inner product between :math:`\vec{u}` and :math:`\Sigma \vec{w}`
        where :math:`\Sigma` is tensor function; :math:`\sigma_{xy} = \sigma_{xz} = \sigma_{yz} = 0`
        for diagonal anisotropy.

        .. math::
            \langle \vec{u}, \Sigma \vec{w} \rangle = \mathbf{u^T \, M_\Sigma \, w}
            \;\;\; \textrm{where} \;\;\;
            \Sigma = \begin{bmatrix}
            \sigma_{xx} & \sigma_{xy} & \sigma_{xz} \\
            \sigma_{xy} & \sigma_{yy} & \sigma_{yz} \\
            \sigma_{xz} & \sigma_{yz} & \sigma_{zz}
            \end{bmatrix}

        Examples
        --------
        Here we provide some examples of face inner product matrices.
        For simplicity, we will work on a 2 x 2 x 2 tensor mesh.
        As seen below, we begin by constructing and imaging the basic
        face inner product matrix.

        >>> from discretize import TensorMesh
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import matplotlib as mpl

        >>> h = np.ones(2)
        >>> mesh = TensorMesh([h, h, h])
        >>> Mf = mesh.get_face_inner_product()

        >>> fig = plt.figure(figsize=(6, 6))
        >>> ax = fig.add_subplot(111)
        >>> ax.imshow(Mf.todense())
        >>> ax.set_title('Basic Face Inner Product Matrix', fontsize=18)
        >>> plt.show()

        Next, we consider the case where the physical properties
        of the cells are defined by consistutive relations. For
        the isotropic, diagonal anisotropic and full tensor cases,
        we show the physical property tensor for a single cell.

        Define 4 constitutive parameters and define the tensor
        for each cell for isotropic, diagonal and tensor cases.

        >>> sig1, sig2, sig3, sig4, sig5, sig6 = 6, 5, 4, 3, 2, 1
        >>> sig_iso_tensor = sig1 * np.eye(3)
        >>> sig_diag_tensor = np.diag(np.array([sig1, sig2, sig3]))
        >>> sig_full_tensor = np.array([
        ...     [sig1, sig4, sig5],
        ...     [sig4, sig2, sig6],
        ...     [sig5, sig6, sig3]
        ... ])

        Then plot matrix entries,

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(15, 5))
            >>> ax1 = fig.add_subplot(131)
            >>> ax1.imshow(sig_iso_tensor)
            >>> ax1.axis('off')
            >>> ax1.set_title("Tensor (isotropic)", fontsize=16)
            >>> ax2 = fig.add_subplot(132)
            >>> ax2.imshow(sig_diag_tensor)
            >>> ax2.axis('off')
            >>> ax2.set_title("Tensor (diagonal anisotropic)", fontsize=16)
            >>> ax3 = fig.add_subplot(133)
            >>> ax3.imshow(sig_full_tensor)
            >>> ax3.axis('off')
            >>> ax3.set_title("Tensor (full anisotropic)", fontsize=16)
            >>> plt.show()

        Here, construct and image the face inner product matrices for
        the isotropic, diagonal anisotropic and full tensor cases.
        Spy plots are used to demonstrate the sparsity of the inner
        product matrices.

        Isotropic case:

        >>> v = np.ones(mesh.nC)
        >>> sig = sig1 * v
        >>> M1 = mesh.get_face_inner_product(sig)

        Diagonal anisotropic case:

        >>> sig = np.c_[sig1*v, sig2*v, sig3*v]
        >>> M2 = mesh.get_face_inner_product(sig)

        Full anisotropic case:

        >>> sig = np.tile(np.c_[sig1, sig2, sig3, sig4, sig5, sig6], (mesh.nC, 1))
        >>> M3 = mesh.get_face_inner_product(sig)

        And then we can plot the sparse representation,

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(12, 4))
            >>> ax1 = fig.add_subplot(131)
            >>> ax1.spy(M1, ms=5)
            >>> ax1.set_title("M (isotropic)", fontsize=16)
            >>> ax2 = fig.add_subplot(132)
            >>> ax2.spy(M2, ms=5)
            >>> ax2.set_title("M (diagonal anisotropic)", fontsize=16)
            >>> ax3 = fig.add_subplot(133)
            >>> ax3.spy(M3, ms=5)
            >>> ax3.set_title("M (full anisotropic)", fontsize=16)
            >>> plt.show()
        """
        raise NotImplementedError(
            f"get_face_inner_product not implemented for {type(self)}"
        )

    def get_edge_inner_product(
        self,
        model=None,
        invert_model=False,
        invert_matrix=False,
        do_fast=True,
        **kwargs,
    ):
        r"""Generate the edge inner product matrix or its inverse.

        This method generates the inner product matrix (or its inverse)
        when discrete variables are defined on mesh edges. It is also capable of
        constructing the inner product matrix when physical properties
        are defined in the form of constitutive relations. For a comprehensive
        description of the inner product matrices that can be constructed
        with **get_edge_inner_product**, see *Notes*.

        Parameters
        ----------
        model : None or numpy.ndarray
            Parameters defining the material properties for every cell in the mesh.
            Inner product matrices can be constructed for the following cases:

            - *None* : returns the basic inner product matrix
            - *(n_cells)* :class:`numpy.ndarray` : returns inner product matrix for an
              isotropic model. The array contains a scalar physical property value for
              each cell.
            - *(n_cells, dim)* :class:`numpy.ndarray` : returns inner product matrix for
              diagonal anisotropic case. Columns are ordered ``np.c_[σ_xx, σ_yy, σ_zz]``.
              This can also a be a 1D array with the same number of total elements in
              column major order.
            - *(n_cells, 3)* :class:`numpy.ndarray` (``dim`` is 2) or
              *(n_cells, 6)* :class:`numpy.ndarray` (``dim`` is 3) :
              returns inner product matrix for full tensor properties case. Columns are
              ordered ``np.c_[σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz]`` This can also be a
              1D array with the same number of total elements in column major order.

        invert_model : bool, optional
            The inverse of *model* is used as the physical property.
        invert_matrix : bool, optional
            Teturns the inverse of the inner product matrix.
            The inverse not implemented for full tensor properties.
        do_fast : bool, optional
            Do a faster implementation (if available).

        Returns
        -------
        (n_edges, n_edges) scipy.sparse.csr_matrix
            inner product matrix

        Notes
        -----
        For continuous vector quantities :math:`\vec{u}` and :math:`\vec{w}`
        whose discrete representations :math:`\mathbf{u}` and :math:`\mathbf{w}`
        live on the edges, **get_edge_inner_product** constructs the inner product
        matrix :math:`\mathbf{M_\ast}` (or its inverse :math:`\mathbf{M_\ast^{-1}}`) for
        the following cases:

        **Basic Inner Product:** the inner product between :math:`\vec{u}` and
        :math:`\vec{w}`.

        .. math::
            \langle \vec{u}, \vec{w} \rangle = \mathbf{u^T \, M \, w}

        **Isotropic Case:** the inner product between :math:`\vec{u}` and
        :math:`\sigma \vec{w}` where :math:`\sigma` is a scalar function.

        .. math::
            \langle \vec{u}, \sigma \vec{w} \rangle = \mathbf{u^T \, M_\sigma \, w}

        **Tensor Case:** the inner product between :math:`\vec{u}` and
        :math:`\Sigma \vec{w}` where :math:`\Sigma` is tensor function;
        :math:`\sigma_{xy} = \sigma_{xz} = \sigma_{yz} = 0` for diagonal anisotropy.

        .. math::
            \langle \vec{u}, \Sigma \vec{w} \rangle =
            \mathbf{u^T \, M_\Sigma \, w} \;\;\; \textrm{where} \;\;\;
            \Sigma = \begin{bmatrix}
            \sigma_{xx} & \sigma_{xy} & \sigma_{xz} \\
            \sigma_{xy} & \sigma_{yy} & \sigma_{yz} \\
            \sigma_{xz} & \sigma_{yz} & \sigma_{zz}
            \end{bmatrix}

        Examples
        --------
        Here we provide some examples of edge inner product matrices.
        For simplicity, we will work on a 2 x 2 x 2 tensor mesh.
        As seen below, we begin by constructing and imaging the basic
        edge inner product matrix.

        >>> from discretize import TensorMesh
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import matplotlib as mpl

        >>> h = np.ones(2)
        >>> mesh = TensorMesh([h, h, h])
        >>> Me = mesh.get_edge_inner_product()

        >>> fig = plt.figure(figsize=(6, 6))
        >>> ax = fig.add_subplot(111)
        >>> ax.imshow(Me.todense())
        >>> ax.set_title('Basic Edge Inner Product Matrix', fontsize=18)
        >>> plt.show()

        Next, we consider the case where the physical properties
        of the cells are defined by consistutive relations. For
        the isotropic, diagonal anisotropic and full tensor cases,
        we show the physical property tensor for a single cell.

        Define 4 constitutive parameters and define the tensor
        for each cell for isotropic, diagonal and tensor cases.

        >>> sig1, sig2, sig3, sig4, sig5, sig6 = 6, 5, 4, 3, 2, 1
        >>> sig_iso_tensor = sig1 * np.eye(3)
        >>> sig_diag_tensor = np.diag(np.array([sig1, sig2, sig3]))
        >>> sig_full_tensor = np.array([
        ...     [sig1, sig4, sig5],
        ...     [sig4, sig2, sig6],
        ...     [sig5, sig6, sig3]
        ... ])

        Then plot the matrix entries,

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(15, 5))
            >>> ax1 = fig.add_subplot(131)
            >>> ax1.imshow(sig_iso_tensor)
            >>> ax1.axis('off')
            >>> ax1.set_title("Tensor (isotropic)", fontsize=16)
            >>> ax2 = fig.add_subplot(132)
            >>> ax2.imshow(sig_diag_tensor)
            >>> ax2.axis('off')
            >>> ax2.set_title("Tensor (diagonal anisotropic)", fontsize=16)
            >>> ax3 = fig.add_subplot(133)
            >>> ax3.imshow(sig_full_tensor)
            >>> ax3.axis('off')
            >>> ax3.set_title("Tensor (full anisotropic)", fontsize=16)
            >>> plt.show()

        Here construct and image the edge inner product matrices for
        the isotropic, diagonal anisotropic and full tensor cases.
        Spy plots are used to demonstrate the sparsity of the inner
        product matrices.

        Isotropic case:

        >>> v = np.ones(mesh.nC)
        >>> sig = sig1 * v
        >>> M1 = mesh.get_edge_inner_product(sig)

        Diagonal anisotropic case:

        >>> sig = np.c_[sig1*v, sig2*v, sig3*v]
        >>> M2 = mesh.get_edge_inner_product(sig)

        Full anisotropic

        >>> sig = np.tile(np.c_[sig1, sig2, sig3, sig4, sig5, sig6], (mesh.nC, 1))
        >>> M3 = mesh.get_edge_inner_product(sig)

        Then plot the sparse representation,

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(12, 4))
            >>> ax1 = fig.add_subplot(131)
            >>> ax1.spy(M1, ms=5)
            >>> ax1.set_title("M (isotropic)", fontsize=16)
            >>> ax2 = fig.add_subplot(132)
            >>> ax2.spy(M2, ms=5)
            >>> ax2.set_title("M (diagonal anisotropic)", fontsize=16)
            >>> ax3 = fig.add_subplot(133)
            >>> ax3.spy(M3, ms=5)
            >>> ax3.set_title("M (full anisotropic)", fontsize=16)
            >>> plt.show()

        """
        raise NotImplementedError(
            f"get_edge_inner_product not implemented for {type(self)}"
        )

    def get_face_inner_product_deriv(
        self, model, do_fast=True, invert_model=False, invert_matrix=False, **kwargs
    ):
        r"""Get a function handle to multiply a vector with derivative of face inner product matrix (or its inverse).

        Let :math:`\mathbf{M}(\mathbf{m})` be the face inner product matrix
        constructed with a set of physical property parameters :math:`\mathbf{m}`
        (or its inverse). **get_face_inner_product_deriv** constructs a function handle

        .. math::
            \mathbf{F}(\mathbf{u}) = \mathbf{u}^T \, \frac{\partial \mathbf{M}(\mathbf{m})}{\partial \mathbf{m}}

        which accepts any numpy.array :math:`\mathbf{u}` of shape (n_faces,). That is,
        **get_face_inner_product_deriv** constructs a function handle for computing
        the dot product between a vector :math:`\mathbf{u}` and the derivative of the
        face inner product matrix (or its inverse) with respect to the property parameters.
        When computed, :math:`\mathbf{F}(\mathbf{u})` returns a ``scipy.sparse.csr_matrix``
        of shape (n_faces, n_param).

        The function handle can be created for isotropic, diagonal
        isotropic and full tensor physical properties; see notes.

        Parameters
        ----------
        model : numpy.ndarray
            Parameters defining the material properties for every cell in the mesh.
            Inner product matrices can be constructed for the following cases:

            - *(n_cells)* :class:`numpy.ndarray` : Isotropic case. *model* contains a
              scalar physical property value for each cell.
            - *(n_cells, dim)* :class:`numpy.ndarray` : Diagonal anisotropic case.
              Columns are ordered ``np.c_[σ_xx, σ_yy, σ_zz]``. This can also a be a 1D
              array with the same number of total elements in column major order.
            - *(n_cells, 3)* :class:`numpy.ndarray` (``dim`` is 2) or
              *(n_cells, 6)* :class:`numpy.ndarray` (``dim`` is 3) : Full tensor properties case. Columns
              are ordered ``np.c_[σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz]`` This can also be
              a 1D array with the same number of total elements in column major order.

        invert_model : bool, optional
            The inverse of *model* is used as the physical property.
        invert_matrix : bool, optional
            Returns the inverse of the inner product matrix.
            The inverse not implemented for full tensor properties.
        do_fast : bool, optional
            Do a faster implementation (if available).

        Returns
        -------
        function
            The function handle :math:`\mathbf{F}(\mathbf{u})` which accepts a
            (``n_faces``) :class:`numpy.ndarray` :math:`\mathbf{u}`. The function
            returns a (``n_faces``, ``n_params``) :class:`scipy.sparse.csr_matrix`.

        Notes
        -----
        Let :math:`\mathbf{M}(\mathbf{m})` be the face inner product matrix (or its inverse)
        for the set of physical property parameters :math:`\mathbf{m}`. And let :math:`\mathbf{u}`
        be a discrete quantity that lives on the faces. **get_face_inner_product_deriv**
        creates a function handle for computing the following:

        .. math::
            \mathbf{F}(\mathbf{u}) = \mathbf{u}^T \, \frac{\partial \mathbf{M}(\mathbf{m})}{\partial \mathbf{m}}

        The dimensions of the sparse matrix constructed by computing :math:`\mathbf{F}(\mathbf{u})`
        for some :math:`\mathbf{u}` depends on the constitutive relation defined for each cell.
        These cases are summarized below.

        **Isotropic Case:** The physical property for each cell is defined by a scalar value.
        Therefore :math:`\mathbf{m}` is a (``n_cells``) :class:`numpy.ndarray`. The sparse matrix
        output by computing :math:`\mathbf{F}(\mathbf{u})` has shape (``n_faces``, ``n_cells``).

        **Diagonal Anisotropic Case:** In this case, the physical properties for each cell are
        defined by a diagonal tensor

        .. math::
            \Sigma = \begin{bmatrix}
            \sigma_{xx} & 0 & 0 \\
            0 & \sigma_{yy} & 0 \\
            0 & 0 & \sigma_{zz}
            \end{bmatrix}

        Thus there are ``dim * n_cells`` physical property parameters and :math:`\mathbf{m}` is
        a (``dim * n_cells``) :class:`numpy.ndarray`.  The sparse matrix
        output by computing :math:`\mathbf{F}(\mathbf{u})` has shape (``n_faces``, ``dim * n_cells``).

        **Full Tensor Case:** In this case, the physical properties for each cell are
        defined by a full tensor

        .. math::
            \Sigma = \begin{bmatrix}
            \sigma_{xx} & \sigma_{xy} & \sigma_{xz} \\
            \sigma_{xy} & \sigma_{yy} & \sigma_{yz} \\
            \sigma_{xz} & \sigma_{yz} & \sigma_{zz}
            \end{bmatrix}

        Thus there are ``6 * n_cells`` physical property parameters in 3 dimensions, or
        ``3 * n_cells`` physical property parameters in 2 dimensions, and
        :math:`\mathbf{m}` is a (``n_params``) :class:`numpy.ndarray`.
        The sparse matrix output by computing :math:`\mathbf{F}(\mathbf{u})`
        has shape (``n_faces``, ``n_params``).

        Examples
        --------
        Here, we construct a 4 cell by 4 cell tensor mesh. For our first example we
        consider isotropic physical properties; that is, the physical properties
        of each cell are defined a scalar value. We construct the face inner product
        matrix and visualize it with a spy plot. We then use
        **get_face_inner_product_deriv** to construct the function handle
        :math:`\mathbf{F}(\mathbf{u})` and plot the evaluation
        of this function on a spy plot.

        >>> from discretize import TensorMesh
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import matplotlib as mpl
        >>> mpl.rcParams.update({'font.size': 14})
        >>> np.random.seed(45)
        >>> mesh = TensorMesh([[(1, 4)], [(1, 4)]])

        Define a model, and a random vector to multiply the derivative with,
        then we grab the respective derivative function and calculate the
        sparse matrix,

        >>> m = np.random.rand(mesh.nC)  # physical property parameters
        >>> u = np.random.rand(mesh.nF)  # vector of shape (n_faces)
        >>> Mf = mesh.get_face_inner_product(m)
        >>> F = mesh.get_face_inner_product_deriv(m)  # Function handle
        >>> dFdm_u = F(u)

        Spy plot for the inner product matrix and its derivative

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(15, 5))
            >>> ax1 = fig.add_axes([0.05, 0.05, 0.3, 0.85])
            >>> ax1.spy(Mf, ms=6)
            >>> ax1.set_title("Face Inner Product Matrix (Isotropic)", fontsize=14, pad=5)
            >>> ax1.set_xlabel("Face Index", fontsize=12)
            >>> ax1.set_ylabel("Face Index", fontsize=12)
            >>> ax2 = fig.add_axes([0.43, 0.05, 0.17, 0.8])
            >>> ax2.spy(dFdm_u, ms=6)
            >>> ax2.set_title(
            ...     "$u^T \, \dfrac{\partial M(m)}{\partial m}$ (Isotropic)",
            ...     fontsize=14, pad=5
            ... )
            >>> ax2.set_xlabel("Parameter Index", fontsize=12)
            >>> ax2.set_ylabel("Face Index", fontsize=12)
            >>> plt.show()

        For our second example, the physical properties on the mesh are fully
        anisotropic; that is, the physical properties of each cell are defined
        by a tensor with parameters :math:`\sigma_1`, :math:`\sigma_2` and :math:`\sigma_3`.
        Once again we construct the face inner product matrix and visualize it with a
        spy plot. We then use **get_face_inner_product_deriv** to construct the
        function handle :math:`\mathbf{F}(\mathbf{u})` and plot the evaluation
        of this function on a spy plot.

        >>> m = np.random.rand(mesh.nC, 3)  # anisotropic physical property parameters
        >>> u = np.random.rand(mesh.nF)     # vector of shape (n_faces)
        >>> Mf = mesh.get_face_inner_product(m)
        >>> F = mesh.get_face_inner_product_deriv(m)  # Function handle
        >>> dFdm_u = F(u)

        Plot the anisotropic inner product matrix and its derivative matrix,

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(15, 5))
            >>> ax1 = fig.add_axes([0.05, 0.05, 0.3, 0.8])
            >>> ax1.spy(Mf, ms=6)
            >>> ax1.set_title("Face Inner Product (Full Tensor)", fontsize=14, pad=5)
            >>> ax1.set_xlabel("Face Index", fontsize=12)
            >>> ax1.set_ylabel("Face Index", fontsize=12)
            >>> ax2 = fig.add_axes([0.4, 0.05, 0.45, 0.85])
            >>> ax2.spy(dFdm_u, ms=6)
            >>> ax2.set_title(
            ...     "$u^T \, \dfrac{\partial M(m)}{\partial m} \;$ (Full Tensor)",
            ...     fontsize=14, pad=5
            ... )
            >>> ax2.set_xlabel("Parameter Index", fontsize=12)
            >>> ax2.set_ylabel("Face Index", fontsize=12)
            >>> plt.show()

        """
        raise NotImplementedError(
            f"get_face_inner_product_deriv not implemented for {type(self)}"
        )

    def get_edge_inner_product_deriv(
        self, model, do_fast=True, invert_model=False, invert_matrix=False, **kwargs
    ):
        r"""Get a function handle to multiply vector with derivative of edge inner product matrix (or its inverse).

        Let :math:`\mathbf{M}(\mathbf{m})` be the edge inner product matrix
        constructed with a set of physical property parameters :math:`\mathbf{m}`
        (or its inverse). **get_edge_inner_product_deriv** constructs a function handle

        .. math::
            \mathbf{F}(\mathbf{u}) = \mathbf{u}^T \, \frac{\partial \mathbf{M}(\mathbf{m})}{\partial \mathbf{m}}

        which accepts any numpy.array :math:`\mathbf{u}` of shape (n_edges,). That is,
        **get_edge_inner_product_deriv** constructs a function handle for computing
        the dot product between a vector :math:`\mathbf{u}` and the derivative of the
        edge inner product matrix (or its inverse) with respect to the property parameters.
        When computed, :math:`\mathbf{F}(\mathbf{u})` returns a ``scipy.sparse.csr_matrix``
        of shape (n_edges, n_param).

        The function handle can be created for isotropic, diagonal
        isotropic and full tensor physical properties; see notes.

        Parameters
        ----------
        model : numpy.ndarray
            Parameters defining the material properties for every cell in the mesh.

            Allows for the following cases:

            - *(n_cells)* :class:`numpy.ndarray` : Isotropic case. *model* contains a
              scalar physical property value for each cell.
            - *(n_cells, dim)* :class:`numpy.ndarray` : Diagonal anisotropic case.
              Columns are ordered ``np.c_[σ_xx, σ_yy, σ_zz]``. This can also a be a 1D
              array with the same number of total elements in column major order.
            - *(n_cells, 3)* :class:`numpy.ndarray` (``dim`` is 2) or
              *(n_cells, 6)* :class:`numpy.ndarray` (``dim`` is 3) : Full tensor properties case. Columns
              are ordered ``np.c_[σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz]`` This can also be
              a 1D array with the same number of total elements in column major order.

        invert_model : bool, optional
            The inverse of *model* is used as the physical property.
        invert_matrix : bool, optional
            Returns the function handle for the inverse of the inner product matrix
            The inverse not implemented for full tensor properties.
        do_fast : bool, optional
            Do a faster implementation (if available).

        Returns
        -------
        function
            The function handle :math:`\mathbf{F}(\mathbf{u})` which accepts a
            (``n_edges``) :class:`numpy.ndarray` :math:`\mathbf{u}`. The function
            returns a (``n_edges``, ``n_params``) :class:`scipy.sparse.csr_matrix`.

        Notes
        -----
        Let :math:`\mathbf{M}(\mathbf{m})` be the edge inner product matrix (or its inverse)
        for the set of physical property parameters :math:`\mathbf{m}`. And let :math:`\mathbf{u}`
        be a discrete quantity that lives on the edges. **get_edge_inner_product_deriv**
        creates a function handle for computing the following:

        .. math::
            \mathbf{F}(\mathbf{u}) = \mathbf{u}^T \, \frac{\partial \mathbf{M}(\mathbf{m})}{\partial \mathbf{m}}

        The dimensions of the sparse matrix constructed by computing :math:`\mathbf{F}(\mathbf{u})`
        for some :math:`\mathbf{u}` depends on the constitutive relation defined for each cell.
        These cases are summarized below.

        **Isotropic Case:** The physical property for each cell is defined by a scalar value.
        Therefore :math:`\mathbf{m}` is a (``n_cells``) :class:`numpy.ndarray`. The sparse matrix
        output by computing :math:`\mathbf{F}(\mathbf{u})` has shape (``n_edges``, ``n_cells``).

        **Diagonal Anisotropic Case:** In this case, the physical properties for each cell are
        defined by a diagonal tensor

        .. math::
            \Sigma = \begin{bmatrix}
            \sigma_{xx} & 0 & 0 \\
            0 & \sigma_{yy} & 0 \\
            0 & 0 & \sigma_{zz}
            \end{bmatrix}

        Thus there are ``dim * n_cells`` physical property parameters and :math:`\mathbf{m}` is
        a (``dim * n_cells``) :class:`numpy.ndarray`.  The sparse matrix
        output by computing :math:`\mathbf{F}(\mathbf{u})` has shape (``n_edges``, ``dim * n_cells``).

        **Full Tensor Case:** In this case, the physical properties for each cell are
        defined by a full tensor

        .. math::
            \Sigma = \begin{bmatrix}
            \sigma_{xx} & \sigma_{xy} & \sigma_{xz} \\
            \sigma_{xy} & \sigma_{yy} & \sigma_{yz} \\
            \sigma_{xz} & \sigma_{yz} & \sigma_{zz}
            \end{bmatrix}

        Thus there are ``6 * n_cells`` physical property parameters in 3 dimensions, or
        ``3 * n_cells`` physical property parameters in 2 dimensions, and :math:`\mathbf{m}` is
        a (``n_params``) :class:`numpy.ndarray`. The sparse matrix
        output by computing :math:`\mathbf{F}(\mathbf{u})` has shape (``n_edges``, ``n_params``).

        Examples
        --------
        Here, we construct a 4 cell by 4 cell tensor mesh. For our first example we
        consider isotropic physical properties; that is, the physical properties
        of each cell are defined a scalar value. We construct the edge inner product
        matrix and visualize it with a spy plot. We then use
        **get_edge_inner_product_deriv** to construct the function handle
        :math:`\mathbf{F}(\mathbf{u})` and plot the evaluation of this function on a spy
        plot.

        >>> from discretize import TensorMesh
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> import matplotlib as mpl
        >>> mpl.rcParams.update({'font.size': 14})
        >>> np.random.seed(45)
        >>> mesh = TensorMesh([[(1, 4)], [(1, 4)]])

        Next we create a random isotropic model vector, and a random vector to multiply
        the derivative with (for illustration purposes).

        >>> m = np.random.rand(mesh.nC)  # physical property parameters
        >>> u = np.random.rand(mesh.nF)  # vector of shape (n_edges)
        >>> Me = mesh.get_edge_inner_product(m)
        >>> F = mesh.get_edge_inner_product_deriv(m)  # Function handle
        >>> dFdm_u = F(u)

        Plot inner product matrix and its derivative matrix

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(15, 5))
            >>> ax1 = fig.add_axes([0.05, 0.05, 0.3, 0.8])
            >>> ax1.spy(Me, ms=6)
            >>> ax1.set_title("Edge Inner Product Matrix (Isotropic)", fontsize=14, pad=5)
            >>> ax1.set_xlabel("Edge Index", fontsize=12)
            >>> ax1.set_ylabel("Edge Index", fontsize=12)
            >>> ax2 = fig.add_axes([0.43, 0.05, 0.17, 0.8])
            >>> ax2.spy(dFdm_u, ms=6)
            >>> ax2.set_title(
            ...     "$u^T \, \dfrac{\partial M(m)}{\partial m}$ (Isotropic)",
            ...     fontsize=14, pad=5
            ... )
            >>> ax2.set_xlabel("Parameter Index", fontsize=12)
            >>> ax2.set_ylabel("Edge Index", fontsize=12)
            >>> plt.show()

        For our second example, the physical properties on the mesh are fully
        anisotropic; that is, the physical properties of each cell are defined
        by a tensor with parameters :math:`\sigma_1`, :math:`\sigma_2` and :math:`\sigma_3`.
        Once again we construct the edge inner product matrix and visualize it with a
        spy plot. We then use **get_edge_inner_product_deriv** to construct the
        function handle :math:`\mathbf{F}(\mathbf{u})` and plot the evaluation
        of this function on a spy plot.

        >>> m = np.random.rand(mesh.nC, 3)  # physical property parameters
        >>> u = np.random.rand(mesh.nF)     # vector of shape (n_edges)
        >>> Me = mesh.get_edge_inner_product(m)
        >>> F = mesh.get_edge_inner_product_deriv(m)  # Function handle
        >>> dFdm_u = F(u)

        Plot the anisotropic inner product matrix and its derivative matrix

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(15, 5))
            >>> ax1 = fig.add_axes([0.05, 0.05, 0.3, 0.8])
            >>> ax1.spy(Me, ms=6)
            >>> ax1.set_title("Edge Inner Product (Full Tensor)", fontsize=14, pad=5)
            >>> ax1.set_xlabel("Edge Index", fontsize=12)
            >>> ax1.set_ylabel("Edge Index", fontsize=12)
            >>> ax2 = fig.add_axes([0.4, 0.05, 0.45, 0.8])
            >>> ax2.spy(dFdm_u, ms=6)
            >>> ax2.set_title(
            ...     "$u^T \, \dfrac{\partial M(m)}{\partial m} \;$ (Full Tensor)",
            ...     fontsize=14, pad=5
            ... )
            >>> ax2.set_xlabel("Parameter Index", fontsize=12)
            >>> ax2.set_ylabel("Edge Index", fontsize=12)
            >>> plt.show()
        """
        raise NotImplementedError(
            f"get_edge_inner_product_deriv not implemented for {type(self)}"
        )

    # Averaging
    @property
    def average_face_to_cell(self):
        r"""Averaging operator from faces to cell centers (scalar quantities).

        This property constructs a 2nd order averaging operator that maps scalar
        quantities from faces to cell centers. This averaging operator is
        used when a discrete scalar quantity defined on mesh faces must be
        projected to cell centers. Once constructed, the operator is stored
        permanently as a property of the mesh. *See notes*.

        Returns
        -------
        (n_cells, n_faces) scipy.sparse.csr_matrix
            The scalar averaging operator from faces to cell centers

        Notes
        -----
        Let :math:`\boldsymbol{\phi_f}` be a discrete scalar quantity that
        lives on mesh faces. **average_face_to_cell** constructs a discrete
        linear operator :math:`\mathbf{A_{fc}}` that projects
        :math:`\boldsymbol{\phi_f}` to cell centers, i.e.:

        .. math::
            \boldsymbol{\phi_c} = \mathbf{A_{fc}} \, \boldsymbol{\phi_f}

        where :math:`\boldsymbol{\phi_c}` approximates the value of the scalar
        quantity at cell centers. For each cell, we are simply averaging
        the values defined on its faces. The operation is implemented as a
        matrix vector product, i.e.::

            phi_c = Afc @ phi_f

        Examples
        --------
        Here we compute the values of a scalar function on the faces. We then create
        an averaging operator to approximate the function at cell centers. We choose
        to define a scalar function that is strongly discontinuous in some places to
        demonstrate how the averaging operator will smooth out discontinuities.

        We start by importing the necessary packages and defining a mesh.

        >>> from discretize import TensorMesh
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        >>> h = np.ones(40)
        >>> mesh = TensorMesh([h, h], x0="CC")

        Then we create a scalar variable on faces

        >>> phi_f = np.zeros(mesh.nF)
        >>> xy = mesh.faces
        >>> phi_f[(xy[:, 1] > 0)] = 25.0
        >>> phi_f[(xy[:, 1] < -10.0) & (xy[:, 0] > -10.0) & (xy[:, 0] < 10.0)] = 50.0

        Next, we construct the averaging operator and apply it to
        the discrete scalar quantity to approximate the value at cell centers.

        >>> Afc = mesh.average_face_to_cell
        >>> phi_c = Afc @ phi_f

        And finally plot the results:

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(11, 5))
            >>> ax1 = fig.add_subplot(121)
            >>> mesh.plot_image(phi_f, ax=ax1, v_type="F")
            >>> ax1.set_title("Variable at faces", fontsize=16)
            >>> ax2 = fig.add_subplot(122)
            >>> mesh.plot_image(phi_c, ax=ax2, v_type="CC")
            >>> ax2.set_title("Averaged to cell centers", fontsize=16)
            >>> plt.show()

        Below, we show a spy plot illustrating the sparsity and mapping
        of the operator

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(9, 9))
            >>> ax1 = fig.add_subplot(111)
            >>> ax1.spy(Afc, ms=1)
            >>> ax1.set_title("Face Index", fontsize=12, pad=5)
            >>> ax1.set_ylabel("Cell Index", fontsize=12)
            >>> plt.show()
        """
        raise NotImplementedError(
            f"average_face_to_cell not implemented for {type(self)}"
        )

    @property
    def average_face_to_cell_vector(self):
        r"""Averaging operator from faces to cell centers (vector quantities).

        This property constructs the averaging operator that independently maps the
        Cartesian components of vector quantities from faces to cell centers.
        This averaging operators is used when a discrete vector quantity defined on mesh faces
        must be approximated at cell centers. Once constructed, the operator is
        stored permanently as a property of the mesh.

        Be aware that the Cartesian components of the original vector
        are defined on their respective faces; e.g. the x-component lives
        on x-faces. However, the x, y and z components are being averaged
        separately to cell centers. The operation is implemented as a
        matrix vector product, i.e.::

            u_c = Afc @ u_f

        Returns
        -------
        (dim * n_cells, n_faces) scipy.sparse.csr_matrix
            The vector averaging operator from faces to cell centers. Since we
            are averaging a vector quantity to cell centers, the first dimension
            of the operator is the mesh dimension times the number of cells.

        Notes
        -----
        Let :math:`\mathbf{u_f}` be the discrete representation of a vector
        quantity whose Cartesian components are defined on their respective faces;
        e.g. :math:`u_x` is defined on x-faces.
        **average_face_to_cell_vector** constructs a discrete linear operator
        :math:`\mathbf{A_{fc}}` that projects each Cartesian component of
        :math:`\mathbf{u_f}` independently to cell centers, i.e.:

        .. math::
            \mathbf{u_c} = \mathbf{A_{fc}} \, \mathbf{u_f}

        where :math:`\mathbf{u_c}` is a discrete vector quantity whose Cartesian
        components defined at the cell centers and organized into a 1D array of
        the form np.r_[ux, uy, uz]. For each cell, and for each Cartesian component,
        we are simply taking the average of the values
        defined on the cell's corresponding faces and placing the result at
        the cell's center.

        Examples
        --------
        Here we compute the values of a vector function discretized to the mesh faces.
        We then create an averaging operator to approximate the function at cell centers.

        We start by importing the necessary packages and defining a mesh.

        >>> from discretize import TensorMesh
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        >>> h = 0.5 * np.ones(40)
        >>> mesh = TensorMesh([h, h], x0="CC")

        Then we create a discrete vector on mesh faces

        >>> faces_x = mesh.faces_x
        >>> faces_y = mesh.faces_y
        >>> u_fx = -(faces_x[:, 1] / np.sqrt(np.sum(faces_x ** 2, axis=1))) * np.exp(
        >>>     -(faces_x[:, 0] ** 2 + faces_x[:, 1] ** 2) / 6 ** 2
        >>> )
        >>> u_fy = (faces_y[:, 0] / np.sqrt(np.sum(faces_y ** 2, axis=1))) * np.exp(
        >>>     -(faces_y[:, 0] ** 2 + faces_y[:, 1] ** 2) / 6 ** 2
        >>> )
        >>> u_f = np.r_[u_fx, u_fy]

        Next, we construct the averaging operator and apply it to
        the discrete vector quantity to approximate the value at cell centers.

        >>> Afc = mesh.average_face_to_cell_vector
        >>> u_c = Afc @ u_f

        And finally, plot the results:

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(11, 5))
            >>> ax1 = fig.add_subplot(121)
            >>> mesh.plot_image(u_f, ax=ax1, v_type="F", view='vec')
            >>> ax1.set_title("Variable at faces", fontsize=16)
            >>> ax2 = fig.add_subplot(122)
            >>> mesh.plot_image(u_c, ax=ax2, v_type="CCv", view='vec')
            >>> ax2.set_title("Averaged to cell centers", fontsize=16)
            >>> plt.show()

        Below, we show a spy plot illustrating the sparsity and mapping
        of the operator

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(9, 9))
            >>> ax1 = fig.add_subplot(111)
            >>> ax1.spy(Afc, ms=1)
            >>> ax1.set_title("Face Index", fontsize=12, pad=5)
            >>> ax1.set_ylabel("Cell Vector Index", fontsize=12)
            >>> plt.show()
        """
        raise NotImplementedError(
            f"average_face_to_cell_vector not implemented for {type(self)}"
        )

    @property
    def average_cell_to_face(self):
        r"""Averaging operator from cell centers to faces (scalar quantities).

        This property constructs an averaging operator that maps scalar
        quantities from cell centers to face. This averaging operator is
        used when a discrete scalar quantity defined cell centers must be
        projected to faces. Once constructed, the operator is stored
        permanently as a property of the mesh. *See notes*.

        Returns
        -------
        (n_faces, n_cells) scipy.sparse.csr_matrix
            The scalar averaging operator from cell centers to faces

        Notes
        -----
        Let :math:`\boldsymbol{\phi_c}` be a discrete scalar quantity that
        lives at cell centers. **average_cell_to_face** constructs a discrete
        linear operator :math:`\mathbf{A_{cf}}` that projects
        :math:`\boldsymbol{\phi_c}` to faces, i.e.:

        .. math::
            \boldsymbol{\phi_f} = \mathbf{A_{cf}} \, \boldsymbol{\phi_c}

        where :math:`\boldsymbol{\phi_f}` approximates the value of the scalar
        quantity at the faces. For each face, we are performing a weighted average
        between the values at adjacent cell centers. In 1D, where adjacent cells
        :math:`i` and :math:`i+1` have widths :math:`h_i` and :math:`h_{i+1}`,
        :math:`\phi` on face is approximated by:

        .. math::
            \phi_{i \! + \! 1/2} \approx \frac{h_{i+1} \phi_i + h_i \phi_{i+1}}{h_i + h_{i+1}}

        On boundary faces, nearest neighbour is used to extrapolate the value
        from the nearest cell center. Once the operator is construct, the averaging
        is implemented as a matrix vector product, i.e.::

            phi_f = Acf @ phi_c

        Examples
        --------
        Here we compute the values of a scalar function at cell centers. We then create
        an averaging operator to approximate the function on the faces. We choose
        to define a scalar function that is strongly discontinuous in some places to
        demonstrate how the averaging operator will smooth out discontinuities.

        We start by importing the necessary packages and defining a mesh.

        >>> from discretize import TensorMesh
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        >>> h = np.ones(40)
        >>> mesh = TensorMesh([h, h], x0="CC")

        Create a scalar variable at cell centers

        >>> phi_c = np.zeros(mesh.nC)
        >>> xy = mesh.cell_centers
        >>> phi_c[(xy[:, 1] > 0)] = 25.0
        >>> phi_c[(xy[:, 1] < -10.0) & (xy[:, 0] > -10.0) & (xy[:, 0] < 10.0)] = 50.0

        Next, we construct the averaging operator and apply it to
        the discrete scalar quantity to approximate the value at the faces.

        >>> Acf = mesh.average_cell_to_face
        >>> phi_f = Acf @ phi_c

        Plot the results

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(11, 5))
            >>> ax1 = fig.add_subplot(121)
            >>> mesh.plot_image(phi_c, ax=ax1, v_type="CC")
            >>> ax1.set_title("Variable at cell centers", fontsize=16)
            >>> ax2 = fig.add_subplot(122)
            >>> mesh.plot_image(phi_f, ax=ax2, v_type="F")
            >>> ax2.set_title("Averaged to faces", fontsize=16)
            >>> plt.show()

        Below, we show a spy plot illustrating the sparsity and mapping
        of the operator.

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(9, 9))
            >>> ax1 = fig.add_subplot(111)
            >>> ax1.spy(Acf, ms=1)
            >>> ax1.set_title("Cell Index", fontsize=12, pad=5)
            >>> ax1.set_ylabel("Face Index", fontsize=12)
            >>> plt.show()
        """
        raise NotImplementedError(
            f"average_cell_to_face not implemented for {type(self)}"
        )

    @property
    def average_cell_vector_to_face(self):
        r"""Averaging operator from cell centers to faces (vector quantities).

        This property constructs the averaging operator that independently maps the
        Cartesian components of vector quantities from cell centers to faces.
        This averaging operators is used when a discrete vector quantity defined at
        cell centers must be approximated on the faces. Once constructed, the operator is
        stored permanently as a property of the mesh.

        Be aware that the Cartesian components of the original vector
        are defined seperately at cell centers in a 1D numpy.array organized [ux, uy, uz].
        Once projected to faces, the Cartesian components are defined on their respective
        faces; e.g. the x-component lives on x-faces. The operation is implemented as a
        matrix vector product, i.e.::

            u_f = Acf @ u_c

        Returns
        -------
        (n_faces, dim * n_cells) scipy.sparse.csr_matrix
            The vector averaging operator from cell centers to faces. Since we
            are averaging a vector quantity from cell centers, the second dimension
            of the operator is the mesh dimension times the number of cells.

        Notes
        -----
        Let :math:`\mathbf{u_c}` be the discrete representation of a vector
        quantity whose Cartesian components are defined separately at cell centers.
        **average_cell_vector_to_face** constructs a discrete linear operator
        :math:`\mathbf{A_{cf}}` that projects each Cartesian component of
        :math:`\mathbf{u_c}` to the faces, i.e.:

        .. math::
            \mathbf{u_f} = \mathbf{A_{cf}} \, \mathbf{u_c}

        where :math:`\mathbf{u_f}` is the discrete vector quantity whose Cartesian
        components are approximated on their respective cell faces; e.g. the x-component is
        approximated on x-faces. For each face (x, y or z), we are simply taking a weighted average
        between the values of the correct Cartesian component at the corresponding cell centers.

        E.g. for the x-component, which is projected to x-faces, the weighted average on
        a 2D mesh would be:

        .. math::
            u_x(i \! + \! 1/2, j) = \frac{h_{i+1} u_x (i,j) + h_i u_x(i \! + \! 1,j)}{hx_i + hx_{i+1}}

        where :math:`h_i` and :math:`h_{i+1}` represent the cell respective cell widths
        in the x-direction. For boundary faces, nearest neighbor is used to extrapolate
        the values.

        Examples
        --------
        Here we compute the values of a vector function discretized to cell centers.
        We then create an averaging operator to approximate the function on the faces.

        We start by importing the necessary packages and defining a mesh.

        >>> from discretize import TensorMesh
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> h = 0.5 * np.ones(40)
        >>> mesh = TensorMesh([h, h], x0="CC")

        Then we create a discrete vector at cell centers,

        >>> centers = mesh.cell_centers
        >>> u_x = -(centers[:, 1] / np.sqrt(np.sum(centers ** 2, axis=1))) * np.exp(
        ...     -(centers[:, 0] ** 2 + centers[:, 1] ** 2) / 6 ** 2
        ... )
        >>> u_y = (centers[:, 0] / np.sqrt(np.sum(centers ** 2, axis=1))) * np.exp(
        ...     -(centers[:, 0] ** 2 + centers[:, 1] ** 2) / 6 ** 2
        ... )
        >>> u_c = np.r_[u_x, u_y]

        Next, we construct the averaging operator and apply it to
        the discrete vector quantity to approximate the value on the faces.

        >>> Acf = mesh.average_cell_vector_to_face
        >>> u_f = Acf @ u_c

        And plot the results

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(11, 5))
            >>> ax1 = fig.add_subplot(121)
            >>> mesh.plot_image(u_c, ax=ax1, v_type="CCv", view='vec')
            >>> ax1.set_title("Variable at faces", fontsize=16)
            >>> ax2 = fig.add_subplot(122)
            >>> mesh.plot_image(u_f, ax=ax2, v_type="F", view='vec')
            >>> ax2.set_title("Averaged to cell centers", fontsize=16)
            >>> plt.show()

        Below, we show a spy plot illustrating the sparsity and mapping
        of the operator

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(9, 9))
            >>> ax1 = fig.add_subplot(111)
            >>> ax1.spy(Acf, ms=1)
            >>> ax1.set_title("Cell Vector Index", fontsize=12, pad=5)
            >>> ax1.set_ylabel("Face Index", fontsize=12)
            >>> plt.show()
        """
        raise NotImplementedError(
            f"average_cell_vector_to_face not implemented for {type(self)}"
        )

    @property
    def average_cell_to_edge(self):
        r"""Averaging operator from cell centers to edges (scalar quantities).

        This property constructs an averaging operator that maps scalar
        quantities from cell centers to edge. This averaging operator is
        used when a discrete scalar quantity defined cell centers must be
        projected to edges. Once constructed, the operator is stored
        permanently as a property of the mesh. *See notes*.

        Returns
        -------
        (n_edges, n_cells) scipy.sparse.csr_matrix
            The scalar averaging operator from edges to cell centers

        Notes
        -----
        Let :math:`\boldsymbol{\phi_c}` be a discrete scalar quantity that
        lives at cell centers. **average_cell_to_edge** constructs a discrete
        linear operator :math:`\mathbf{A_{ce}}` that projects
        :math:`\boldsymbol{\phi_c}` to edges, i.e.:

        .. math::
            \boldsymbol{\phi_e} = \mathbf{A_{ce}} \, \boldsymbol{\phi_c}

        where :math:`\boldsymbol{\phi_e}` approximates the value of the scalar
        quantity at the edges. For each edge, we are performing a weighted average
        between the values at adjacent cell centers. In 1D, where adjacent cells
        :math:`i` and :math:`i+1` have widths :math:`h_i` and :math:`h_{i+1}`,
        :math:`\phi` on edge (node location in 1D) is approximated by:

        .. math::
            \phi_{i \! + \! 1/2} \approx \frac{h_{i+1} \phi_i + h_i \phi_{i+1}}{h_i + h_{i+1}}

        On boundary edges, nearest neighbour is used to extrapolate the value
        from the nearest cell center. Once the operator is construct, the averaging
        is implemented as a matrix vector product, i.e.::

            phi_e = Ace @ phi_c

        Examples
        --------
        Here we compute the values of a scalar function at cell centers. We then create
        an averaging operator to approximate the function on the edges. We choose
        to define a scalar function that is strongly discontinuous in some places to
        demonstrate how the averaging operator will smooth out discontinuities.

        We start by importing the necessary packages and defining a mesh.

        >>> from discretize import TensorMesh
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> h = np.ones(40)
        >>> mesh = TensorMesh([h, h], x0="CC")

        Then we create a scalar variable at cell centers

        >>> phi_c = np.zeros(mesh.nC)
        >>> xy = mesh.cell_centers
        >>> phi_c[(xy[:, 1] > 0)] = 25.0
        >>> phi_c[(xy[:, 1] < -10.0) & (xy[:, 0] > -10.0) & (xy[:, 0] < 10.0)] = 50.0

        Next, we construct the averaging operator and apply it to
        the discrete scalar quantity to approximate the value at the edges.

        >>> Ace = mesh.average_cell_to_edge
        >>> phi_e = Ace @ phi_c

        And plot the results:

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(11, 5))
            >>> ax1 = fig.add_subplot(121)
            >>> mesh.plot_image(phi_c, ax=ax1, v_type="CC")
            >>> ax1.set_title("Variable at cell centers", fontsize=16)
            >>> ax2 = fig.add_subplot(122)
            >>> mesh.plot_image(phi_e, ax=ax2, v_type="E")
            >>> ax2.set_title("Averaged to edges", fontsize=16)
            >>> plt.show()

        Below, we show a spy plot illustrating the sparsity and mapping
        of the operator.

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(9, 9))
            >>> ax1 = fig.add_subplot(111)
            >>> ax1.spy(Ace, ms=1)
            >>> ax1.set_title("Cell Index", fontsize=12, pad=5)
            >>> ax1.set_ylabel("Edge Index", fontsize=12)
            >>> plt.show()
        """
        raise NotImplementedError(
            f"average_cell_to_edge not implemented for {type(self)}"
        )

    @property
    def average_edge_to_cell(self):
        r"""Averaging operator from edges to cell centers (scalar quantities).

        This property constructs a 2nd order averaging operator that maps scalar
        quantities from edges to cell centers. This averaging operator is
        used when a discrete scalar quantity defined on mesh edges must be
        projected to cell centers. Once constructed, the operator is stored
        permanently as a property of the mesh. *See notes*.

        Returns
        -------
        (n_cells, n_edges) scipy.sparse.csr_matrix
            The scalar averaging operator from edges to cell centers

        Notes
        -----
        Let :math:`\boldsymbol{\phi_e}` be a discrete scalar quantity that
        lives on mesh edges. **average_edge_to_cell** constructs a discrete
        linear operator :math:`\mathbf{A_{ec}}` that projects
        :math:`\boldsymbol{\phi_e}` to cell centers, i.e.:

        .. math::
            \boldsymbol{\phi_c} = \mathbf{A_{ec}} \, \boldsymbol{\phi_e}

        where :math:`\boldsymbol{\phi_c}` approximates the value of the scalar
        quantity at cell centers. For each cell, we are simply averaging
        the values defined on its edges. The operation is implemented as a
        matrix vector product, i.e.::

            phi_c = Aec @ phi_e

        Examples
        --------
        Here we compute the values of a scalar function on the edges. We then create
        an averaging operator to approximate the function at cell centers. We choose
        to define a scalar function that is strongly discontinuous in some places to
        demonstrate how the averaging operator will smooth out discontinuities.

        We start by importing the necessary packages and defining a mesh.

        >>> from discretize import TensorMesh
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> h = np.ones(40)
        >>> mesh = TensorMesh([h, h], x0="CC")

        Then we create a scalar variable on edges,

        >>> phi_e = np.zeros(mesh.nE)
        >>> xy = mesh.edges
        >>> phi_e[(xy[:, 1] > 0)] = 25.0
        >>> phi_e[(xy[:, 1] < -10.0) & (xy[:, 0] > -10.0) & (xy[:, 0] < 10.0)] = 50.0

        Next, we construct the averaging operator and apply it to
        the discrete scalar quantity to approximate the value at cell centers.

        >>> Aec = mesh.average_edge_to_cell
        >>> phi_c = Aec @ phi_e

        And plot the results:

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(11, 5))
            >>> ax1 = fig.add_subplot(121)
            >>> mesh.plot_image(phi_e, ax=ax1, v_type="E")
            >>> ax1.set_title("Variable at edges", fontsize=16)
            >>> ax2 = fig.add_subplot(122)
            >>> mesh.plot_image(phi_c, ax=ax2, v_type="CC")
            >>> ax2.set_title("Averaged to cell centers", fontsize=16)
            >>> plt.show()

        Below, we show a spy plot illustrating the sparsity and mapping
        of the operator

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(9, 9))
            >>> ax1 = fig.add_subplot(111)
            >>> ax1.spy(Aec, ms=1)
            >>> ax1.set_title("Edge Index", fontsize=12, pad=5)
            >>> ax1.set_ylabel("Cell Index", fontsize=12)
            >>> plt.show()
        """
        raise NotImplementedError(
            f"average_edge_to_cell not implemented for {type(self)}"
        )

    @property
    def average_edge_to_cell_vector(self):
        r"""Averaging operator from edges to cell centers (vector quantities).

        This property constructs the averaging operator that independently maps the
        Cartesian components of vector quantities from edges to cell centers.
        This averaging operators is used when a discrete vector quantity defined on mesh edges
        must be approximated at cell centers. Once constructed, the operator is
        stored permanently as a property of the mesh.

        Be aware that the Cartesian components of the original vector
        are defined on their respective edges; e.g. the x-component lives
        on x-edges. However, the x, y and z components are being averaged
        separately to cell centers. The operation is implemented as a
        matrix vector product, i.e.::

            u_c = Aec @ u_e

        Returns
        -------
        (dim * n_cells, n_edges) scipy.sparse.csr_matrix
            The vector averaging operator from edges to cell centers. Since we
            are averaging a vector quantity to cell centers, the first dimension
            of the operator is the mesh dimension times the number of cells.

        Notes
        -----
        Let :math:`\mathbf{u_e}` be the discrete representation of a vector
        quantity whose Cartesian components are defined on their respective edges;
        e.g. :math:`u_x` is defined on x-edges.
        **average_edge_to_cell_vector** constructs a discrete linear operator
        :math:`\mathbf{A_{ec}}` that projects each Cartesian component of
        :math:`\mathbf{u_e}` independently to cell centers, i.e.:

        .. math::
            \mathbf{u_c} = \mathbf{A_{ec}} \, \mathbf{u_e}

        where :math:`\mathbf{u_c}` is a discrete vector quantity whose Cartesian
        components defined at the cell centers and organized into a 1D array of
        the form np.r_[ux, uy, uz]. For each cell, and for each Cartesian component,
        we are simply taking the average of the values
        defined on the cell's corresponding edges and placing the result at
        the cell's center.

        Examples
        --------
        Here we compute the values of a vector function discretized to the edges.
        We then create an averaging operator to approximate the function at cell centers.

        We start by importing the necessary packages and defining a mesh.

        >>> from discretize import TensorMesh
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> h = 0.5 * np.ones(40)
        >>> mesh = TensorMesh([h, h], x0="CC")

        Then we create a discrete vector on mesh edges

        >>> edges_x = mesh.edges_x
        >>> edges_y = mesh.edges_y
        >>> u_ex = -(edges_x[:, 1] / np.sqrt(np.sum(edges_x ** 2, axis=1))) * np.exp(
        ...     -(edges_x[:, 0] ** 2 + edges_x[:, 1] ** 2) / 6 ** 2
        ... )
        >>> u_ey = (edges_y[:, 0] / np.sqrt(np.sum(edges_y ** 2, axis=1))) * np.exp(
        ...     -(edges_y[:, 0] ** 2 + edges_y[:, 1] ** 2) / 6 ** 2
        ... )
        >>> u_e = np.r_[u_ex, u_ey]

        Next, we construct the averaging operator and apply it to
        the discrete vector quantity to approximate the value at cell centers.

        >>> Aec = mesh.average_edge_to_cell_vector
        >>> u_c = Aec @ u_e

        And plot the results:

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(11, 5))
            >>> ax1 = fig.add_subplot(121)
            >>> mesh.plot_image(u_e, ax=ax1, v_type="E", view='vec')
            >>> ax1.set_title("Variable at edges", fontsize=16)
            >>> ax2 = fig.add_subplot(122)
            >>> mesh.plot_image(u_c, ax=ax2, v_type="CCv", view='vec')
            >>> ax2.set_title("Averaged to cell centers", fontsize=16)
            >>> plt.show()

        Below, we show a spy plot illustrating the sparsity and mapping
        of the operator

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(9, 9))
            >>> ax1 = fig.add_subplot(111)
            >>> ax1.spy(Aec, ms=1)
            >>> ax1.set_title("Edge Index", fontsize=12, pad=5)
            >>> ax1.set_ylabel("Cell Vector Index", fontsize=12)
            >>> plt.show()
        """
        raise NotImplementedError(
            f"average_edge_to_cell_vector not implemented for {type(self)}"
        )

    @property
    def average_edge_to_face_vector(self):
        r"""Averaging operator from edges to faces (vector quantities).

        This property constructs the averaging operator that independently maps the
        Cartesian components of vector quantities from edges to faces.
        This averaging operators is used when a discrete vector quantity defined on mesh edges
        must be approximated at faces. The operation is implemented as a
        matrix vector product, i.e.::

            u_f = Aef @ u_e

        Once constructed, the operator is stored permanently as a property of the mesh.

        Returns
        -------
        (n_faces, n_edges) scipy.sparse.csr_matrix
            The vector averaging operator from edges to faces.

        Notes
        -----
        Let :math:`\mathbf{u_e}` be the discrete representation of a vector
        quantity whose Cartesian components are defined on their respective edges;
        e.g. the x-component is defined on x-edges. **average_edge_to_face_vector**
        constructs a discrete linear operator :math:`\mathbf{A_{ef}}` that
        projects each Cartesian component of :math:`\mathbf{u_e}` to
        its corresponding face, i.e.:

        .. math::
            \mathbf{u_f} = \mathbf{A_{ef}} \, \mathbf{u_e}

        where :math:`\mathbf{u_f}` is a discrete vector quantity whose Cartesian
        components are defined on their respective faces; e.g. the x-component is
        defined on x-faces.

        Examples
        --------
        Here we compute the values of a vector function discretized to the edges.
        We then create an averaging operator to approximate the function on
        the faces.

        We start by importing the necessary packages and defining a mesh.

        >>> from discretize import TensorMesh
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> h = 0.5 * np.ones(40)
        >>> mesh = TensorMesh([h, h], x0="CC")

        Create a discrete vector on mesh edges

        >>> edges_x = mesh.edges_x
        >>> edges_y = mesh.edges_y
        >>> u_ex = -(edges_x[:, 1] / np.sqrt(np.sum(edges_x ** 2, axis=1))) * np.exp(
        ...     -(edges_x[:, 0] ** 2 + edges_x[:, 1] ** 2) / 6 ** 2
        ... )
        >>> u_ey = (edges_y[:, 0] / np.sqrt(np.sum(edges_y ** 2, axis=1))) * np.exp(
        ...     -(edges_y[:, 0] ** 2 + edges_y[:, 1] ** 2) / 6 ** 2
        ... )
        >>> u_e = np.r_[u_ex, u_ey]

        Next, we construct the averaging operator and apply it to
        the discrete vector quantity to approximate the value at the faces.

        >>> Aef = mesh.average_edge_to_face_vector
        >>> u_f = Aef @ u_e

        Plot the results,

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(11, 5))
            >>> ax1 = fig.add_subplot(121)
            >>> mesh.plot_image(u_e, ax=ax1, v_type="E", view='vec')
            >>> ax1.set_title("Variable at edges", fontsize=16)
            >>> ax2 = fig.add_subplot(122)
            >>> mesh.plot_image(u_f, ax=ax2, v_type="F", view='vec')
            >>> ax2.set_title("Averaged to faces", fontsize=16)
            >>> plt.show()

        Below, we show a spy plot illustrating the sparsity and mapping
        of the operator

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(9, 9))
            >>> ax1 = fig.add_subplot(111)
            >>> ax1.spy(Aef, ms=1)
            >>> ax1.set_title("Edge Index", fontsize=12, pad=5)
            >>> ax1.set_ylabel("Face Index", fontsize=12)
            >>> plt.show()
        """
        raise NotImplementedError(
            f"average_edge_to_face_vector not implemented for {type(self)}"
        )

    @property
    def average_node_to_cell(self):
        r"""Averaging operator from nodes to cell centers (scalar quantities).

        This property constructs a 2nd order averaging operator that maps scalar
        quantities from nodes to cell centers. This averaging operator is
        used when a discrete scalar quantity defined on mesh nodes must be
        projected to cell centers. Once constructed, the operator is stored
        permanently as a property of the mesh. *See notes*.

        Returns
        -------
        (n_cells, n_nodes) scipy.sparse.csr_matrix
            The scalar averaging operator from nodes to cell centers

        Notes
        -----
        Let :math:`\boldsymbol{\phi_n}` be a discrete scalar quantity that
        lives on mesh nodes. **average_node_to_cell** constructs a discrete
        linear operator :math:`\mathbf{A_{nc}}` that projects
        :math:`\boldsymbol{\phi_f}` to cell centers, i.e.:

        .. math::
            \boldsymbol{\phi_c} = \mathbf{A_{nc}} \, \boldsymbol{\phi_n}

        where :math:`\boldsymbol{\phi_c}` approximates the value of the scalar
        quantity at cell centers. For each cell, we are simply averaging
        the values defined on its nodes. The operation is implemented as a
        matrix vector product, i.e.::

            phi_c = Anc @ phi_n

        Examples
        --------
        Here we compute the values of a scalar function on the nodes. We then create
        an averaging operator to approximate the function at cell centers. We choose
        to define a scalar function that is strongly discontinuous in some places to
        demonstrate how the averaging operator will smooth out discontinuities.

        We start by importing the necessary packages and defining a mesh.

        >>> from discretize import TensorMesh
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> h = np.ones(40)
        >>> mesh = TensorMesh([h, h], x0="CC")

        Then we Create a scalar variable on nodes

        >>> phi_n = np.zeros(mesh.nN)
        >>> xy = mesh.nodes
        >>> phi_n[(xy[:, 1] > 0)] = 25.0
        >>> phi_n[(xy[:, 1] < -10.0) & (xy[:, 0] > -10.0) & (xy[:, 0] < 10.0)] = 50.0

        Next, we construct the averaging operator and apply it to
        the discrete scalar quantity to approximate the value at cell centers.

        >>> Anc = mesh.average_node_to_cell
        >>> phi_c = Anc @ phi_n

        Plot the results,

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(11, 5))
            >>> ax1 = fig.add_subplot(121)
            >>> mesh.plot_image(phi_n, ax=ax1, v_type="N")
            >>> ax1.set_title("Variable at nodes", fontsize=16)
            >>> ax2 = fig.add_subplot(122)
            >>> mesh.plot_image(phi_c, ax=ax2, v_type="CC")
            >>> ax2.set_title("Averaged to cell centers", fontsize=16)
            >>> plt.show()

        Below, we show a spy plot illustrating the sparsity and mapping
        of the operator

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(9, 9))
            >>> ax1 = fig.add_subplot(111)
            >>> ax1.spy(Anc, ms=1)
            >>> ax1.set_title("Node Index", fontsize=12, pad=5)
            >>> ax1.set_ylabel("Cell Index", fontsize=12)
            >>> plt.show()
        """
        raise NotImplementedError(
            f"average_node_to_cell not implemented for {type(self)}"
        )

    @property
    def average_node_to_edge(self):
        r"""Averaging operator from nodes to edges (scalar quantities).

        This property constructs a 2nd order averaging operator that maps scalar
        quantities from nodes to edges; scalar at edges is organized in a 1D numpy.array
        of the form [x-edges, y-edges, z-edges]. This averaging operator is
        used when a discrete scalar quantity defined on mesh nodes must be
        projected to edges. Once constructed, the operator is stored
        permanently as a property of the mesh. *See notes*.

        Returns
        -------
        (n_edges, n_nodes) scipy.sparse.csr_matrix
            The scalar averaging operator from nodes to edges

        Notes
        -----
        Let :math:`\boldsymbol{\phi_n}` be a discrete scalar quantity that
        lives on mesh nodes. **average_node_to_edge** constructs a discrete
        linear operator :math:`\mathbf{A_{ne}}` that projects
        :math:`\boldsymbol{\phi_n}` to edges, i.e.:

        .. math::
            \boldsymbol{\phi_e} = \mathbf{A_{ne}} \, \boldsymbol{\phi_n}

        where :math:`\boldsymbol{\phi_e}` approximates the value of the scalar
        quantity at edges. For each edge, we are simply averaging
        the values defined on the nodes it connects. The operation is implemented as a
        matrix vector product, i.e.::

            phi_e = Ane @ phi_n

        Examples
        --------
        Here we compute the values of a scalar function on the nodes. We then create
        an averaging operator to approximate the function at the edges. We choose
        to define a scalar function that is strongly discontinuous in some places to
        demonstrate how the averaging operator will smooth out discontinuities.

        We start by importing the necessary packages and defining a mesh.

        >>> from discretize import TensorMesh
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> h = np.ones(40)
        >>> mesh = TensorMesh([h, h], x0="CC")

        Then we create a scalar variable on nodes,

        >>> phi_n = np.zeros(mesh.nN)
        >>> xy = mesh.nodes
        >>> phi_n[(xy[:, 1] > 0)] = 25.0
        >>> phi_n[(xy[:, 1] < -10.0) & (xy[:, 0] > -10.0) & (xy[:, 0] < 10.0)] = 50.0

        Next, we construct the averaging operator and apply it to
        the discrete scalar quantity to approximate the value on the edges.

        >>> Ane = mesh.average_node_to_edge
        >>> phi_e = Ane @ phi_n

        Plot the results,

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(11, 5))
            >>> ax1 = fig.add_subplot(121)
            >>> mesh.plot_image(phi_n, ax=ax1, v_type="N")
            >>> ax1.set_title("Variable at nodes")
            >>> ax2 = fig.add_subplot(122)
            >>> mesh.plot_image(phi_e, ax=ax2, v_type="E")
            >>> ax2.set_title("Averaged to edges")
            >>> plt.show()

        Below, we show a spy plot illustrating the sparsity and mapping
        of the operator

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(9, 9))
            >>> ax1 = fig.add_subplot(111)
            >>> ax1.spy(Ane, ms=1)
            >>> ax1.set_title("Node Index", fontsize=12, pad=5)
            >>> ax1.set_ylabel("Edge Index", fontsize=12)
            >>> plt.show()
        """
        raise NotImplementedError(
            f"average_node_to_edge not implemented for {type(self)}"
        )

    @property
    def average_node_to_face(self):
        r"""Averaging operator from nodes to faces (scalar quantities).

        This property constructs a 2nd order averaging operator that maps scalar
        quantities from nodes to edges; scalar at faces is organized in a 1D numpy.array
        of the form [x-faces, y-faces, z-faces]. This averaging operator is
        used when a discrete scalar quantity defined on mesh nodes must be
        projected to faces. Once constructed, the operator is stored
        permanently as a property of the mesh. *See notes*.

        Returns
        -------
        (n_faces, n_nodes) scipy.sparse.csr_matrix
            The scalar averaging operator from nodes to faces

        Notes
        -----
        Let :math:`\boldsymbol{\phi_n}` be a discrete scalar quantity that
        lives on mesh nodes. **average_node_to_face** constructs a discrete
        linear operator :math:`\mathbf{A_{nf}}` that projects
        :math:`\boldsymbol{\phi_n}` to faces, i.e.:

        .. math::
            \boldsymbol{\phi_f} = \mathbf{A_{nf}} \, \boldsymbol{\phi_n}

        where :math:`\boldsymbol{\phi_f}` approximates the value of the scalar
        quantity at faces. For each face, we are simply averaging the values at
        the nodes which outline the face. The operation is implemented as a
        matrix vector product, i.e.::

            phi_f = Anf @ phi_n

        Examples
        --------
        Here we compute the values of a scalar function on the nodes. We then create
        an averaging operator to approximate the function at the faces. We choose
        to define a scalar function that is strongly discontinuous in some places to
        demonstrate how the averaging operator will smooth out discontinuities.

        We start by importing the necessary packages and defining a mesh.

        >>> from discretize import TensorMesh
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> h = np.ones(40)
        >>> mesh = TensorMesh([h, h], x0="CC")

        Then we, create a scalar variable on nodes

        >>> phi_n = np.zeros(mesh.nN)
        >>> xy = mesh.nodes
        >>> phi_n[(xy[:, 1] > 0)] = 25.0
        >>> phi_n[(xy[:, 1] < -10.0) & (xy[:, 0] > -10.0) & (xy[:, 0] < 10.0)] = 50.0

        Next, we construct the averaging operator and apply it to
        the discrete scalar quantity to approximate the value on the faces.

        >>> Anf = mesh.average_node_to_face
        >>> phi_f = Anf @ phi_n

        Plot the results,

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(11, 5))
            >>> ax1 = fig.add_subplot(121)
            >>> mesh.plot_image(phi_n, ax=ax1, v_type="N")
            >>> ax1.set_title("Variable at nodes")
            >>> ax2 = fig.add_subplot(122)
            >>> mesh.plot_image(phi_f, ax=ax2, v_type="F")
            >>> ax2.set_title("Averaged to faces")
            >>> plt.show()

        Below, we show a spy plot illustrating the sparsity and mapping
        of the operator

        .. collapse:: Expand to show scripting for plot

            >>> fig = plt.figure(figsize=(9, 9))
            >>> ax1 = fig.add_subplot(111)
            >>> ax1.spy(Anf, ms=1)
            >>> ax1.set_title("Node Index", fontsize=12, pad=5)
            >>> ax1.set_ylabel("Face Index", fontsize=12)
            >>> plt.show()
        """
        raise NotImplementedError(
            f"average_node_to_face not implemented for {type(self)}"
        )

    @property
    def project_face_to_boundary_face(self):
        r"""Projection matrix from all faces to boundary faces.

        Constructs and returns a matrix :math:`\mathbf{P}` that projects from
        all mesh faces to boundary faces. That is, for a discrete vector
        :math:`\mathbf{u}` that lives on the faces, the values on the boundary
        faces :math:`\mathbf{u_b}` can be extracted via the following
        matrix-vector product::

            ub = P @ u

        Returns
        -------
        scipy.sparse.csr_matrix
            (n_boundary_faces, n_faces) Projection matrix with shape
        """
        raise NotImplementedError(
            f"project_face_to_boundary_face not implemented for {type(self)}"
        )

    @property
    def project_edge_to_boundary_edge(self):
        r"""Projection matrix from all edges to boundary edges.

        Constructs and returns a matrix :math:`\mathbf{P}` that projects from
        all mesh edges to boundary edges. That is, for a discrete vector
        :math:`\mathbf{u}` that lives on the edges, the values on the boundary
        edges :math:`\mathbf{u_b}` can be extracted via the following
        matrix-vector product::

            ub = P @ u

        Returns
        -------
        (n_boundary_edges, n_edges) scipy.sparse.csr_matrix
            Projection matrix with shape
        """
        raise NotImplementedError(
            f"project_edge_to_boundary_edge not implemented for {type(self)}"
        )

    @property
    def project_node_to_boundary_node(self):
        r"""Projection matrix from all nodes to boundary nodes.

        Constructs and returns a matrix :math:`\mathbf{P}` that projects from
        all mesh nodes to boundary nodes. That is, for a discrete scalar
        :math:`\mathbf{u}` that lives on the nodes, the values on the boundary
        nodes :math:`\mathbf{u_b}` can be extracted via the following
        matrix-vector product::

            ub = P @ u

        Returns
        -------
        (n_boundary_nodes, n_nodes) scipy.sparse.csr_matrix
            Projection matrix with shape
        """
        raise NotImplementedError(
            f"project_node_to_boundary_node not implemented for {type(self)}"
        )

    def closest_points_index(self, locations, grid_loc="CC", discard=False):
        """Find the indicies for the nearest grid location for a set of points.

        Parameters
        ----------
        locations : (n, dim) numpy.ndarray
            Points to query.
        grid_loc : {'CC', 'N', 'Fx', 'Fy', 'Fz', 'Ex', 'Ex', 'Ey', 'Ez'}
            Specifies the grid on which points are being moved to.
        discard : bool, optional
            Whether to discard the intenally created `scipy.spatial.KDTree`.

        Returns
        -------
        (n ) numpy.ndarray of int
            Vector of length *n* containing the indicies for the closest
            respective cell center, node, face or edge.

        Examples
        --------
        Here we define a set of random (x, y) locations and find the closest
        cell centers and nodes on a mesh.

        >>> from discretize import TensorMesh
        >>> from discretize.utils import closest_points_index
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt
        >>> h = 2*np.ones(5)
        >>> mesh = TensorMesh([h, h], x0='00')

        Define some random locations, grid cell centers and grid nodes,

        >>> xy_random = np.random.uniform(0, 10, size=(4,2))
        >>> xy_centers = mesh.cell_centers
        >>> xy_nodes = mesh.nodes

        Find indicies of closest cell centers and nodes,

        >>> ind_centers = mesh.closest_points_index(xy_random, 'cell_centers')
        >>> ind_nodes = mesh.closest_points_index(xy_random, 'nodes')

        Plot closest cell centers and nodes

        >>> fig = plt.figure(figsize=(5, 5))
        >>> ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        >>> mesh.plot_grid(ax=ax)
        >>> ax.scatter(xy_random[:, 0], xy_random[:, 1], 50, 'k')
        >>> ax.scatter(xy_centers[ind_centers, 0], xy_centers[ind_centers, 1], 50, 'r')
        >>> ax.scatter(xy_nodes[ind_nodes, 0], xy_nodes[ind_nodes, 1], 50, 'b')
        >>> plt.show()
        """
        locations = as_array_n_by_dim(locations, self.dim)

        grid_loc = self._parse_location_type(grid_loc)
        tree_name = f"_{grid_loc}_tree"

        tree = getattr(self, tree_name, None)
        if tree is None:
            grid = as_array_n_by_dim(getattr(self, grid_loc), self.dim)
            tree = KDTree(grid)
        _, ind = tree.query(locations)

        if not discard:
            setattr(self, tree_name, tree)

        return ind

    def point2index(self, locs):
        """Find cells that contain the given points.

        Returns an array of index values of the cells that contain the given
        points

        Parameters
        ----------
        locs: (N, dim) array_like
            points to search for the location of

        Returns
        -------
        (N) array_like of int
            Cell indices that contain the points
        """
        raise NotImplementedError(f"point2index not implemented for {type(self)}")

    def get_interpolation_matrix(
        self, loc, location_type="cell_centers", zeros_outside=False, **kwargs
    ):
        """Construct a linear interpolation matrix from mesh.

        This method constructs a linear interpolation matrix from tensor locations
        (nodes, cell-centers, faces, etc...) on the mesh to a set of arbitrary locations.

        Parameters
        ----------
        loc : (n_pts, dim) numpy.ndarray
            Location of points being to interpolate to. Must have same dimensions as the mesh.
        location_type : str, optional
            Tensor locations on the mesh being interpolated from. *location_type* must be one of:

            - 'Ex', 'edges_x'           -> x-component of field defined on x edges
            - 'Ey', 'edges_y'           -> y-component of field defined on y edges
            - 'Ez', 'edges_z'           -> z-component of field defined on z edges
            - 'Fx', 'faces_x'           -> x-component of field defined on x faces
            - 'Fy', 'faces_y'           -> y-component of field defined on y faces
            - 'Fz', 'faces_z'           -> z-component of field defined on z faces
            - 'N', 'nodes'              -> scalar field defined on nodes
            - 'CC', 'cell_centers'      -> scalar field defined on cell centers
            - 'CCVx', 'cell_centers_x'  -> x-component of vector field defined on cell centers
            - 'CCVy', 'cell_centers_y'  -> y-component of vector field defined on cell centers
            - 'CCVz', 'cell_centers_z'  -> z-component of vector field defined on cell centers
        zeros_outside : bool, optional
            If *False*, nearest neighbour is used to compute the interpolate value
            at locations outside the mesh. If *True* , values at locations outside
            the mesh will be zero.

        Returns
        -------
        (n_pts, n_loc_type) scipy.sparse.csr_matrix
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
        >>> np.random.seed(14)

        >>> locs = np.random.rand(50)*0.8+0.1
        >>> dense = np.linspace(0, 1, 200)
        >>> fun = lambda x: np.cos(2*np.pi*x)

        >>> hx = 0.125 * np.ones(8)
        >>> mesh1D = TensorMesh([hx])
        >>> Q = mesh1D.get_interpolation_matrix(locs, 'nodes')

        .. collapse:: Expand to see scripting for plot

            >>> plt.figure(figsize=(5, 3))
            >>> plt.plot(dense, fun(dense), ':', c="C0", lw=3, label="True Function")
            >>> plt.plot(mesh1D.nodes, fun(mesh1D.nodes), 's', c="C0", ms=8, label="True sampled")
            >>> plt.plot(locs, Q*fun(mesh1D.nodes), 'o', ms=4, label="Interpolated")
            >>> plt.legend()
            >>> plt.show()

        Here, demonstrate a similar example on a 2D mesh using a 2D Gaussian distribution.
        We interpolate the Gaussian from the nodes to cell centers and examine the relative
        error.

        >>> hx = np.ones(10)
        >>> hy = np.ones(10)
        >>> mesh2D = TensorMesh([hx, hy], x0='CC')
        >>> def fun(x, y):
        ...     return np.exp(-(x**2 + y**2)/2**2)

        >>> nodes = mesh2D.nodes
        >>> val_nodes = fun(nodes[:, 0], nodes[:, 1])
        >>> centers = mesh2D.cell_centers
        >>> val_centers = fun(centers[:, 0], centers[:, 1])
        >>> A = mesh2D.get_interpolation_matrix(centers, 'nodes')
        >>> val_interp = A.dot(val_nodes)

        .. collapse:: Expand to see scripting for plot

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
        raise NotImplementedError(
            f"get_interpolation_matrix not implemented for {type(self)}"
        )

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

    # DEPRECATED
    normals = deprecate_property(
        "face_normals", "normals", removal_version="1.0.0", future_warn=True
    )
    tangents = deprecate_property(
        "edge_tangents", "tangents", removal_version="1.0.0", future_warn=True
    )
    projectEdgeVector = deprecate_method(
        "project_edge_vector",
        "projectEdgeVector",
        removal_version="1.0.0",
        future_warn=True,
    )
    projectFaceVector = deprecate_method(
        "project_face_vector",
        "projectFaceVector",
        removal_version="1.0.0",
        future_warn=True,
    )
    getInterpolationMat = deprecate_method(
        "get_interpolation_matrix",
        "getInterpolationMat",
        removal_version="1.0.0",
        future_warn=True,
    )
    nodalGrad = deprecate_property(
        "nodal_gradient", "nodalGrad", removal_version="1.0.0", future_warn=True
    )
    nodalLaplacian = deprecate_property(
        "nodal_laplacian", "nodalLaplacian", removal_version="1.0.0", future_warn=True
    )
    faceDiv = deprecate_property(
        "face_divergence", "faceDiv", removal_version="1.0.0", future_warn=True
    )
    edgeCurl = deprecate_property(
        "edge_curl", "edgeCurl", removal_version="1.0.0", future_warn=True
    )
    getFaceInnerProduct = deprecate_method(
        "get_face_inner_product",
        "getFaceInnerProduct",
        removal_version="1.0.0",
        future_warn=True,
    )
    getEdgeInnerProduct = deprecate_method(
        "get_edge_inner_product",
        "getEdgeInnerProduct",
        removal_version="1.0.0",
        future_warn=True,
    )
    getFaceInnerProductDeriv = deprecate_method(
        "get_face_inner_product_deriv",
        "getFaceInnerProductDeriv",
        removal_version="1.0.0",
        future_warn=True,
    )
    getEdgeInnerProductDeriv = deprecate_method(
        "get_edge_inner_product_deriv",
        "getEdgeInnerProductDeriv",
        removal_version="1.0.0",
        future_warn=True,
    )
    vol = deprecate_property(
        "cell_volumes", "vol", removal_version="1.0.0", future_warn=True
    )
    area = deprecate_property(
        "face_areas", "area", removal_version="1.0.0", future_warn=True
    )
    edge = deprecate_property(
        "edge_lengths", "edge", removal_version="1.0.0", future_warn=True
    )
