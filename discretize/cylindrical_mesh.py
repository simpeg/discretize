import numpy as np
import scipy.sparse as sp
from scipy.constants import pi

from discretize.utils import (
    kron3,
    ndgrid,
    av,
    speye,
    ddx,
    sdiag,
    spzeros,
    interpolation_matrix,
    cyl2cart,
)
from discretize.base import BaseTensorMesh, BaseRectangularMesh
from discretize.operators import DiffOperators, InnerProducts
from discretize.mixins import InterfaceMixins
from discretize.utils.code_utils import (
    deprecate_class,
    deprecate_property,
    deprecate_method,
)
import warnings


class CylindricalMesh(
    BaseTensorMesh, BaseRectangularMesh, InnerProducts, DiffOperators, InterfaceMixins
):
    """
    Class for cylindrical meshes.

    ``CylindricalMesh`` is a mesh class for problems with rotational symmetry.
    It supports both cylindrically symmetric and 3D cylindrical meshes where the azimuthal
    discretization is user-defined. In discretize, the coordinates for cylindrical
    meshes are as follows:

    - **x:** radial direction (:math:`r`)
    - **y:** azimuthal direction (:math:`\\phi`)
    - **z:** vertical direction (:math:`z`)

    Parameters
    ----------
    h : (dim) iterable of int, numpy.ndarray, or tuple
        Defines the cell widths along each axis. The length of the iterable object is
        equal to the dimension of the mesh (1, 2 or 3). For a 3D mesh, the list would
        have the form *[hr, hphi, hz]* . Note that the sum of cell widths in the phi
        direction **must** equal :math:`2\\pi`. You can also use a flat value of
        *hphi* = *1* to define a cylindrically symmetric mesh.

        Along each axis, the user has 3 choices for defining the cells widths:

        - :class:`int` -> A unit interval is equally discretized into `N` cells.
        - :class:`numpy.ndarray` -> The widths are explicity given for each cell
        - the widths are defined as a :class:`list` of :class:`tuple` of the form
          *(dh, nc, [npad])* where *dh* is the cell width, *nc* is the number of cells,
          and *npad* (optional)is a padding factor denoting exponential
          increase/decrease in the cell width or each cell; e.g.
          *[(2., 10, -1.3), (2., 50), (2., 10, 1.3)]*

    origin : (dim) iterable, default: 0
        Define the origin or 'anchor point' of the mesh; i.e. the bottom-left-frontmost
        corner. By default, the mesh is anchored such that its origin is at
        ``[0, 0, 0]``.

        For each dimension (r, theta or z), The user may set the origin 2 ways:

        - a ``scalar`` which explicitly defines origin along that dimension.
        - **{'0', 'C', 'N'}** a :class:`str` specifying whether the zero coordinate
          along each axis is the first node location ('0'), in the center ('C') or the
          last node location ('N').

    Examples
    --------
    To create a general 3D cylindrical mesh, we discretize along the radial,
    azimuthal and vertical axis. For example:

    >>> from discretize import CylindricalMesh
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np

    >>> ncr = 10  # number of mesh cells in r
    >>> ncp = 8   # number of mesh cells in phi
    >>> ncz = 15  # number of mesh cells in z
    >>> dr = 15   # cell width r
    >>> dz = 10   # cell width z

    >>> hr = dr * np.ones(ncr)
    >>> hp = (2 * np.pi / ncp) * np.ones(ncp)
    >>> hz = dz * np.ones(ncz)
    >>> mesh1 = CylindricalMesh([hr, hp, hz])

    >>> mesh1.plot_grid()
    >>> plt.show()

    For a cylindrically symmetric mesh, the disretization along the
    azimuthal direction is set with a flag of *1*. This reduces the
    size of numerical systems given that the derivative along the
    azimuthal direction is 0. For example:

    >>> ncr = 10      # number of mesh cells in r
    >>> ncz = 15      # number of mesh cells in z
    >>> dr = 15       # cell width r
    >>> dz = 10       # cell width z
    >>> npad_r = 4    # number of padding cells in r
    >>> npad_z = 4    # number of padding cells in z
    >>> exp_r = 1.25  # expansion rate of padding cells in r
    >>> exp_z = 1.25  # expansion rate of padding cells in z

    A value of 1 is used to define the discretization in phi for this case.

    >>> hr = [(dr, ncr), (dr, npad_r, exp_r)]
    >>> hz = [(dz, npad_z, -exp_z), (dz, ncz), (dz, npad_z, exp_z)]
    >>> mesh2 = CylindricalMesh([hr, 1, hz])

    >>> mesh2.plot_grid()
    >>> plt.show()
    """

    _meshType = "CYL"
    _unitDimensions = [1, 2 * np.pi, 1]
    _aliases = {
        **DiffOperators._aliases,
        **BaseRectangularMesh._aliases,
        **BaseTensorMesh._aliases,
    }

    _items = BaseTensorMesh._items | {"cartesian_origin"}

    def __init__(self, h, origin=None, cartesian_origin=None, **kwargs):
        kwargs.pop("reference_system", None)  # reference system must be cylindrical
        if "cartesianOrigin" in kwargs.keys():
            cartesian_origin = kwargs.pop("cartesianOrigin")
        super().__init__(h=h, origin=origin, reference_system="cylindrical", **kwargs)

        if not np.abs(self.h[1].sum() - 2 * np.pi) < 1e-10:
            raise AssertionError("The 2nd dimension must sum to 2*pi")

        if self.dim == 2:
            print("Warning, a disk mesh has not been tested thoroughly.")

        if cartesian_origin is None:
            cartesian_origin = np.zeros(self.dim)
        self.cartesian_origin = cartesian_origin

    @property
    def cartesian_origin(self):
        """Cartesian origin of the mesh

        Returns the origin or 'anchor point' of the cylindrical mesh
        in Cartesian coordinates; i.e. [x0, y0, z0]. For cylindrical
        meshes, the origin is the bottom of the z-axis which defines
        the mesh's rotational symmetry.

        Returns
        -------
        (dim) numpy.ndarray
            The Cartesian origin (or anchor point) of the mesh
        """
        return self._cartesian_origin

    @cartesian_origin.setter
    def cartesian_origin(self, value):
        # ensure the value is a numpy array
        value = np.asarray(value, dtype=np.float64)
        value = np.atleast_1d(value)
        if len(value) != self.dim:
            raise ValueError(
                f"cartesian origin and shape must be the same length, got {len(value)} and {len(self.dim)}"
            )
        self._cartesian_origin = value

    @property
    def is_symmetric(self):
        """Validates whether mesh is symmetric.

        Symmetric cylindrical meshes have useful mathematical
        properties that allow us to reduce the computational cost
        of solving radially symmetric 3D problems.
        When constructing cylindrical meshes in discretize, we almost
        always use a flag of *1* when defining the discretization in
        the azimuthal direction. By doing so, we define a mesh that is
        symmetric. In this case, the *is_symmetric* returns a value of
        *True* . If the discretization in the azimuthal direction was
        defined explicitly, the mesh would not be symmetric and
        *is_symmetric* would return a value of *False* .

        Returns
        -------
        bool
            *True* if the mesh is symmetric, *False* otherwise
        """
        return self.shape_cells[1] == 1

    @property
    def shape_nodes(self):
        """Returns the number of nodes along each axis

        This property returns a tuple containing the number of nodes along
        the :math:`x` (radial), :math:`y` (azimuthal) and :math:`z` (vertical)
        directions, respectively.
        In the case where the mesh is symmetric, the number of nodes
        defining the discretization in the azimuthal direction is *0* ;
        see :py:attr:`~.CylindricalMesh.is_symmetric`.

        Returns
        -------
        (dim) tuple of int
            Number of nodes in the :math:`x` (radial),
            :math:`y` (azimuthal) and :math:`z` (vertical) directions, respectively.
        """
        vnC = self.shape_cells
        if self.is_symmetric:
            return (vnC[0], 0, vnC[2] + 1)
        else:
            return (vnC[0] + 1, vnC[1], vnC[2] + 1)

    @property
    def _shape_total_nodes(self):
        vnC = self.shape_cells
        if self.is_symmetric:
            return (vnC[0], 1, vnC[2] + 1)
        else:
            return tuple(x + 1 for x in vnC)

    @property
    def n_nodes(self):
        """Returns total number of mesh nodes

        For non-symmetric cylindrical meshes, this property returns
        the total number of nodes. For symmetric meshes, this property
        returns a value of 0; see :py:attr:`~.CylindricalMesh.is_symmetric`.
        The symmetric mesh case is unique because the azimuthal position
        of the nodes is undefined.

        Returns
        -------
        int
            Total number of nodes for non-symmetric meshes, 0 for symmetric meshes
        """
        if self.is_symmetric:
            return 0
        nx, ny, nz = self.shape_nodes
        return (nx - 1) * ny * nz + nz

    @property
    def _shape_total_faces_x(self):
        """
        vector number of total Fx (prior to deflating)
        """
        return self._shape_total_nodes[:1] + self.shape_cells[1:]

    @property
    def _n_total_faces_x(self):
        """
        number of total Fx (prior to defplating)
        """
        return int(np.prod(self._shape_total_faces_x))

    @property
    def _n_hanging_faces_x(self):
        """
        Number of hanging Fx
        """
        return int(np.prod(self.shape_cells[1:]))

    @property
    def shape_faces_x(self):
        """Number of x-faces along each axis direction

        This property returns the number of x-faces along the
        :math:`x` (radial), :math:`y` (azimuthal) and :math:`z` (vertical)
        directions, respectively. Note that for symmetric meshes, the number of x-faces along
        the azimuthal direction is 1; see :py:attr:`~.CylindricalMesh.is_symmetric`.

        Returns
        -------
        (dim) tuple of int
            Number of x-faces along the :math:`x` (radial),
            :math:`y` (azimuthal) and :math:`z` (vertical) directions, respectively.
        """
        return self.shape_cells

    @property
    def _shape_total_faces_y(self):
        """
        vector number of total Fy (prior to deflating)
        """
        vnC = self.shape_cells
        return (vnC[0], self._shape_total_nodes[1]) + vnC[2:]

    @property
    def _n_total_faces_y(self):
        """
        number of total Fy (prior to deflating)
        """
        return int(np.prod(self._shape_total_faces_y))

    @property
    def _n_hanging_faces_y(self):
        """
        number of hanging y-faces
        """
        return int(np.prod(self.shape_cells[::2]))

    @property
    def _shape_total_faces_z(self):
        """
        vector number of total Fz (prior to deflating)
        """
        return self.shape_cells[:-1] + self._shape_total_nodes[-1:]

    @property
    def _n_total_faces_z(self):
        """
        number of total Fz (prior to deflating)
        """
        return int(np.prod(self._shape_total_faces_z))

    @property
    def _n_hanging_faces_z(self):
        """
        number of hanging Fz
        """
        return int(np.prod(self.shape_cells[::2]))

    @property
    def _shape_total_edges_x(self):
        """
        vector number of total Ex (prior to deflating)
        """
        return self.shape_cells[:1] + self._shape_total_nodes[1:]

    @property
    def _n_total_edges_x(self):
        """
        number of total Ex (prior to deflating)
        """
        return int(np.prod(self._shape_total_edges_x))

    @property
    def _shape_total_edges_y(self):
        """
        vector number of total Ey (prior to deflating)
        """
        _shape_total_nodes = self._shape_total_nodes
        return (_shape_total_nodes[0], self.shape_cells[1], _shape_total_nodes[2])

    @property
    def _n_total_edges_y(self):
        """
        number of total Ey (prior to deflating)
        """
        return int(np.prod(self._shape_total_edges_y))

    @property
    def shape_edges_y(self):
        """Number of y-edges along each axis direction

        This property returns the number of y-edges along the
        :math:`x` (radial), :math:`y` (azimuthal) and :math:`z` (vertical)
        directions, respectively. Note that for symmetric meshes, the number of y-edges along
        the azimuthal direction is 1; see :py:attr:`~.CylindricalMesh.is_symmetric`.

        Returns
        -------
        (dim) tuple of int
            Number of y-edges along the :math:`x` (radial),
            :math:`y` (azimuthal) and :math:`z` (vertical) directions, respectively.
        """
        return tuple(x + y for x, y in zip(self.shape_cells, [0, 0, 1]))

    @property
    def _shape_total_edges_z(self):
        """
        vector number of total Ez (prior to deflating)
        """
        return self._shape_total_nodes[:-1] + self.shape_cells[-1:]

    @property
    def _n_total_edges_z(self):
        """
        number of total Ez (prior to deflating)
        """
        return int(np.prod(self._shape_total_edges_z))

    @property
    def shape_edges_z(self):
        """Number of z-edges along each axis direction

        This property returns the number of z-edges along the
        :math:`x` (radial), :math:`y` (azimuthal) and :math:`z` (vertical)
        directions, respectively. Note that for symmetric meshes, the number of z-edges along
        the azimuthal direction is 0; see :py:attr:`~.CylindricalMesh.is_symmetric`.
        The symmetric mesh case is unique because the azimuthal position
        of the z-edges is undefined.

        Returns
        -------
        (dim) tuple of int
            Number of z-edges along the :math:`x` (radial),
            :math:`y` (azimuthal) and :math:`z` (vertical) direction, respectively.
        """
        return self.shape_nodes[:-1] + self.shape_cells[-1:]

    @property
    def n_edges_z(self):
        """Total number of z-edges in the mesh

        This property returns the total number of z-edges for
        non-symmetric cyindrical meshes; see :py:attr:`~.CylindricalMesh.is_symmetric`.
        If the mesh is symmetric, the property returns *0* because the azimuthal
        position of the z-edges is undefined.

        Returns
        -------
        int
            Number of z-edges for non-symmetric meshes and *0* for symmetric meshes
        """
        z_shape = self.shape_edges_z
        cell_shape = self.shape_cells
        if self.is_symmetric:
            return int(np.prod(z_shape))
        return int(np.prod([z_shape[0] - 1, z_shape[1], cell_shape[2]])) + cell_shape[2]

    @property
    def cell_centers_x(self):
        """Returns the x-positions of cell centers along the x-direction

        This property returns a 1D vector containing the x-position values
        of the cell centers along the x-direction (radial). The length of the vector
        is equal to the number of cells in the x-direction.

        Returns
        -------
        (n_cells_x) numpy.ndarray
            x-positions of cell centers along the x-direction
        """
        return np.r_[0, self.h[0][:-1].cumsum()] + self.h[0] * 0.5

    @property
    def cell_centers_y(self):
        """Returns the y-positions of cell centers along the y-direction (azimuthal)

        This property returns a 1D vector containing the y-position values
        of the cell centers along the y-direction (azimuthal). The length of the vector
        is equal to the number of cells in the y-direction. If the mesh is symmetric,
        this property returns a numpy array with a single entry of *0* ;
        indicating all cell-centers have a y-position of 0.
        see :py:attr:`~.CylindricalMesh.is_symmetric`.

        Returns
        -------
        (n_cells_y) numpy.ndarray
            y-positions of cell centers along the y-direction
        """
        if self.is_symmetric:
            return np.r_[0, self.h[1][:-1]]
        return np.r_[0, self.h[1][:-1].cumsum()] + self.h[1] * 0.5

    @property
    def nodes_x(self):
        """Returns the x-positions of nodes along the x-direction (radial)

        This property returns a 1D vector containing the x-position values
        of the nodes along the x-direction (radial). The length of the vector
        is equal to the number of nodes in the x-direction.

        Returns
        -------
        (n_nodes_x) numpy.ndarray
            x-positions of nodes along the x-direction
        """
        if self.is_symmetric:
            return self.h[0].cumsum()
        return np.r_[0, self.h[0]].cumsum()

    @property
    def _nodes_y_full(self):
        """
        full nodal y vector (prior to deflating)
        """
        if self.is_symmetric:
            return np.r_[0]
        return np.r_[0, self.h[1].cumsum()]

    @property
    def nodes_y(self):
        """Returns the y-positions of nodes along the y-direction (azimuthal)

        This property returns a 1D vector containing the y-position values
        of the nodes along the y-direction (azimuthal). If the mesh is symmetric,
        this property returns a numpy array with a single entry of *0* ;
        indicating all nodes have a y-position of 0. See :py:attr:`~.CylindricalMesh.is_symmetric`.

        Returns
        -------
        (n_nodes_y) numpy.ndarray
            y-positions of nodes along the y-direction
        """
        return np.r_[0, self.h[1][:-1].cumsum()]

    @property
    def _edge_x_lengths_full(self):
        """
        full x-edge lengths (prior to deflating)
        """
        nx, ny, nz = self._shape_total_nodes
        return np.kron(np.ones(nz), np.kron(np.ones(ny), self.h[0]))

    @property
    def edge_x_lengths(self):
        """x-edge lengths for entire mesh

        If the mesh is not symmetric, this property returns a 1D vector
        containing the lengths of all x-edges in the mesh. If the mesh
        is symmetric, this property returns an empty numpy array since
        there are no x-edges;
        see :py:attr:`~CylindricalMesh.is_symmetric`.

        Returns
        -------
        (n_edges_x) numpy.ndarray
            A 1D array containing the x-edge lengths for the entire mesh
        """
        if getattr(self, "_edge_lengths_x", None) is None:
            self._edge_lengths_x = self._edge_x_lengths_full[~self._ishanging_edges_x]
        return self._edge_lengths_x

    @property
    def _edge_y_lengths_full(self):
        """
        full vector of y-edge lengths (prior to deflating)
        """
        if self.is_symmetric:
            return 2 * pi * self.nodes[:, 0]
        return np.kron(
            np.ones(self._shape_total_nodes[2]), np.kron(self.h[1], self.nodes_x)
        )

    @property
    def edge_y_lengths(self):
        """arc-lengths of y-edges for entire mesh

        This property returns a 1D vector containing the arc-lengths
        of all y-edges in the mesh. For a single y-edge at radial location
        :math:`r` with azimuthal width :math:`\\Delta \\phi`, the arc-length
        is given by:

        .. math::
            \\Delta y = r \\Delta \\phi

        Returns
        -------
        (n_edges_y) numpy.ndarray
            A 1D array containing the y-edge arc-lengths for the entire mesh
        """
        if getattr(self, "_edge_lengths_y", None) is None:
            if self.is_symmetric:
                self._edge_lengths_y = self._edge_y_lengths_full
            else:
                self._edge_lengths_y = self._edge_y_lengths_full[
                    ~self._ishanging_edges_y
                ]
        return self._edge_lengths_y

    @property
    def _edge_z_lengths_full(self):
        """
        full z-edge lengths (prior to deflation)
        """
        nx, ny, nz = self._shape_total_nodes
        return np.kron(self.h[2], np.kron(np.ones(ny), np.ones(nx)))

    @property
    def edge_z_lengths(self):
        """z-edge lengths for entire mesh

        If the mesh is not symmetric, this property returns a 1D vector
        containing the lengths of all z-edges in the mesh. If the mesh
        is symmetric, this property returns an empty numpy array
        since there are no z-edges; see :py:attr:`~CylindricalMesh.is_symmetric`.

        Returns
        -------
        (n_edges_z) numpy.ndarray
            A 1D array containing the z-edge lengths for the entire mesh
        """
        if getattr(self, "_edge_lengths_z", None) is None:
            self._edge_lengths_z = self._edge_z_lengths_full[~self._ishanging_edges_z]
        return self._edge_lengths_z

    @property
    def _edge_lengths_full(self):
        """
        full edge lengths [r-edges, theta-edgesm z-edges] (prior to
        deflation)
        """
        if self.is_symmetric:
            raise NotImplementedError
        else:
            return np.r_[
                self._edge_x_lengths_full,
                self._edge_y_lengths_full,
                self._edge_z_lengths_full,
            ]

    @property
    def edge_lengths(self):
        """Lengths of all mesh edges

        This property returns a 1D vector containing the lengths
        of all edges in the mesh organized by x-edges, y-edges, then z-edges;
        i.e. radial, azimuthal, then vertical. However if the mesh
        is symmetric, there are no x or z-edges and calling the property
        returns the y-edge lengths; see :py:attr:`~CylindricalMesh.is_symmetric`.
        Note that y-edge lengths take curvature into account; see
        :py:attr:`~.CylindricalMesh.edge_y_lengths`.

        Returns
        -------
        (n_edges) numpy.ndarray
            Edge lengths of all mesh edges organized x (radial), y (azimuthal), then z (vertical)
        """
        if self.is_symmetric:
            return self.edge_y_lengths
        else:
            return np.r_[self.edge_x_lengths, self.edge_y_lengths, self.edge_z_lengths]

    @property
    def _face_x_areas_full(self):
        """
        area of x-faces prior to deflation
        """
        if self.is_symmetric:
            return np.kron(self.h[2], 2 * pi * self.nodes_x)
        return np.kron(self.h[2], np.kron(self.h[1], self.nodes_x))

    @property
    def face_x_areas(self):
        """x-face areas for the entire mesh

        This property returns a 1D vector containing the areas of the
        x-faces of the mesh. The surface area takes into account curvature.
        For a single x-face at radial location
        :math:`r` with azimuthal width :math:`\\Delta \\phi` and vertical
        width :math:`h_z`, the area is given by:

        .. math::
            A_x = r \\Delta phi h_z

        Returns
        -------
        (n_faces_x) numpy.ndarray
            A 1D array containing the x-face areas for the entire mesh
        """
        if getattr(self, "_face_x_areas", None) is None:
            if self.is_symmetric:
                self._face_x_areas = self._face_x_areas_full
            else:
                self._face_x_areas = self._face_x_areas_full[~self._ishanging_faces_x]
        return self._face_x_areas

    @property
    def _face_y_areas_full(self):
        """
        Area of y-faces (Azimuthal faces), prior to deflation.
        """
        return np.kron(
            self.h[2], np.kron(np.ones(self._shape_total_nodes[1]), self.h[0])
        )

    @property
    def face_y_areas(self):
        """y-face areas for the entire mesh

        This property returns a 1D vector containing the areas of the
        y-faces of the mesh. For a single y-face with edge lengths
        :math:`h_x` and :math:`h_z`, the area is given by:

        .. math::
            A_y = h_x h_z

        *Note that for symmetric meshes* , there are no y-faces and calling
        this property will return an error.

        Returns
        -------
        (n_faces_y) numpy.ndarray
            A 1D array containing the y-face areas in the case of non-symmetric meshes.
            Returns an error for symmetric meshes.
        """
        if getattr(self, "_face_y_areas", None) is None:
            if self.is_symmetric:
                raise Exception("There are no y-faces on the Cyl Symmetric mesh")
            self._face_y_areas = self._face_y_areas_full[~self._ishanging_faces_y]
        return self._face_y_areas

    @property
    def _face_z_areas_full(self):
        """
        area of z-faces prior to deflation
        """
        if self.is_symmetric:
            return np.kron(
                np.ones_like(self.nodes_z),
                pi * (self.nodes_x ** 2 - np.r_[0, self.nodes_x[:-1]] ** 2),
            )
        return np.kron(
            np.ones(self._shape_total_nodes[2]),
            np.kron(
                self.h[1],
                0.5 * (self.nodes_x[1:] ** 2 - self.nodes_x[:-1] ** 2),
            ),
        )

    @property
    def face_z_areas(self):
        """z-face areas for the entire mesh

        This property returns a 1D vector containing the areas of the
        z-faces of the mesh. The surface area takes into account curvature.
        For a single z-face at between :math:`r_1` and :math:`r_2`
        with azimuthal width :math:`\\Delta \\phi`, the area is given by:

        .. math::
            A_z = \\frac{\\Delta \\phi}{2} (r_2^2 - r_1^2)

        Returns
        -------
        (n_faces_z) numpy.ndarray
            A 1D array containing the z-face areas for the entire mesh
        """
        if getattr(self, "_face_z_areas", None) is None:
            if self.is_symmetric:
                self._face_z_areas = self._face_z_areas_full
            else:
                self._face_z_areas = self._face_z_areas_full[~self._ishanging_faces_z]
        return self._face_z_areas

    @property
    def _face_areas_full(self):
        """
        Area of all faces (prior to delflation)
        """
        return np.r_[
            self._face_x_areas_full, self._face_y_areas_full, self._face_z_areas_full
        ]

    @property
    def face_areas(self):
        """Face areas for the entire mesh

        This property returns a 1D vector containing the areas of all
        mesh faces organized by x-faces, y-faces, then z-faces;
        i.e. faces normal to the radial, azimuthal, then vertical direction.
        Note that for symmetric meshes, there are no y-faces and calling the
        property will return only the x and z-faces. To see how the face
        areas corresponding to each component are computed, see
        :py:attr:`~.CylindricalMesh.face_x_areas`,
        :py:attr:`~.CylindricalMesh.face_y_areas` and
        :py:attr:`~.CylindricalMesh.face_z_areas`.

        Returns
        -------
        (n_faces) numpy.ndarray
            Areas of all faces in the mesh
        """
        # if getattr(self, '_area', None) is None:
        if self.is_symmetric:
            return np.r_[self.face_x_areas, self.face_z_areas]
        else:
            return np.r_[self.face_x_areas, self.face_y_areas, self.face_z_areas]

    @property
    def cell_volumes(self):
        """Volumes of all mesh cells

        This property returns a 1D vector containing the volumes of
        all cells in the mesh. When computing the volume of each cell,
        we take into account curvature. Thus a cell lying within
        radial distance :math:`r_1` and :math:`r_2`, with height
        :math:`h_z` and with azimuthal width :math:`\\Delta \\phi`,
        the volume is given by:

        .. math::
            V = \\frac{\\Delta \\phi \\, h_z}{2} (r_2^2 - r_1^2)

        Returns
        -------
        (n_cells numpy.ndarray
            Volumes of all mesh cells
        """
        if getattr(self, "_cell_volumes", None) is None:
            if self.is_symmetric:
                az = pi * (self.nodes_x ** 2 - np.r_[0, self.nodes_x[:-1]] ** 2)
                self._cell_volumes = np.kron(self.h[2], az)
            else:
                self._cell_volumes = np.kron(
                    self.h[2],
                    np.kron(
                        self.h[1],
                        0.5 * (self.nodes_x[1:] ** 2 - self.nodes_x[:-1] ** 2),
                    ),
                )
        return self._cell_volumes

    ###########################################################################
    # Active and Hanging Edges and Faces
    #
    #    To find the active edges, faces, we use krons of bools (sorry). It is
    #    more efficient than working with 3D matrices. For example...
    #
    #    The computation of `ishangingFx` (is the Fx face hanging? a vector of
    #    True and False corresponding to each face) can be computed using krons
    #    of bools:
    #
    #          hang_x = np.zeros(self._ntNx, dtype=bool)
    #          hang_x[0] = True
    #          ishangingFx_bool = np.kron(
    #              np.ones(self.shape_cells[2], dtype=bool),  # 1 * 0 == 0
    #              np.kron(np.ones(self.shape_cells[1], dtype=bool), hang_x)
    #          )
    #          return self._ishanging_faces_x_bool
    #
    #
    #   This is equivalent to forming the 3D matrix and indexing the
    #   corresponding rows and columns (here, the hanging faces are all of
    #   the first x-faces along the axis of symmetry):
    #
    #         hang_x = np.zeros(self._shape_total_faces_x, dtype=bool)
    #         hang_x[0, :, :] = True
    #         isHangingFx_bool = mkvc(hang_x)
    #
    #
    # but krons of bools is more efficient.
    #
    ###########################################################################

    @property
    def _ishanging_faces_x(self):
        """
        bool vector indicating if an x-face is hanging or not
        """
        if getattr(self, "_ishanging_faces_x_bool", None) is None:

            # the following is equivalent to
            #     hang_x = np.zeros(self._shape_total_faces_x, dtype=bool)
            #     hang_x[0, :, :] = True
            #     isHangingFx_bool = mkvc(hang_x)
            #
            # but krons of bools is more efficient

            hang_x = np.zeros(self._shape_total_nodes[0], dtype=bool)
            hang_x[0] = True
            self._ishanging_faces_x_bool = np.kron(
                np.ones(self.shape_cells[2], dtype=bool),  # 1 * 0 == 0
                np.kron(np.ones(self.shape_cells[1], dtype=bool), hang_x),
            )
        return self._ishanging_faces_x_bool

    @property
    def _hanging_faces_x(self):
        """
        dictionary of the indices of the hanging x-faces (keys) and a list
        of indices that the eliminated faces map to (if applicable)
        """
        if getattr(self, "_hanging_faces_x_dict", None) is None:
            self._hanging_faces_x_dict = dict(
                zip(
                    np.nonzero(self._ishanging_faces_x)[0].tolist(),
                    [None] * self._n_hanging_faces_x,
                )
            )
        return self._hanging_faces_x_dict

    @property
    def _ishanging_faces_y(self):
        """
        bool vector indicating if a y-face is hanging or not
        """

        if getattr(self, "_ishanging_faces_y_bool", None) is None:
            hang_y = np.zeros(self._shape_total_nodes[1], dtype=bool)
            hang_y[-1] = True
            self._ishanging_faces_y_bool = np.kron(
                np.ones(self.shape_cells[2], dtype=bool),
                np.kron(hang_y, np.ones(self.shape_cells[0], dtype=bool)),
            )
        return self._ishanging_faces_y_bool

    @property
    def _hanging_faces_y(self):
        """
        dictionary of the indices of the hanging y-faces (keys) and a list
        of indices that the eliminated faces map to (if applicable)
        """
        if getattr(self, "_hanging_faces_y_dict", None) is None:
            deflate_y = np.zeros(self._shape_total_nodes[1], dtype=bool)
            deflate_y[0] = True
            deflateFy = np.nonzero(
                np.kron(
                    np.ones(self.shape_cells[2], dtype=bool),
                    np.kron(deflate_y, np.ones(self.shape_cells[0], dtype=bool)),
                )
            )[0].tolist()
            self._hanging_faces_y_dict = dict(
                zip(np.nonzero(self._ishanging_faces_y)[0].tolist(), deflateFy)
            )
        return self._hanging_faces_y_dict

    @property
    def _ishanging_faces_z(self):
        """
        bool vector indicating if a z-face is hanging or not
        """
        if getattr(self, "_ishanging_faces_z_bool", None) is None:
            self._ishanging_faces_z_bool = np.zeros(self._n_total_faces_z, dtype=bool)
        return self._ishanging_faces_z_bool

    @property
    def _hanging_faces_z(self):
        """
        dictionary of the indices of the hanging z-faces (keys) and a list
        of indices that the eliminated faces map to (if applicable)
        """
        return {}

    @property
    def _ishanging_edges_x(self):
        """
        bool vector indicating if a x-edge is hanging or not
        """
        if getattr(self, "_ishanging_edges_x_bool", None) is None:
            nx, ny, nz = self._shape_total_nodes
            hang_y = np.zeros(ny, dtype=bool)
            hang_y[-1] = True
            self._ishanging_edges_x_bool = np.kron(
                np.ones(nz, dtype=bool),
                np.kron(hang_y, np.ones(self.shape_cells[0], dtype=bool)),
            )
        return self._ishanging_edges_x_bool

    @property
    def _hanging_edges_x(self):
        """
        dictionary of the indices of the hanging x-edges (keys) and a list
        of indices that the eliminated faces map to (if applicable)
        """
        if getattr(self, "_hanging_edges_x_dict", None) is None:
            nx, ny, nz = self._shape_total_nodes
            deflate_y = np.zeros(ny, dtype=bool)
            deflate_y[0] = True
            deflateEx = np.nonzero(
                np.kron(
                    np.ones(nz, dtype=bool),
                    np.kron(deflate_y, np.ones(self.shape_cells[0], dtype=bool)),
                )
            )[0].tolist()
            self._hanging_edges_x_dict = dict(
                zip(np.nonzero(self._ishanging_edges_x)[0].tolist(), deflateEx)
            )
        return self._hanging_edges_x_dict

    @property
    def _ishanging_edges_y(self):
        """
        bool vector indicating if a y-edge is hanging or not
        """
        if getattr(self, "_ishanging_edges_y_bool", None) is None:
            nx, ny, nz = self._shape_total_nodes
            hang_x = np.zeros(nx, dtype=bool)
            hang_x[0] = True
            self._ishanging_edges_y_bool = np.kron(
                np.ones(nz, dtype=bool),
                np.kron(np.ones(self.shape_cells[1], dtype=bool), hang_x),
            )
        return self._ishanging_edges_y_bool

    @property
    def _hanging_edges_y(self):
        """
        dictionary of the indices of the hanging y-edges (keys) and a list
        of indices that the eliminated faces map to (if applicable)
        """
        if getattr(self, "_hanging_edges_y_dict", None) is None:
            self._hanging_edges_y_dict = dict(
                zip(
                    np.nonzero(self._ishanging_edges_y)[0].tolist(),
                    [None] * len(self._ishanging_edges_y_bool),
                )
            )
        return self._hanging_edges_y_dict

    @property
    def _axis_of_symmetry_edges_z(self):
        """
        bool vector indicating if a z-edge is along the axis of symmetry or not
        """
        if getattr(self, "_axis_of_symmetry_edges_z_bool", None) is None:
            nx, ny, nz = self._shape_total_nodes
            axis_x = np.zeros(nx, dtype=bool)
            axis_x[0] = True

            axis_y = np.zeros(ny, dtype=bool)
            axis_y[0] = True
            self._axis_of_symmetry_edges_z_bool = np.kron(
                np.ones(self.shape_cells[2], dtype=bool), np.kron(axis_y, axis_x)
            )
        return self._axis_of_symmetry_edges_z_bool

    @property
    def _ishanging_edges_z(self):
        """
        bool vector indicating if a z-edge is hanging or not
        """
        if getattr(self, "_ishanging_edges_z_bool", None) is None:
            if self.is_symmetric:
                self._ishanging_edges_z_bool = np.ones(
                    self._n_total_edges_z, dtype=bool
                )
            else:
                nx, ny, nz = self._shape_total_nodes
                hang_x = np.zeros(nx, dtype=bool)
                hang_x[0] = True

                hang_y = np.zeros(ny, dtype=bool)
                hang_y[-1] = True

                hangingEz = np.kron(
                    np.ones(self.shape_cells[2], dtype=bool),
                    (
                        # True * False = False
                        np.kron(np.ones(ny, dtype=bool), hang_x)
                        | np.kron(hang_y, np.ones(nx, dtype=bool))
                    ),
                )

                self._ishanging_edges_z_bool = (
                    hangingEz & ~self._axis_of_symmetry_edges_z
                )

        return self._ishanging_edges_z_bool

    @property
    def _hanging_edges_z(self):
        """
        dictionary of the indices of the hanging z-edges (keys) and a list
        of indices that the eliminated faces map to (if applicable)
        """
        if getattr(self, "_hanging_edges_z_dict", None) is None:
            nx, ny, nz = self._shape_total_nodes
            # deflate
            deflateEz = np.hstack(
                [
                    np.hstack(
                        [np.zeros(ny - 1, dtype=int), np.arange(1, nx, dtype=int)]
                    )
                    + i * int(nx * ny)
                    for i in range(self.shape_cells[2])
                ]
            )
            deflate = zip(np.nonzero(self._ishanging_edges_z)[0].tolist(), deflateEz)

            self._hanging_edges_z_dict = dict(deflate)
        return self._hanging_edges_z_dict

    @property
    def _axis_of_symmetry_nodes(self):
        """
        bool vector indicating if a node is along the axis of symmetry or not
        """
        if getattr(self, "_axis_of_symmetry_nodes_bool", None) is None:
            nx, ny, nz = self._shape_total_nodes
            axis_x = np.zeros(nx, dtype=bool)
            axis_x[0] = True

            axis_y = np.zeros(ny, dtype=bool)
            axis_y[0] = True
            self._axis_of_symmetry_nodes_bool = np.kron(
                np.ones(nz, dtype=bool), np.kron(axis_y, axis_x)
            )
        return self._axis_of_symmetry_nodes_bool

    @property
    def _ishanging_nodes(self):
        """
        bool vector indicating if a node is hanging or not
        """
        if getattr(self, "_ishanging_nodes_bool", None) is None:
            nx, ny, nz = self._shape_total_nodes
            hang_x = np.zeros(nx, dtype=bool)
            hang_x[0] = True

            hang_y = np.zeros(ny, dtype=bool)
            hang_y[-1] = True

            hangingN = np.kron(
                np.ones(nz, dtype=bool),
                (
                    np.kron(np.ones(ny, dtype=bool), hang_x)
                    | np.kron(hang_y, np.ones(nx, dtype=bool))
                ),
            )

            self._ishanging_nodes_bool = hangingN & ~self._axis_of_symmetry_nodes

        return self._ishanging_nodes_bool

    @property
    def _hanging_nodes(self):
        """
        dictionary of the indices of the hanging nodes (keys) and a list
        of indices that the eliminated faces map to (if applicable)
        """
        if getattr(self, "_hanging_nodes_dict", None) is None:
            nx, ny, nz = self._shape_total_nodes
            # go by layer
            deflateN = np.hstack(
                [
                    np.hstack(
                        [np.zeros(ny - 1, dtype=int), np.arange(1, nx, dtype=int)]
                    )
                    + i * int(nx * ny)
                    for i in range(nz)
                ]
            ).tolist()
            self._hanging_nodes_dict = dict(
                zip(np.nonzero(self._ishanging_nodes)[0].tolist(), deflateN)
            )
        return self._hanging_nodes_dict

    ####################################################
    # Grids
    ####################################################

    @property
    def _nodes_full(self):
        """
        Full Nodal grid (including hanging nodes)
        """
        return ndgrid([self.nodes_x, self._nodes_y_full, self.nodes_z])

    @property
    def nodes(self):
        """Gridded node locations

        This property outputs a numpy array containing the gridded
        locations of all mesh nodes in cylindrical ccordinates;
        i.e. :math:`(r, \\phi, z)`. Note that for symmetric meshes, the azimuthal
        position of all nodes is set to :math:`\\phi = 0`.

        Returns
        -------
        (n_nodes, dim) numpy.ndarray
            gridded node locations
        """
        if self.is_symmetric:
            self._nodes = self._nodes_full
        if getattr(self, "_nodes", None) is None:
            self._nodes = self._nodes_full[~self._ishanging_nodes, :]
        return self._nodes

    @property
    def _faces_x_full(self):
        """
        Full Fx grid (including hanging faces)
        """
        return ndgrid([self.nodes_x, self.cell_centers_y, self.cell_centers_z])

    @property
    def faces_x(self):
        """Gridded x-face (radial face) locations

        This property outputs a numpy array containing the gridded
        locations of all x-faces (radial faces) in cylindrical coordinates;
        i.e. the :math:`(r, \\phi, z)` position of the center of each face.
        The shape of the array is (n_faces_x, 3). Note that for symmetric meshes,
        the azimuthal position of all x-faces is set to :math:`\\phi = 0`.

        Returns
        -------
        (n_faces_x, dim) numpy.ndarray
            gridded x-face (radial face) locations
        """
        if getattr(self, "_faces_x", None) is None:
            if self.is_symmetric:
                return super().faces_x
            else:
                self._faces_x = self._faces_x_full[~self._ishanging_faces_x, :]
        return self._faces_x

    @property
    def _edges_y_full(self):
        """
        Full grid of y-edges (including eliminated edges)
        """
        return super().edges_y

    @property
    def edges_y(self):
        """Gridded y-edge (azimuthal edge) locations

        This property outputs a numpy array containing the gridded
        locations of all y-edges (azimuthal edges) in cylindrical coordinates;
        i.e. the :math:`(r, \\phi, z)` position of the middle of each y-edge.
        The shape of the array is (n_edges_y, 3). Note that for symmetric meshes,
        the azimuthal position of all y-edges is set to :math:`\\phi = 0`.

        Returns
        -------
        (n_edges_y, dim) numpy.ndarray
            gridded y-edge (azimuthal edge) locations
        """
        if getattr(self, "_edges_y", None) is None:
            if self.is_symmetric:
                return self._edges_y_full
            else:
                self._edges_y = self._edges_y_full[~self._ishanging_edges_y, :]
        return self._edges_y

    @property
    def _edges_z_full(self):
        """
        Full z-edge grid (including hanging edges)
        """
        return ndgrid([self.nodes_x, self._nodes_y_full, self.cell_centers_z])

    @property
    def edges_z(self):
        """Gridded z-edge (vertical edge) locations

        This property outputs a numpy array containing the gridded
        locations of all z-edges (vertical edges) in cylindrical coordinates;
        i.e. the :math:`(r, \\phi, z)` position of the middle of each z-edge.
        The shape of the array is (n_edges_z, 3). In the case of symmetric
        meshes, there are no z-edges and this property returns *None*.

        Returns
        -------
        (n_edges_z, dim) numpy.ndarray or None
            gridded z-edge (vertical edge) locations. Returns *None* for symmetric meshes.
        """
        if getattr(self, "_edges_z", None) is None:
            if self.is_symmetric:
                self._edges_z = None
            else:
                self._edges_z = self._edges_z_full[~self._ishanging_edges_z, :]
        return self._edges_z

    ####################################################
    # Operators
    ####################################################

    @property
    def face_divergence(self):
        if getattr(self, "_face_divergence", None) is None:
            # Compute faceDivergence operator on faces
            D1 = self.face_x_divergence
            D3 = self.face_z_divergence
            if self.is_symmetric:
                D = sp.hstack((D1, D3), format="csr")
            elif self.shape_cells[1] > 1:
                D2 = self.face_y_divergence
                D = sp.hstack((D1, D2, D3), format="csr")
            self._face_divergence = D
        return self._face_divergence

    @property
    def face_x_divergence(self):
        if getattr(self, "_face_x_divergence", None) is None:
            if self.is_symmetric:
                ncx, ncy, ncz = self.shape_cells
                D1 = kron3(speye(ncz), speye(ncy), ddx(ncx)[:, 1:])
            else:
                D1 = super()._face_x_divergence_stencil

            S = self._face_x_areas_full
            V = self.cell_volumes
            self._face_x_divergence = sdiag(1 / V) * D1 * sdiag(S)

            if not self.is_symmetric:
                self._face_x_divergence = (
                    self._face_x_divergence
                    * self._deflation_matrix("Fx", as_ones=True).T
                )

        return self._face_x_divergence

    @property
    def face_y_divergence(self):
        if getattr(self, "_face_y_divergence", None) is None:
            D2 = super()._face_y_divergence_stencil
            S = self._face_y_areas_full  # self.reshape(self.face_areas, 'F', 'Fy', 'V')
            V = self.cell_volumes
            self._face_y_divergence = (
                sdiag(1 / V)
                * D2
                * sdiag(S)
                * self._deflation_matrix("Fy", as_ones=True).T
            )
        return self._face_y_divergence

    @property
    def face_z_divergence(self):
        if getattr(self, "_face_z_divergence", None) is None:
            D3 = super()._face_z_divergence_stencil
            S = self._face_z_areas_full
            V = self.cell_volumes
            self._face_z_divergence = sdiag(1 / V) * D3 * sdiag(S)
        return self._face_z_divergence

    # @property
    # def stencil_cell_gradient_x(self):
    #     n = self.vnC

    #     if self.is_symmetric:
    #         G1 = sp.kron(speye(n[2]), ddxCellGrad(n[0], BC))
    #     else:
    #         G1 = self._deflation_matrix('Fx').T * kron3(
    #             speye(n[2]), speye(n[1]), ddxCellGrad(n[0], BC)
    #         )
    #     return G1

    @property
    def cell_gradient_x(self):
        raise NotImplementedError("Cell Grad is not yet implemented.")
        # if getattr(self, '_cellGradx', None) is None:
        #     G1 = super(CylindricalMesh, self).stencil_cell_gradient_x
        #     V = self._deflation_matrix('F', withHanging='True', as_ones='True')*self.aveCC2F*self.cell_volumes
        #     A = self.face_areas
        #     L = (A/V)[:self._n_total_faces_x]
        #     # L = self.reshape(L, 'F', 'Fx', 'V')
        #     # L = A[:self.nFx] / V
        #     self._cellGradx = self._deflation_matrix('Fx')*sdiag(L)*G1
        # return self._cellGradx

    @property
    def stencil_cell_gradient_y(self):
        raise NotImplementedError("Cell Grad is not yet implemented.")

    @property
    def stencil_cell_gradient_z(self):
        raise NotImplementedError("Cell Grad is not yet implemented.")

    @property
    def stencil_cell_gradient(self):
        raise NotImplementedError("Cell Grad is not yet implemented.")

    @property
    def cell_gradient(self):
        raise NotImplementedError("Cell Grad is not yet implemented.")

    # @property
    # def _nodal_gradient_x_stencil(self):
    #     if self.is_symmetric:
    #         return None
    #     return kron3(speye(self.shape_nodes[2]), speye(self.shape_nodes[1]), ddx(self.shape_cells[0]))

    # @property
    # def _nodal_gradient_y_stencil(self):
    #     if self.is_symmetric:
    #         None
    #         # return kron3(speye(self.shape_nodes[2]), ddx(self.shape_cells[1]), speye(self.shape_nodes[0])) * self._deflation_matrix('Ey')
    #     return kron3(speye(self.shape_nodes[2]), ddx(self.shape_cells[1]), speye(self.shape_nodes[0]))

    # @property
    # def _nodal_gradient_z_stencil(self):
    #     if self.is_symmetric:
    #         return None
    #     return kron3(ddx(self.shape_cells[2]), speye(self.shape_nodes[1]), speye(self.shape_nodes[0]))

    # @property
    # def _nodal_gradient_stencil(self):
    #     if self.is_symmetric:
    #         return None
    #     else:
    #         G = self._deflation_matrix('E').T * sp.vstack((
    #             self._nodal_gradient_x_stencil,
    #             self._nodal_gradient_y_stencil,
    #             self._nodal_gradient_z_stencil
    #         ), format="csr") * self._deflation_matrix('N')
    #     return G

    @property
    def nodal_gradient(self):
        if self.is_symmetric:
            return None
        raise NotImplementedError("nodalGrad not yet implemented")

    @property
    def nodal_laplacian(self):
        raise NotImplementedError("nodalLaplacian not yet implemented")

    @property
    def edge_curl(self):
        if getattr(self, "_edge_curl", None) is None:
            A = self.face_areas
            E = self.edge_lengths

            if self.is_symmetric:
                nCx, nCy, nCz = self.shape_cells
                # 1D Difference matricies
                dr = sp.spdiags(
                    (np.ones((nCx + 1, 1)) * [-1, 1]).T, [-1, 0], nCx, nCx, format="csr"
                )
                dz = sp.spdiags(
                    (np.ones((nCz + 1, 1)) * [-1, 1]).T,
                    [0, 1],
                    nCz,
                    nCz + 1,
                    format="csr",
                )
                # 2D Difference matricies
                Dr = sp.kron(sp.identity(nCz + 1), dr)
                Dz = -sp.kron(dz, sp.identity(nCx))

                # Edge curl operator
                self._edge_curl = sdiag(1 / A) * sp.vstack((Dz, Dr)) * sdiag(E)
            else:
                self._edge_curl = (
                    sdiag(1 / self.face_areas)
                    * self._deflation_matrix("F", as_ones=False)
                    * self._edge_curl_stencil
                    * sdiag(self._edge_lengths_full)
                    * self._deflation_matrix("E", as_ones=True).T
                )

        return self._edge_curl

    @property
    def average_edge_x_to_cell(self):
        if self.is_symmetric:
            raise Exception("There are no x-edges on a cyl symmetric mesh")
        return (
            kron3(
                av(self.shape_cells[2]),
                av(self.shape_cells[1]),
                speye(self.shape_cells[0]),
            )
            * self._deflation_matrix("Ex", as_ones=True).T
        )

    @property
    def average_edge_y_to_cell(self):
        if self.is_symmetric:
            avR = av(self.shape_cells[0])[:, 1:]
            return sp.kron(av(self.shape_cells[2]), avR, format="csr")
        else:
            return (
                kron3(
                    av(self.shape_cells[2]),
                    speye(self.shape_cells[1]),
                    av(self.shape_cells[0]),
                )
                * self._deflation_matrix("Ey", as_ones=True).T
            )

    @property
    def average_edge_z_to_cell(self):
        if self.is_symmetric:
            raise Exception("There are no z-edges on a cyl symmetric mesh")
        return (
            kron3(
                speye(self.shape_cells[2]),
                av(self.shape_cells[1]),
                av(self.shape_cells[0]),
            )
            * self._deflation_matrix("Ez", as_ones=True).T
        )

    @property
    def average_edge_to_cell(self):
        if getattr(self, "_average_edge_to_cell", None) is None:
            # The number of cell centers in each direction
            # n = self.vnC
            if self.is_symmetric:
                self._average_edge_to_cell = self.aveEy2CC
            else:
                self._average_edge_to_cell = (
                    1.0
                    / self.dim
                    * sp.hstack(
                        (self.aveEx2CC, self.aveEy2CC, self.aveEz2CC), format="csr"
                    )
                )
        return self._average_edge_to_cell

    @property
    def average_edge_to_cell_vector(self):
        if self.is_symmetric:
            return self.average_edge_to_cell
        else:
            if getattr(self, "_average_edge_to_cell_vector", None) is None:
                self._average_edge_to_cell_vector = sp.block_diag(
                    (self.aveEx2CC, self.aveEy2CC, self.aveEz2CC), format="csr"
                )
        return self._average_edge_to_cell_vector

    @property
    def average_face_x_to_cell(self):
        avR = av(self.vnC[0])[
            :, 1:
        ]  # TODO: this should be handled by a deflation matrix
        return kron3(speye(self.vnC[2]), speye(self.vnC[1]), avR)

    @property
    def average_face_y_to_cell(self):
        return (
            kron3(speye(self.vnC[2]), av(self.vnC[1]), speye(self.vnC[0]))
            * self._deflation_matrix("Fy", as_ones=True).T
        )

    @property
    def average_face_z_to_cell(self):
        return kron3(av(self.vnC[2]), speye(self.vnC[1]), speye(self.vnC[0]))

    @property
    def average_face_to_cell(self):
        if getattr(self, "_average_face_to_cell", None) is None:
            if self.is_symmetric:
                self._average_face_to_cell = 0.5 * (
                    sp.hstack((self.aveFx2CC, self.aveFz2CC), format="csr")
                )
            else:
                self._average_face_to_cell = (
                    1.0
                    / self.dim
                    * (
                        sp.hstack(
                            (self.aveFx2CC, self.aveFy2CC, self.aveFz2CC), format="csr"
                        )
                    )
                )
        return self._average_face_to_cell

    @property
    def average_face_to_cell_vector(self):
        if getattr(self, "_average_face_to_cell_vector", None) is None:
            # n = self.vnC
            if self.is_symmetric:
                self._average_face_to_cell_vector = sp.block_diag(
                    (self.aveFx2CC, self.aveFz2CC), format="csr"
                )
            else:
                self._average_face_to_cell_vector = sp.block_diag(
                    (self.aveFx2CC, self.aveFy2CC, self.aveFz2CC), format="csr"
                )
        return self._average_face_to_cell_vector

    ####################################################
    # Deflation Matrices
    ####################################################

    def _deflation_matrix(self, location, as_ones=False):
        """
        construct the deflation matrix to remove hanging edges / faces / nodes
        from the operators
        """
        location = self._parse_location_type(location)
        if location not in [
            "nodes",
            "faces",
            "faces_x",
            "faces_y",
            "faces_z",
            "edges",
            "edges_x",
            "edges_y",
            "edges_z",
            "cell_centers",
        ]:
            raise AssertionError(
                "Location must be a grid location, not {}".format(location)
            )
        if location == "cell_centers":
            return speye(self.nC)

        elif location in ["edges", "faces"]:
            if self.is_symmetric:
                if location == "edges":
                    return self._deflation_matrix("edges_y", as_ones=as_ones)
                elif location == "faces":
                    return sp.block_diag(
                        [
                            self._deflation_matrix(location + coord, as_ones=as_ones)
                            for coord in ["_x", "_z"]
                        ]
                    )
            return sp.block_diag(
                [
                    self._deflation_matrix(location + coord, as_ones=as_ones)
                    for coord in ["_x", "_y", "_z"]
                ]
            )

        R = speye(getattr(self, "_n_total_{}".format(location)))
        hanging_dict = getattr(self, "_hanging_{}".format(location))
        nothanging = ~getattr(self, "_ishanging_{}".format(location))

        # remove eliminated edges / faces (eg. Fx just doesn't exist)
        hang = {k: v for k, v in hanging_dict.items() if v is not None}

        values = list(hang.values())
        entries = np.ones(len(values))

        if not as_ones and len(hang) > 0:
            repeats = set(values)
            repeat_locs = [(np.r_[values] == repeat).nonzero()[0] for repeat in repeats]
            for loc in repeat_locs:
                entries[loc] = 1.0 / len(loc)

        Hang = sp.csr_matrix(
            (entries, (values, list(hang.keys()))),
            shape=(
                getattr(self, "_n_total_{}".format(location)),
                getattr(self, "_n_total_{}".format(location)),
            ),
        )
        R = R + Hang

        R = R[nothanging, :]

        if not as_ones:
            R = sdiag(1.0 / R.sum(1)) * R

        return R

    ####################################################
    # Interpolation
    ####################################################

    def get_interpolation_matrix(
        self, loc, location_type="cell_centers", zeros_outside=False, **kwargs
    ):
        """Construct interpolation matrix from mesh

        This method allows the user to construct a sparse linear-interpolation
        matrix which interpolates discrete quantities from mesh centers, nodes,
        edges or faces to an arbitrary set of locations in 3D space.
        Locations are defined in cylindrical coordinates; i.e. :math:`(r, \\phi, z)`.

        Parameters
        ----------
        loc : (n_pts, dim) numpy.ndarray
            Location of points to interpolate to in cylindrical coordinates ; i.e.
            :math:`(r, \\phi, z)`
        location_type : str
            What discrete quantity on the mesh you are interpolating from. Options are:

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

        zeros_outside : bool
            If *False* , nearest neighbour is used to compute the value for
            locations outside the mesh. If *True* , values outside the mesh
            will be equal to zero.

        Returns
        -------
        (n_pts, n_loc_type) scipy.sparse.csr_matrix
            The interpolation matrix

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

        location_type = self._parse_location_type(location_type)

        if self.is_symmetric and location_type in ["edges_x", "edges_z", "faces_y"]:
            raise Exception(
                "Symmetric CylindricalMesh does not support {0!s} interpolation, "
                "as this variable does not exist.".format(location_type)
            )

        if location_type in ["cell_centers_x", "cell_centers_y", "cell_centers_z"]:
            Q = interpolation_matrix(loc, *self.get_tensor("cell_centers"))
            Z = spzeros(loc.shape[0], self.nC)
            if location_type[-1] == "x":
                Q = sp.hstack([Q, Z])
            elif location_type[-1] == "y":
                Q = sp.hstack([Q])
            elif location_type[-1] == "z":
                Q = sp.hstack([Z, Q])

            if zeros_outside:
                indZeros = np.logical_not(self.is_inside(loc))
                loc[indZeros, :] = np.array([v.mean() for v in self.get_tensor("CC")])
                Q[indZeros, :] = 0

            return Q.tocsr()

        return self._getInterpolationMat(loc, location_type, zeros_outside)

    def cartesian_grid(self, location_type="cell_centers", theta_shift=None, **kwargs):
        """
        Takes a grid location ('CC', 'N', 'Ex', 'Ey', 'Ez', 'Fx', 'Fy', 'Fz')
        and returns that grid in cartesian coordinates

        Parameters
        ----------
        location_type : {'CC', 'N', 'Ex', 'Ey', 'Ez', 'Fx', 'Fy', 'Fz'}
            grid location
        theta_shift : float, optional
            shift for theta

        Returns
        -------
        (n_items, dim) numpy.ndarray
            cartesian coordinates for the cylindrical grid
        """
        if "locType" in kwargs:
            warnings.warn(
                "The locType keyword argument has been deprecated, please use location_type. "
                "This will be removed in discretize 1.0.0",
                DeprecationWarning,
            )
            location_type = kwargs["locType"]
        try:
            grid = getattr(self, location_type).copy()
        except AttributeError:
            grid = getattr(self, f"grid{location_type}").copy()
        if theta_shift is not None:
            grid[:, 1] = grid[:, 1] - theta_shift
        return cyl2cart(grid)  # TODO: account for cartesian origin

    def get_interpolation_matrix_cartesian_mesh(
        self, Mrect, location_type="cell_centers", location_type_to=None, **kwargs
    ):
        """Construct projection matrix from ``CylindricalMesh`` to other mesh.

        This method is used to construct a sparse linear interpolation matrix from gridded
        locations on the cylindrical mesh to gridded locations on a different mesh
        type. That is, an interpolation from the centers, nodes, faces or edges of the
        cylindrical mesh to the centers, nodes, faces or edges of another mesh.
        This method is generally used to interpolate from cylindrical meshes to meshes
        defined in Cartesian coordinates; e.g. :class:`~discretize.TensorMesh`,
        :class:`~discretize.TreeMesh` or :class:`~discretize.CurvilinearMesh`.

        Parameters
        ----------
        Mrect : discretize.base.BaseMesh
            the mesh we are interpolating onto
        location_type : {'CC', 'N', 'Ex', 'Ey', 'Ez', 'Fx', 'Fy', 'Fz'}
            gridded locations of the cylindrical mesh.
        location_type_to : {None, 'CC', 'N', 'Ex', 'Ey', 'Ez', 'Fx', 'Fy', 'Fz'}
            gridded locations being interpolated to on the other mesh.
            If *None*, this method will use the same type as *location_type*.

        Returns
        -------
        scipy.sparse.csr_matrix
            interpolation matrix from gridded locations on cylindrical mesh to gridded locations
            on another mesh
        """
        if "locType" in kwargs:
            warnings.warn(
                "The locType keyword argument has been deprecated, please use location_type. "
                "This will be removed in discretize 1.0.0",
                DeprecationWarning,
            )
            location_type = kwargs["locType"]
        if "locTypeTo" in kwargs:
            warnings.warn(
                "The locTypeTo keyword argument has been deprecated, please use location_type_to. "
                "This will be removed in discretize 1.0.0",
                DeprecationWarning,
            )
            location_type_to = kwargs["locTypeTo"]

        location_type = self._parse_location_type(location_type)

        if not self.is_symmetric:
            raise AssertionError(
                "Currently we have not taken into account other projections "
                "for more complicated CylindricalMeshes"
            )

        if location_type_to is None:
            location_type_to = location_type
        location_type_to = self._parse_location_type(location_type_to)

        if location_type == "faces":
            # do this three times for each component
            X = self.get_interpolation_matrix_cartesian_mesh(
                Mrect, location_type="faces_x", location_type_to=location_type_to + "_x"
            )
            Y = self.get_interpolation_matrix_cartesian_mesh(
                Mrect, location_type="faces_y", location_type_to=location_type_to + "_y"
            )
            Z = self.get_interpolation_matrix_cartesian_mesh(
                Mrect, location_type="faces_z", location_type_to=location_type_to + "_z"
            )
            return sp.vstack((X, Y, Z))
        if location_type == "edges":
            X = self.get_interpolation_matrix_cartesian_mesh(
                Mrect, location_type="edges_x", location_type_to=location_type_to + "_x"
            )
            Y = self.get_interpolation_matrix_cartesian_mesh(
                Mrect, location_type="edges_y", location_type_to=location_type_to + "_y"
            )
            Z = spzeros(getattr(Mrect, "n_" + location_type_to + "_z"), self.n_edges)
            return sp.vstack((X, Y, Z))

        grid = getattr(Mrect, location_type_to)
        # This is unit circle stuff, 0 to 2*pi, starting at x-axis, rotating
        # counter clockwise in an x-y slice
        theta = (
            -np.arctan2(
                grid[:, 0] - self.cartesian_origin[0],
                grid[:, 1] - self.cartesian_origin[1],
            )
            + np.pi / 2
        )
        theta[theta < 0] += np.pi * 2.0
        r = (
            (grid[:, 0] - self.cartesian_origin[0]) ** 2
            + (grid[:, 1] - self.cartesian_origin[1]) ** 2
        ) ** 0.5

        if location_type in ["cell_centers", "nodes", "faces_z", "edges_z"]:
            G, proj = np.c_[r, theta, grid[:, 2]], np.ones(r.size)
        else:
            dotMe = {
                "faces_x": Mrect.face_normals[: Mrect.nFx, :],
                "faces_y": Mrect.face_normals[Mrect.nFx : (Mrect.nFx + Mrect.nFy), :],
                "faces_z": Mrect.face_normals[-Mrect.nFz :, :],
                "edges_x": Mrect.edge_tangents[: Mrect.nEx, :],
                "edges_y": Mrect.edge_tangents[Mrect.nEx : (Mrect.nEx + Mrect.nEy), :],
                "edges_z": Mrect.edge_tangents[-Mrect.nEz :, :],
            }[location_type_to]
            if "faces" in location_type:
                normals = np.c_[np.cos(theta), np.sin(theta), np.zeros(theta.size)]
                proj = (normals * dotMe).sum(axis=1)
            elif "edges" in location_type:
                tangents = np.c_[-np.sin(theta), np.cos(theta), np.zeros(theta.size)]
                proj = (tangents * dotMe).sum(axis=1)
            G = np.c_[r, theta, grid[:, 2]]

        interp_type = location_type
        if interp_type == "faces_y":
            interp_type = "faces_x"
        elif interp_type == "edges_x":
            interp_type = "edges_y"

        Pc2r = self.get_interpolation_matrix(G, interp_type)
        Proj = sdiag(proj)
        return Proj * Pc2r

    # DEPRECATIONS
    vol = deprecate_property("cell_volumes", "vol", removal_version="1.0.0")
    area = deprecate_property("face_areas", "area", removal_version="1.0.0")
    areaFx = deprecate_property("face_x_areas", "areaFx", removal_version="1.0.0")
    areaFy = deprecate_property("face_y_areas", "areaFy", removal_version="1.0.0")
    areaFz = deprecate_property("face_z_areas", "areaFz", removal_version="1.0.0")
    edgeEx = deprecate_property("edge_x_lengths", "edgeEx", removal_version="1.0.0")
    edgeEy = deprecate_property("edge_y_lengths", "edgeEy", removal_version="1.0.0")
    edgeEz = deprecate_property("edge_z_lengths", "edgeEz", removal_version="1.0.0")
    edge = deprecate_property("edge_lengths", "edge", removal_version="1.0.0")
    isSymmetric = deprecate_property(
        "is_symmetric", "isSymmetric", removal_version="1.0.0"
    )
    cartesianOrigin = deprecate_property(
        "cartesian_origin", "cartesianOrigin", removal_version="1.0.0"
    )
    getInterpolationMatCartMesh = deprecate_method(
        "get_interpolation_matrix_cartesian_mesh",
        "getInterpolationMatCartMesh",
        removal_version="1.0.0",
    )
    cartesianGrid = deprecate_method(
        "cartesian_grid", "cartesianGrid", removal_version="1.0.0"
    )


@deprecate_class(removal_version="1.0.0")
class CylMesh(CylindricalMesh):
    pass
