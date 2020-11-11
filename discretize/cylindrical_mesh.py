import numpy as np
import properties
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
from discretize.utils.code_utils import (
    deprecate_class,
    deprecate_property,
    deprecate_method,
)
import warnings


class CylindricalMesh(
    BaseTensorMesh, BaseRectangularMesh, InnerProducts, DiffOperators
):
    """
    CylindricalMesh is a mesh class for cylindrical problems. It supports both
    cylindrically symmetric and 3D cylindrical meshes that include an azimuthal
    discretization.

    For a cylindrically symmetric mesh use :code:`h = [hx, 1, hz]`. For example:

    .. plot::
        :include-source:

        import discretize
        from discretize import utils

        cs, nc, npad = 20., 30, 8
        hx = utils.unpack_widths([(cs, npad+10, -0.7), (cs, nc), (cs, npad, 1.3)])
        hz = utils.unpack_widths([(cs, npad ,-1.3), (cs, nc), (cs, npad, 1.3)])
        mesh = discretize.CylindricalMesh([hx, 1, hz], origin=[0, 0, -hz.sum()/2])
        mesh.plot_grid()

    To create a 3D cylindrical mesh, we also include an azimuthal discretization

    .. plot::
        :include-source:

        import discretize
        from discretize import utils

        cs, nc, npad = 20., 30, 8
        nc_theta = 8
        hx = utils.unpack_widths([(cs, npad+10, -0.7), (cs, nc), (cs, npad, 1.3)])
        hy = 2 * np.pi/nc_theta * np.ones(nc_theta)
        hz = utils.unpack_widths([(cs,npad, -1.3), (cs,nc), (cs, npad, 1.3)])
        mesh = discretize.CylindricalMesh([hx, hy, hz], origin=[0, 0, -hz.sum()/2])
        mesh.plot_grid()

    """

    _meshType = "CYL"
    _unitDimensions = [1, 2 * np.pi, 1]
    _aliases = {
        **DiffOperators._aliases,
        **BaseRectangularMesh._aliases,
        **BaseTensorMesh._aliases,
    }

    cartesian_origin = properties.Array(
        "Cartesian origin of the mesh", dtype=float, shape=("*",)
    )

    def __init__(self, h=None, origin=None, **kwargs):
        super().__init__(h=h, origin=origin, **kwargs)
        self.reference_system = "cylindrical"

        if not np.abs(self.h[1].sum() - 2 * np.pi) < 1e-10:
            raise AssertionError("The 2nd dimension must sum to 2*pi")

        if self.dim == 2:
            print("Warning, a disk mesh has not been tested thoroughly.")

        if "cartesian_origin" in kwargs.keys():
            self.cartesian_origin = kwargs.pop("cartesian_origin")
        elif "cartesianOrigin" in kwargs.keys():
            self.cartesian_origin = kwargs.pop("cartesianOrigin")
        else:
            self.cartesian_origin = np.zeros(self.dim)

    @properties.validator("cartesian_origin")
    def check_cartesian_origin_shape(self, change):
        change["value"] = np.array(change["value"], dtype=float).ravel()
        if len(change["value"]) != self.dim:
            raise Exception(
                "Dimension mismatch. The mesh dimension is {}, and the "
                "cartesian_origin provided has length {}".format(
                    self.dim, len(change["value"])
                )
            )

    @property
    def is_symmetric(self):
        """
        Is the mesh cylindrically symmetric?

        Returns
        -------
        bool
            True if the mesh is cylindrically symmetric, False otherwise
        """
        return self.shape_cells[1] == 1

    @property
    def shape_nodes(self):
        vnC = self.shape_cells
        if self.is_symmetric:
            return (vnC[0], 0, vnC[2] + 1)
        else:
            return (vnC[0] + 1, vnC[1], vnC[2] + 1)

    @property
    def _vntN(self):
        vnC = self.shape_cells
        if self.is_symmetric:
            return (vnC[0], 1, vnC[2] + 1)
        else:
            return tuple(x + 1 for x in vnC)

    @property
    def n_nodes(self):
        """
        Returns
        -------
        int
            Total number of nodes
        """
        if self.is_symmetric:
            return 0
        nx, ny, nz = self.shape_nodes
        return (nx - 1) * ny * nz + nz

    @property
    def _vntFx(self):
        """
        vector number of total Fx (prior to deflating)
        """
        return self._vntN[:1] + self.shape_cells[1:]

    @property
    def _ntFx(self):
        """
        number of total Fx (prior to defplating)
        """
        return int(np.prod(self._vntFx))

    @property
    def _nhFx(self):
        """
        Number of hanging Fx
        """
        return int(np.prod(self.shape_cells[1:]))

    @property
    def shape_faces_x(self):
        """
        Returns
        -------
        numpy.ndarray
            Number of x-faces in each direction, (dim, )
        """
        return self.shape_cells

    @property
    def _vntFy(self):
        """
        vector number of total Fy (prior to deflating)
        """
        vnC = self.shape_cells
        return (vnC[0], self._vntN[1]) + vnC[2:]

    @property
    def _ntFy(self):
        """
        number of total Fy (prior to deflating)
        """
        return int(np.prod(self._vntFy))

    @property
    def _nhFy(self):
        """
        number of hanging y-faces
        """
        return int(np.prod(self.shape_cells[::2]))

    @property
    def _vntFz(self):
        """
        vector number of total Fz (prior to deflating)
        """
        return self.shape_cells[:-1] + self._vntN[-1:]

    @property
    def _ntFz(self):
        """
        number of total Fz (prior to deflating)
        """
        return int(np.prod(self._vntFz))

    @property
    def _nhFz(self):
        """
        number of hanging Fz
        """
        return int(np.prod(self.shape_cells[::2]))

    @property
    def _vntEx(self):
        """
        vector number of total Ex (prior to deflating)
        """
        return self.shape_cells[:1] + self._vntN[1:]

    @property
    def _ntEx(self):
        """
        number of total Ex (prior to deflating)
        """
        return int(np.prod(self._vntEx))

    @property
    def _vntEy(self):
        """
        vector number of total Ey (prior to deflating)
        """
        _vntN = self._vntN
        return (_vntN[0], self.shape_cells[1], _vntN[2])

    @property
    def _ntEy(self):
        """
        number of total Ey (prior to deflating)
        """
        return int(np.prod(self._vntEy))

    @property
    def shape_edges_y(self):
        """
        Number of y-edges in each direction

        Returns
        -------
        tuple of ints
            vnEy or None if dim < 2, (dim, )
        """
        return tuple(x + y for x, y in zip(self.shape_cells, [0, 0, 1]))

    @property
    def _vntEz(self):
        """
        vector number of total Ez (prior to deflating)
        """
        return self._vntN[:-1] + self.shape_cells[-1:]

    @property
    def _ntEz(self):
        """
        number of total Ez (prior to deflating)
        """
        return int(np.prod(self._vntEz))

    @property
    def shape_edges_z(self):
        """
        Returns
        -------
        tuple of ints
            Number of z-edges in each direction or None if nCy > 1, (dim, )
        """
        return self.shape_nodes[:-1] + self.shape_cells[-1:]

    @property
    def n_edges_z(self):
        """
        Returns
        -------
        int
            Number of z-edges
        """
        z_shape = self.shape_edges_z
        cell_shape = self.shape_cells
        if self.is_symmetric:
            return int(np.prod(z_shape))
        return int(np.prod([z_shape[0] - 1, z_shape[1], cell_shape[2]])) + cell_shape[2]

    @property
    def cell_centers_x(self):
        """Cell-centered grid vector (1D) in the x direction."""
        return np.r_[0, self.h[0][:-1].cumsum()] + self.h[0] * 0.5

    @property
    def cell_centers_y(self):
        """Cell-centered grid vector (1D) in the y direction."""
        if self.is_symmetric:
            return np.r_[0, self.h[1][:-1]]
        return np.r_[0, self.h[1][:-1].cumsum()] + self.h[1] * 0.5

    @property
    def nodes_x(self):
        """Nodal grid vector (1D) in the x direction."""
        if self.is_symmetric:
            return self.h[0].cumsum()
        return np.r_[0, self.h[0]].cumsum()

    @property
    def _vectorNyFull(self):
        """
        full nodal y vector (prior to deflating)
        """
        if self.is_symmetric:
            return np.r_[0]
        return np.r_[0, self.h[1].cumsum()]

    @property
    def nodes_y(self):
        """Nodal grid vector (1D) in the y direction."""
        # if self.is_symmetric:
        #     # There aren't really any nodes, but all the grids need
        #     # somewhere to live, why not zero?!
        #     return np.r_[0]
        return np.r_[0, self.h[1][:-1].cumsum()]

    @property
    def _edgeExFull(self):
        """
        full x-edge lengths (prior to deflating)
        """
        nx, ny, nz = self._vntN
        return np.kron(np.ones(nz), np.kron(np.ones(ny), self.h[0]))

    @property
    def edge_x_lengths(self):
        """
        x-edge lengths - these are the radial edges. Radial edges only exist
        for a 3D cyl mesh.

        Returns
        -------
        numpy.ndarray
            vector of radial edge lengths
        """
        if getattr(self, "_edgeEx", None) is None:
            self._edgeEx = self._edgeExFull[~self._ishangingEx]
        return self._edgeEx

    @property
    def _edgeEyFull(self):
        """
        full vector of y-edge lengths (prior to deflating)
        """
        if self.is_symmetric:
            return 2 * pi * self.nodes[:, 0]
        return np.kron(np.ones(self._vntN[2]), np.kron(self.h[1], self.nodes_x))

    @property
    def edge_y_lengths(self):
        """
        y-edge lengths - these are the azimuthal edges. Azimuthal edges exist
        for all cylindrical meshes. These are arc-lengths (:math:`\\theta r`)

        Returns
        -------
        numpy.ndarray
            vector of the azimuthal edges
        """
        if getattr(self, "_edgeEy", None) is None:
            if self.is_symmetric:
                self._edgeEy = self._edgeEyFull
            else:
                self._edgeEy = self._edgeEyFull[~self._ishangingEy]
        return self._edgeEy

    @property
    def _edgeEzFull(self):
        """
        full z-edge lengths (prior to deflation)
        """
        nx, ny, nz = self._vntN
        return np.kron(self.h[2], np.kron(np.ones(ny), np.ones(nx)))

    @property
    def edge_z_lengths(self):
        """
        z-edge lengths - these are the vertical edges. Vertical edges only
        exist for a 3D cyl mesh.

        Returns
        -------
        numpy.ndarray
            vector of the vertical edges
        """
        if getattr(self, "_edgeEz", None) is None:
            self._edgeEz = self._edgeEzFull[~self._ishangingEz]
        return self._edgeEz

    @property
    def _edgeFull(self):
        """
        full edge lengths [r-edges, theta-edgesm z-edges] (prior to
        deflation)
        """
        if self.is_symmetric:
            raise NotImplementedError
        else:
            return np.r_[self._edgeExFull, self._edgeEyFull, self._edgeEzFull]

    @property
    def edge_lengths(self):
        """
        Edge lengths

        Returns
        -------
        numpy.ndarray
            vector of edge lengths :math:`(r, \\theta, z)`
        """
        if self.is_symmetric:
            return self.edge_y_lengths
        else:
            return np.r_[self.edge_x_lengths, self.edge_y_lengths, self.edge_z_lengths]

    @property
    def _areaFxFull(self):
        """
        area of x-faces prior to deflation
        """
        if self.is_symmetric:
            return np.kron(self.h[2], 2 * pi * self.nodes_x)
        return np.kron(self.h[2], np.kron(self.h[1], self.nodes_x))

    @property
    def face_x_areas(self):
        """
        Area of the x-faces (radial faces). Radial faces exist on all
        cylindrical meshes

        .. math::
            A_x = r \\theta h_z

        Returns
        -------
        numpy.ndarray
            area of x-faces
        """
        if getattr(self, "_areaFx", None) is None:
            if self.is_symmetric:
                self._areaFx = self._areaFxFull
            else:
                self._areaFx = self._areaFxFull[~self._ishangingFx]
        return self._areaFx

    @property
    def _areaFyFull(self):
        """
        Area of y-faces (Azimuthal faces), prior to deflation.
        """
        return np.kron(self.h[2], np.kron(np.ones(self._vntN[1]), self.h[0]))

    @property
    def face_y_areas(self):
        """
        Area of y-faces (Azimuthal faces). Azimuthal faces exist only on 3D
        cylindrical meshes.

        .. math::
            A_y = h_x h_z

        Returns
        -------
        numpy.ndarray
            area of y-faces
        """
        if getattr(self, "_areaFy", None) is None:
            if self.is_symmetric:
                raise Exception("There are no y-faces on the Cyl Symmetric mesh")
            self._areaFy = self._areaFyFull[~self._ishangingFy]
        return self._areaFy

    @property
    def _areaFzFull(self):
        """
        area of z-faces prior to deflation
        """
        if self.is_symmetric:
            return np.kron(
                np.ones_like(self.nodes_z),
                pi * (self.nodes_x ** 2 - np.r_[0, self.nodes_x[:-1]] ** 2),
            )
        return np.kron(
            np.ones(self._vntN[2]),
            np.kron(
                self.h[1],
                0.5 * (self.nodes_x[1:] ** 2 - self.nodes_x[:-1] ** 2),
            ),
        )

    @property
    def face_z_areas(self):
        """
        Area of z-faces.

        .. math::
            A_z = \\frac{\\theta}{2} (r_2^2 - r_1^2)z

        Returns
        -------
        numpy.ndarray
            area of the z-faces
        """
        if getattr(self, "_areaFz", None) is None:
            if self.is_symmetric:
                self._areaFz = self._areaFzFull
            else:
                self._areaFz = self._areaFzFull[~self._ishangingFz]
        return self._areaFz

    @property
    def _areaFull(self):
        """
        Area of all faces (prior to delflation)
        """
        return np.r_[self._areaFxFull, self._areaFyFull, self._areaFzFull]

    @property
    def face_areas(self):
        """
        Face areas

        For a 3D cyl mesh: [radial, azimuthal, vertical], while a cylindrically
        symmetric mesh doesn't have y-Faces, so it returns [radial, vertical]

        Returns
        -------
        numpy.ndarray
            face areas
        """
        # if getattr(self, '_area', None) is None:
        if self.is_symmetric:
            return np.r_[self.face_x_areas, self.face_z_areas]
        else:
            return np.r_[self.face_x_areas, self.face_y_areas, self.face_z_areas]

    @property
    def cell_volumes(self):
        """
        Volume of each cell

        Returns
        -------
        numpy.ndarray
            cell volumes
        """
        if getattr(self, "_vol", None) is None:
            if self.is_symmetric:
                az = pi * (
                    self.nodes_x ** 2 - np.r_[0, self.nodes_x[:-1]] ** 2
                )
                self._vol = np.kron(self.h[2], az)
            else:
                self._vol = np.kron(
                    self.h[2],
                    np.kron(
                        self.h[1],
                        0.5
                        * (self.nodes_x[1:] ** 2 - self.nodes_x[:-1] ** 2),
                    ),
                )
        return self._vol

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
    #          ishangingFxBool = np.kron(
    #              np.ones(self.shape_cells[2], dtype=bool),  # 1 * 0 == 0
    #              np.kron(np.ones(self.shape_cells[1], dtype=bool), hang_x)
    #          )
    #          return self._ishangingFxBool
    #
    #
    #   This is equivalent to forming the 3D matrix and indexing the
    #   corresponding rows and columns (here, the hanging faces are all of
    #   the first x-faces along the axis of symmetry):
    #
    #         hang_x = np.zeros(self._vntFx, dtype=bool)
    #         hang_x[0, :, :] = True
    #         isHangingFxBool = mkvc(hang_x)
    #
    #
    # but krons of bools is more efficient.
    #
    ###########################################################################

    @property
    def _ishangingFx(self):
        """
        bool vector indicating if an x-face is hanging or not
        """
        if getattr(self, "_ishangingFxBool", None) is None:

            # the following is equivalent to
            #     hang_x = np.zeros(self._vntFx, dtype=bool)
            #     hang_x[0, :, :] = True
            #     isHangingFxBool = mkvc(hang_x)
            #
            # but krons of bools is more efficient

            hang_x = np.zeros(self._vntN[0], dtype=bool)
            hang_x[0] = True
            self._ishangingFxBool = np.kron(
                np.ones(self.shape_cells[2], dtype=bool),  # 1 * 0 == 0
                np.kron(np.ones(self.shape_cells[1], dtype=bool), hang_x),
            )
        return self._ishangingFxBool

    @property
    def _hangingFx(self):
        """
        dictionary of the indices of the hanging x-faces (keys) and a list
        of indices that the eliminated faces map to (if applicable)
        """
        if getattr(self, "_hangingFxDict", None) is None:
            self._hangingFxDict = dict(
                zip(np.nonzero(self._ishangingFx)[0].tolist(), [None] * self._nhFx)
            )
        return self._hangingFxDict

    @property
    def _ishangingFy(self):
        """
        bool vector indicating if a y-face is hanging or not
        """

        if getattr(self, "_ishangingFyBool", None) is None:
            hang_y = np.zeros(self._vntN[1], dtype=bool)
            hang_y[-1] = True
            self._ishangingFyBool = np.kron(
                np.ones(self.shape_cells[2], dtype=bool),
                np.kron(hang_y, np.ones(self.shape_cells[0], dtype=bool)),
            )
        return self._ishangingFyBool

    @property
    def _hangingFy(self):
        """
        dictionary of the indices of the hanging y-faces (keys) and a list
        of indices that the eliminated faces map to (if applicable)
        """
        if getattr(self, "_hangingFyDict", None) is None:
            deflate_y = np.zeros(self._vntN[1], dtype=bool)
            deflate_y[0] = True
            deflateFy = np.nonzero(
                np.kron(
                    np.ones(self.shape_cells[2], dtype=bool),
                    np.kron(deflate_y, np.ones(self.shape_cells[0], dtype=bool)),
                )
            )[0].tolist()
            self._hangingFyDict = dict(
                zip(np.nonzero(self._ishangingFy)[0].tolist(), deflateFy)
            )
        return self._hangingFyDict

    @property
    def _ishangingFz(self):
        """
        bool vector indicating if a z-face is hanging or not
        """
        if getattr(self, "_ishangingFzBool", None) is None:
            self._ishangingFzBool = np.zeros(self._ntFz, dtype=bool)
        return self._ishangingFzBool

    @property
    def _hangingFz(self):
        """
        dictionary of the indices of the hanging z-faces (keys) and a list
        of indices that the eliminated faces map to (if applicable)
        """
        return {}

    @property
    def _ishangingEx(self):
        """
        bool vector indicating if a x-edge is hanging or not
        """
        if getattr(self, "_ishangingExBool", None) is None:
            nx, ny, nz = self._vntN
            hang_y = np.zeros(ny, dtype=bool)
            hang_y[-1] = True
            self._ishangingExBool = np.kron(
                np.ones(nz, dtype=bool),
                np.kron(hang_y, np.ones(self.shape_cells[0], dtype=bool)),
            )
        return self._ishangingExBool

    @property
    def _hangingEx(self):
        """
        dictionary of the indices of the hanging x-edges (keys) and a list
        of indices that the eliminated faces map to (if applicable)
        """
        if getattr(self, "_hangingExDict", None) is None:
            nx, ny, nz = self._vntN
            deflate_y = np.zeros(ny, dtype=bool)
            deflate_y[0] = True
            deflateEx = np.nonzero(
                np.kron(
                    np.ones(nz, dtype=bool),
                    np.kron(deflate_y, np.ones(self.shape_cells[0], dtype=bool)),
                )
            )[0].tolist()
            self._hangingExDict = dict(
                zip(np.nonzero(self._ishangingEx)[0].tolist(), deflateEx)
            )
        return self._hangingExDict

    @property
    def _ishangingEy(self):
        """
        bool vector indicating if a y-edge is hanging or not
        """
        if getattr(self, "_ishangingEyBool", None) is None:
            nx, ny, nz = self._vntN
            hang_x = np.zeros(nx, dtype=bool)
            hang_x[0] = True
            self._ishangingEyBool = np.kron(
                np.ones(nz, dtype=bool),
                np.kron(np.ones(self.shape_cells[1], dtype=bool), hang_x),
            )
        return self._ishangingEyBool

    @property
    def _hangingEy(self):
        """
        dictionary of the indices of the hanging y-edges (keys) and a list
        of indices that the eliminated faces map to (if applicable)
        """
        if getattr(self, "_hangingEyDict", None) is None:
            self._hangingEyDict = dict(
                zip(
                    np.nonzero(self._ishangingEy)[0].tolist(),
                    [None] * len(self._ishangingEyBool),
                )
            )
        return self._hangingEyDict

    @property
    def _axis_of_symmetry_Ez(self):
        """
        bool vector indicating if a z-edge is along the axis of symmetry or not
        """
        if getattr(self, "_axis_of_symmetry_EzBool", None) is None:
            nx, ny, nz = self._vntN
            axis_x = np.zeros(nx, dtype=bool)
            axis_x[0] = True

            axis_y = np.zeros(ny, dtype=bool)
            axis_y[0] = True
            self._axis_of_symmetry_EzBool = np.kron(
                np.ones(self.shape_cells[2], dtype=bool), np.kron(axis_y, axis_x)
            )
        return self._axis_of_symmetry_EzBool

    @property
    def _ishangingEz(self):
        """
        bool vector indicating if a z-edge is hanging or not
        """
        if getattr(self, "_ishangingEzBool", None) is None:
            if self.is_symmetric:
                self._ishangingEzBool = np.ones(self._ntEz, dtype=bool)
            else:
                nx, ny, nz = self._vntN
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

                self._ishangingEzBool = hangingEz & ~self._axis_of_symmetry_Ez

        return self._ishangingEzBool

    @property
    def _hangingEz(self):
        """
        dictionary of the indices of the hanging z-edges (keys) and a list
        of indices that the eliminated faces map to (if applicable)
        """
        if getattr(self, "_hangingEzDict", None) is None:
            nx, ny, nz = self._vntN
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
            deflate = zip(np.nonzero(self._ishangingEz)[0].tolist(), deflateEz)

            self._hangingEzDict = dict(deflate)
        return self._hangingEzDict

    @property
    def _axis_of_symmetry_N(self):
        """
        bool vector indicating if a node is along the axis of symmetry or not
        """
        if getattr(self, "_axis_of_symmetry_NBool", None) is None:
            nx, ny, nz = self._vntN
            axis_x = np.zeros(nx, dtype=bool)
            axis_x[0] = True

            axis_y = np.zeros(ny, dtype=bool)
            axis_y[0] = True
            self._axis_of_symmetry_NBool = np.kron(
                np.ones(nz, dtype=bool), np.kron(axis_y, axis_x)
            )
        return self._axis_of_symmetry_NBool

    @property
    def _ishangingN(self):
        """
        bool vector indicating if a node is hanging or not
        """
        if getattr(self, "_ishangingNBool", None) is None:
            nx, ny, nz = self._vntN
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

            self._ishangingNBool = hangingN & ~self._axis_of_symmetry_N

        return self._ishangingNBool

    @property
    def _hangingN(self):
        """
        dictionary of the indices of the hanging nodes (keys) and a list
        of indices that the eliminated faces map to (if applicable)
        """
        if getattr(self, "_hangingNDict", None) is None:
            nx, ny, nz = self._vntN
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
            self._hangingNDict = dict(
                zip(np.nonzero(self._ishangingN)[0].tolist(), deflateN)
            )
        return self._hangingNDict

    ####################################################
    # Grids
    ####################################################

    @property
    def _gridNFull(self):
        """
        Full Nodal grid (including hanging nodes)
        """
        return ndgrid([self.nodes_x, self._vectorNyFull, self.nodes_z])

    @property
    def nodes(self):
        """
        Nodal grid in cylindrical coordinates :math:`(r, \\theta, z)`.
        Nodes do not exist in a cylindrically symmetric mesh.

        Returns
        -------
        numpy.ndarray
            grid locations of nodes
        """
        if self.is_symmetric:
            self._gridN = self._gridNFull
        if getattr(self, "_gridN", None) is None:
            self._gridN = self._gridNFull[~self._ishangingN, :]
        return self._gridN

    @property
    def _gridFxFull(self):
        """
        Full Fx grid (including hanging faces)
        """
        return ndgrid(
            [self.nodes_x, self.cell_centers_y, self.cell_centers_z]
        )

    @property
    def faces_x(self):
        """
        Grid of x-faces (radial-faces) in cylindrical coordinates
        :math:`(r, \\theta, z)`.

        Returns
        -------
        numpy.ndarray
            grid locations of radial faces
        """
        if getattr(self, "_gridFx", None) is None:
            if self.is_symmetric:
                return super().faces_x
            else:
                self._gridFx = self._gridFxFull[~self._ishangingFx, :]
        return self._gridFx

    @property
    def _gridEyFull(self):
        """
        Full grid of y-edges (including eliminated edges)
        """
        return super().edges_y

    @property
    def edges_y(self):
        """
        Grid of y-edges (azimuthal-faces) in cylindrical coordinates
        :math:`(r, \\theta, z)`.

        Returns
        -------
        numpy.ndarray
            grid locations of azimuthal faces
        """
        if getattr(self, "_gridEy", None) is None:
            if self.is_symmetric:
                return self._gridEyFull
            else:
                self._gridEy = self._gridEyFull[~self._ishangingEy, :]
        return self._gridEy

    @property
    def _gridEzFull(self):
        """
        Full z-edge grid (including hanging edges)
        """
        return ndgrid([self.nodes_x, self._vectorNyFull, self.cell_centers_z])

    @property
    def edges_z(self):
        """
        Grid of z-faces (vertical-faces) in cylindrical coordinates
        :math:`(r, \\theta, z)`.

        Returns
        -------
        numpy.ndarray
            grid locations of radial faces
        """
        if getattr(self, "_gridEz", None) is None:
            if self.is_symmetric:
                self._gridEz = None
            else:
                self._gridEz = self._gridEzFull[~self._ishangingEz, :]
        return self._gridEz

    ####################################################
    # Operators
    ####################################################

    @property
    def face_divergence(self):
        """
        Construct divergence operator (faces to cell-centres).
        """
        if getattr(self, "_faceDiv", None) is None:
            # Compute faceDivergence operator on faces
            D1 = self.face_x_divergence
            D3 = self.face_z_divergence
            if self.is_symmetric:
                D = sp.hstack((D1, D3), format="csr")
            elif self.shape_cells[1] > 1:
                D2 = self.face_y_divergence
                D = sp.hstack((D1, D2, D3), format="csr")
            self._faceDiv = D
        return self._faceDiv

    @property
    def face_x_divergence(self):
        """
        Construct divergence operator in the x component
        (faces to cell-centres).
        """
        if getattr(self, "_faceDivx", None) is None:
            if self.is_symmetric:
                ncx, ncy, ncz = self.shape_cells
                D1 = kron3(speye(ncz), speye(ncy), ddx(ncx)[:, 1:])
            else:
                D1 = super()._faceDivStencilx

            S = self._areaFxFull
            V = self.cell_volumes
            self._faceDivx = sdiag(1 / V) * D1 * sdiag(S)

            if not self.is_symmetric:
                self._faceDivx = (
                    self._faceDivx * self._deflationMatrix("Fx", as_ones=True).T
                )

        return self._faceDivx

    @property
    def face_y_divergence(self):
        """
        Construct divergence operator in the y component
        (faces to cell-centres).
        """
        if getattr(self, "_faceDivy", None) is None:
            D2 = super()._faceDivStencily
            S = self._areaFyFull  # self.reshape(self.face_areas, 'F', 'Fy', 'V')
            V = self.cell_volumes
            self._faceDivy = (
                sdiag(1 / V)
                * D2
                * sdiag(S)
                * self._deflationMatrix("Fy", as_ones=True).T
            )
        return self._faceDivy

    @property
    def face_z_divergence(self):
        """
        Construct divergence operator in the z component
        (faces to cell-centres).
        """
        if getattr(self, "_faceDivz", None) is None:
            D3 = super()._faceDivStencilz
            S = self._areaFzFull
            V = self.cell_volumes
            self._faceDivz = sdiag(1 / V) * D3 * sdiag(S)
        return self._faceDivz

    # @property
    # def _cellGradxStencil(self):
    #     n = self.vnC

    #     if self.is_symmetric:
    #         G1 = sp.kron(speye(n[2]), ddxCellGrad(n[0], BC))
    #     else:
    #         G1 = self._deflationMatrix('Fx').T * kron3(
    #             speye(n[2]), speye(n[1]), ddxCellGrad(n[0], BC)
    #         )
    #     return G1

    @property
    def cell_gradient_x(self):
        raise NotImplementedError("Cell Grad is not yet implemented.")
        # if getattr(self, '_cellGradx', None) is None:
        #     G1 = super(CylindricalMesh, self)._cellGradxStencil
        #     V = self._deflationMatrix('F', withHanging='True', as_ones='True')*self.aveCC2F*self.cell_volumes
        #     A = self.face_areas
        #     L = (A/V)[:self._ntFx]
        #     # L = self.reshape(L, 'F', 'Fx', 'V')
        #     # L = A[:self.nFx] / V
        #     self._cellGradx = self._deflationMatrix('Fx')*sdiag(L)*G1
        # return self._cellGradx

    @property
    def _cellGradyStencil(self):
        raise NotImplementedError("Cell Grad is not yet implemented.")

    @property
    def _cellGradzStencil(self):
        raise NotImplementedError("Cell Grad is not yet implemented.")

    @property
    def _cellGradStencil(self):
        raise NotImplementedError("Cell Grad is not yet implemented.")

    @property
    def cell_gradient(self):
        raise NotImplementedError("Cell Grad is not yet implemented.")

    # @property
    # def _nodalGradStencilx(self):
    #     if self.is_symmetric:
    #         return None
    #     return kron3(speye(self.shape_nodes[2]), speye(self.shape_nodes[1]), ddx(self.shape_cells[0]))

    # @property
    # def _nodalGradStencily(self):
    #     if self.is_symmetric:
    #         None
    #         # return kron3(speye(self.shape_nodes[2]), ddx(self.shape_cells[1]), speye(self.shape_nodes[0])) * self._deflationMatrix('Ey')
    #     return kron3(speye(self.shape_nodes[2]), ddx(self.shape_cells[1]), speye(self.shape_nodes[0]))

    # @property
    # def _nodalGradStencilz(self):
    #     if self.is_symmetric:
    #         return None
    #     return kron3(ddx(self.shape_cells[2]), speye(self.shape_nodes[1]), speye(self.shape_nodes[0]))

    # @property
    # def _nodalGradStencil(self):
    #     if self.is_symmetric:
    #         return None
    #     else:
    #         G = self._deflationMatrix('E').T * sp.vstack((
    #             self._nodalGradStencilx,
    #             self._nodalGradStencily,
    #             self._nodalGradStencilz
    #         ), format="csr") * self._deflationMatrix('N')
    #     return G

    @property
    def nodal_gradient(self):
        """Construct gradient operator (nodes to edges)."""
        if self.is_symmetric:
            return None
        raise NotImplementedError("nodalGrad not yet implemented")

    @property
    def nodal_laplacian(self):
        """Construct laplacian operator (nodes to edges)."""
        raise NotImplementedError("nodalLaplacian not yet implemented")

    @property
    def edge_curl(self):
        """
        The edgeCurl (edges to faces)

        Returns
        -------
        scipy.sparse.csr_matrix
            edge curl operator
        """
        if getattr(self, "_edgeCurl", None) is None:
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
                self._edgeCurl = sdiag(1 / A) * sp.vstack((Dz, Dr)) * sdiag(E)
            else:
                self._edgeCurl = (
                    sdiag(1 / self.face_areas)
                    * self._deflationMatrix("F", as_ones=False)
                    * self._edgeCurlStencil
                    * sdiag(self._edgeFull)
                    * self._deflationMatrix("E", as_ones=True).T
                )

        return self._edgeCurl

    @property
    def average_edge_x_to_cell(self):
        """
        averaging operator of x-edges (radial) to cell centers

        Returns
        -------
        scipy.sparse.csr_matrix
            matrix that averages from x-edges to cell centers
        """
        if self.is_symmetric:
            raise Exception("There are no x-edges on a cyl symmetric mesh")
        return (
            kron3(
                av(self.shape_cells[2]),
                av(self.shape_cells[1]),
                speye(self.shape_cells[0]),
            )
            * self._deflationMatrix("Ex", as_ones=True).T
        )

    @property
    def average_edge_y_to_cell(self):
        """
        averaging operator of y-edges (azimuthal) to cell centers

        Returns
        -------
        scipy.sparse.csr_matrix
            matrix that averages from y-edges to cell centers
        """
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
                * self._deflationMatrix("Ey", as_ones=True).T
            )

    @property
    def average_edge_z_to_cell(self):
        """
        averaging operator of z-edges to cell centers

        Returns
        -------
        scipy.sparse.csr_matrix
            matrix that averages from z-edges to cell centers
        """
        if self.is_symmetric:
            raise Exception("There are no z-edges on a cyl symmetric mesh")
        return (
            kron3(
                speye(self.shape_cells[2]),
                av(self.shape_cells[1]),
                av(self.shape_cells[0]),
            )
            * self._deflationMatrix("Ez", as_ones=True).T
        )

    @property
    def average_edge_to_cell(self):
        """
        averaging operator of edges to cell centers

        Returns
        -------
        scipy.sparse.csr_matrix
            matrix that averages from edges to cell centers
        """
        if getattr(self, "_aveE2CC", None) is None:
            # The number of cell centers in each direction
            # n = self.vnC
            if self.is_symmetric:
                self._aveE2CC = self.aveEy2CC
            else:
                self._aveE2CC = (
                    1.0
                    / self.dim
                    * sp.hstack(
                        (self.aveEx2CC, self.aveEy2CC, self.aveEz2CC), format="csr"
                    )
                )
        return self._aveE2CC

    @property
    def average_edge_to_cell_vector(self):
        """
        averaging operator of edges to a cell centered vector

        Returns
        -------
        scipy.sparse.csr_matrix
            matrix that averages from edges to cell centered vectors
        """
        if self.is_symmetric:
            return self.average_edge_to_cell
        else:
            if getattr(self, "_aveE2CCV", None) is None:
                self._aveE2CCV = sp.block_diag(
                    (self.aveEx2CC, self.aveEy2CC, self.aveEz2CC), format="csr"
                )
        return self._aveE2CCV

    @property
    def average_face_x_to_cell(self):
        """
        averaging operator of x-faces (radial) to cell centers

        Returns
        -------
        scipy.sparse.csr_matrix
            matrix that averages from x-faces to cell centers
        """
        avR = av(self.vnC[0])[
            :, 1:
        ]  # TODO: this should be handled by a deflation matrix
        return kron3(speye(self.vnC[2]), speye(self.vnC[1]), avR)

    @property
    def average_face_y_to_cell(self):
        """
        averaging operator of y-faces (azimuthal) to cell centers

        Returns
        -------
        scipy.sparse.csr_matrix
            matrix that averages from y-faces to cell centers
        """
        return (
            kron3(speye(self.vnC[2]), av(self.vnC[1]), speye(self.vnC[0]))
            * self._deflationMatrix("Fy", as_ones=True).T
        )

    @property
    def average_face_z_to_cell(self):
        """
        averaging operator of z-faces (vertical) to cell centers

        Returns
        -------
        scipy.sparse.csr_matrix
            matrix that averages from z-faces to cell centers
        """

        return kron3(av(self.vnC[2]), speye(self.vnC[1]), speye(self.vnC[0]))

    @property
    def average_face_to_cell(self):
        """
        averaging operator of faces to cell centers

        Returns
        -------
        scipy.sparse.csr_matrix
            matrix that averages from faces to cell centers
        """
        if getattr(self, "_aveF2CC", None) is None:
            if self.is_symmetric:
                self._aveF2CC = 0.5 * (
                    sp.hstack((self.aveFx2CC, self.aveFz2CC), format="csr")
                )
            else:
                self._aveF2CC = (
                    1.0
                    / self.dim
                    * (
                        sp.hstack(
                            (self.aveFx2CC, self.aveFy2CC, self.aveFz2CC), format="csr"
                        )
                    )
                )
        return self._aveF2CC

    @property
    def average_face_to_cell_vector(self):
        """
        averaging operator of x-faces (radial) to cell centered vectors

        Returns
        -------
        scipy.sparse.csr_matrix
            matrix that averages from faces to cell centered vectors
        """
        if getattr(self, "_aveF2CCV", None) is None:
            # n = self.vnC
            if self.is_symmetric:
                self._aveF2CCV = sp.block_diag(
                    (self.aveFx2CC, self.aveFz2CC), format="csr"
                )
            else:
                self._aveF2CCV = sp.block_diag(
                    (self.aveFx2CC, self.aveFy2CC, self.aveFz2CC), format="csr"
                )
        return self._aveF2CCV

    ####################################################
    # Deflation Matrices
    ####################################################

    def _deflationMatrix(self, location, as_ones=False):
        """
        construct the deflation matrix to remove hanging edges / faces / nodes
        from the operators
        """
        if location not in ["N", "F", "Fx", "Fy", "Fz", "E", "Ex", "Ey", "Ez", "CC"]:
            raise AssertionError(
                "Location must be a grid location, not {}".format(location)
            )
        if location == "CC":
            return speye(self.nC)

        elif location in ["E", "F"]:
            if self.is_symmetric:
                if location == "E":
                    return self._deflationMatrix("Ey", as_ones=as_ones)
                elif location == "F":
                    return sp.block_diag(
                        [
                            self._deflationMatrix(location + coord, as_ones=as_ones)
                            for coord in ["x", "z"]
                        ]
                    )
            return sp.block_diag(
                [
                    self._deflationMatrix(location + coord, as_ones=as_ones)
                    for coord in ["x", "y", "z"]
                ]
            )

        R = speye(getattr(self, "_nt{}".format(location)))
        hanging_dict = getattr(self, "_hanging{}".format(location))
        nothanging = ~getattr(self, "_ishanging{}".format(location))

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
                getattr(self, "_nt{}".format(location)),
                getattr(self, "_nt{}".format(location)),
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
        self, loc, location_type="CC", zeros_outside=False, **kwargs
    ):
        """Produces interpolation matrix

        Parameters
        ----------
        loc : numpy.ndarray
            Location of points to interpolate to

        location_type : str
            What to interpolate location_type can be::

            'Ex'    -> x-component of field defined on edges
            'Ey'    -> y-component of field defined on edges
            'Ez'    -> z-component of field defined on edges
            'Fx'    -> x-component of field defined on faces
            'Fy'    -> y-component of field defined on faces
            'Fz'    -> z-component of field defined on faces
            'N'     -> scalar field defined on nodes
            'CC'    -> scalar field defined on cell centers
            'CCVx'  -> x-component of vector field defined on cell centers
            'CCVy'  -> y-component of vector field defined on cell centers
            'CCVz'  -> z-component of vector field defined on cell centers

        Returns
        -------
        scipy.sparse.csr_matrix
            M, the interpolation matrix

        """
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

        if self.is_symmetric and location_type in ["Ex", "Ez", "Fy"]:
            raise Exception(
                "Symmetric CylindricalMesh does not support {0!s} interpolation, "
                "as this variable does not exist.".format(location_type)
            )

        if location_type in ["CCVx", "CCVy", "CCVz"]:
            Q = interpolation_matrix(loc, *self.get_tensor("CC"))
            Z = spzeros(loc.shape[0], self.nC)
            if location_type == "CCVx":
                Q = sp.hstack([Q, Z])
            elif location_type == "CCVy":
                Q = sp.hstack([Q])
            elif location_type == "CCVz":
                Q = sp.hstack([Z, Q])

            if zeros_outside:
                indZeros = np.logical_not(self.is_inside(loc))
                loc[indZeros, :] = np.array([v.mean() for v in self.get_tensor("CC")])
                Q[indZeros, :] = 0

            return Q.tocsr()

        return self._getInterpolationMat(loc, location_type, zeros_outside)

    def cartesian_grid(self, location_type="CC", theta_shift=None, **kwargs):
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
        numpy.ndarray
            cartesian coordinates for the cylindrical grid
        """
        if "locType" in kwargs:
            warnings.warn(
                "The locType keyword argument has been deprecated, please use location_type. "
                "This will be removed in discretize 1.0.0",
                FutureWarning,
            )
            location_type = kwargs["locType"]
        grid = getattr(self, "grid{}".format(location_type)).copy()
        if theta_shift is not None:
            grid[:, 1] = grid[:, 1] - theta_shift
        return cyl2cart(grid)  # TODO: account for cartesian origin

    def get_interpolation_matrix_cartesian_mesh(
        self, Mrect, location_type="CC", location_type_to=None, **kwargs
    ):
        """
        Takes a cartesian mesh and returns a projection to translate onto
        the cartesian grid.

        Parameters
        ----------
        Mrect : discretize.base.BaseMesh
            the mesh to interpolate on to
        location_type : {'CC', 'N', 'Ex', 'Ey', 'Ez', 'Fx', 'Fy', 'Fz'}
            grid location
        location_type_to : {'CC', 'N', 'Ex', 'Ey', 'Ez', 'Fx', 'Fy', 'Fz'}, or None, optional
            grid location to interpolate to. If None, the same grid type as `location_type` will be assumed

        Returns
        -------
        scipy.sparse.csr_matrix
            M, the interpolation matrix
        """
        if "locType" in kwargs:
            warnings.warn(
                "The locType keyword argument has been deprecated, please use location_type. "
                "This will be removed in discretize 1.0.0",
                FutureWarning,
            )
            location_type = kwargs["locType"]
        if "locTypeTo" in kwargs:
            warnings.warn(
                "The locTypeTo keyword argument has been deprecated, please use location_type_to. "
                "This will be removed in discretize 1.0.0",
                FutureWarning,
            )
            location_type_to = kwargs["locTypeTo"]

        if not self.is_symmetric:
            raise AssertionError(
                "Currently we have not taken into account other projections "
                "for more complicated CylindricalMeshes"
            )

        if location_type_to is None:
            location_type_to = location_type

        if location_type == "F":
            # do this three times for each component
            X = self.get_interpolation_matrix_cartesian_mesh(
                Mrect, location_type="Fx", location_type_to=location_type_to + "x"
            )
            Y = self.get_interpolation_matrix_cartesian_mesh(
                Mrect, location_type="Fy", location_type_to=location_type_to + "y"
            )
            Z = self.get_interpolation_matrix_cartesian_mesh(
                Mrect, location_type="Fz", location_type_to=location_type_to + "z"
            )
            return sp.vstack((X, Y, Z))
        if location_type == "E":
            X = self.get_interpolation_matrix_cartesian_mesh(
                Mrect, location_type="Ex", location_type_to=location_type_to + "x"
            )
            Y = self.get_interpolation_matrix_cartesian_mesh(
                Mrect, location_type="Ey", location_type_to=location_type_to + "y"
            )
            Z = spzeros(getattr(Mrect, "n" + location_type_to + "z"), self.nE)
            return sp.vstack((X, Y, Z))

        grid = getattr(Mrect, "grid" + location_type_to)
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

        if location_type in ["CC", "N", "Fz", "Ez"]:
            G, proj = np.c_[r, theta, grid[:, 2]], np.ones(r.size)
        else:
            dotMe = {
                "Fx": Mrect.face_normals[:Mrect.nFx, :],
                "Fy": Mrect.face_normals[Mrect.nFx:(Mrect.nFx + Mrect.nFy), :],
                "Fz": Mrect.face_normals[-Mrect.nFz:, :],
                "Ex": Mrect.edge_tangents[: Mrect.nEx, :],
                "Ey": Mrect.edge_tangents[Mrect.nEx: (Mrect.nEx + Mrect.nEy), :],
                "Ez": Mrect.edge_tangents[-Mrect.nEz:, :],
            }[location_type_to]
            if "F" in location_type:
                normals = np.c_[np.cos(theta), np.sin(theta), np.zeros(theta.size)]
                proj = (normals * dotMe).sum(axis=1)
            if "E" in location_type:
                tangents = np.c_[-np.sin(theta), np.cos(theta), np.zeros(theta.size)]
                proj = (tangents * dotMe).sum(axis=1)
            G = np.c_[r, theta, grid[:, 2]]

        interpType = location_type
        if interpType == "Fy":
            interpType = "Fx"
        elif interpType == "Ex":
            interpType = "Ey"

        Pc2r = self.get_interpolation_matrix(G, interpType)
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
