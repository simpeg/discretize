from __future__ import print_function

import numpy as np
import properties
import scipy.sparse as sp
from scipy.constants import pi

from .utils import (
    kron3, ndgrid, av, speye, ddx, sdiag, interpmat, spzeros, cyl2cart
)
from .TensorMesh import BaseTensorMesh, BaseRectangularMesh
from .InnerProducts import InnerProducts
from .View import CylView
from .DiffOperators import DiffOperators


class CylMesh(
    BaseTensorMesh, BaseRectangularMesh, InnerProducts, CylView, DiffOperators
):
    """
    CylMesh is a mesh class for cylindrical problems. It supports both
    cylindrically symmetric and 3D cylindrical meshes that include an azimuthal
    discretization.

    For a cylindrically symmetric mesh use :code:`h = [hx, 1, hz]`. For example:

    .. plot::
        :include-source:

        import discretize
        from discretize import utils

        cs, nc, npad = 20., 30, 8
        hx = utils.meshTensor([(cs, npad+10, -0.7), (cs, nc), (cs, npad, 1.3)])
        hz = utils.meshTensor([(cs, npad ,-1.3), (cs, nc), (cs, npad, 1.3)])
        mesh = discretize.CylMesh([hx, 1, hz], x0=[0, 0, -hz.sum()/2])
        mesh.plotGrid()

    To create a 3D cylindrical mesh, we also include an azimuthal discretization

    .. plot::
        :include-source:

        import discretize
        from discretize import utils

        cs, nc, npad = 20., 30, 8
        nc_theta = 8
        hx = utils.meshTensor([(cs, npad+10, -0.7), (cs, nc), (cs, npad, 1.3)])
        hy = 2 * np.pi/nc_theta * np.ones(nc_theta)
        hz = utils.meshTensor([(cs,npad, -1.3), (cs,nc), (cs, npad, 1.3)])
        mesh = discretize.CylMesh([hx, hy, hz], x0=[0, 0, -hz.sum()/2])
        mesh.plotGrid()

    """

    _meshType = 'CYL'
    _unitDimensions = [1, 2*np.pi, 1]

    cartesianOrigin = properties.Array(
        "Cartesian origin of the mesh",
        dtype=float,
        shape=('*',)
    )

    def __init__(self, h=None, x0=None, **kwargs):
        super(CylMesh, self).__init__(h=h, x0=x0, **kwargs)
        self.reference_system = 'cylindrical'

        if not np.abs(self.hy.sum() - 2*np.pi) < 1e-10:
            raise AssertionError("The 2nd dimension must sum to 2*pi")

        if self.dim == 2:
            print('Warning, a disk mesh has not been tested thoroughly.')

        if 'cartesianOrigin' in kwargs.keys():
            self.cartesianOrigin = kwargs.pop('cartesianOrigin')
        else:
            self.cartesianOrigin = np.zeros(self.dim)

    @properties.validator('cartesianOrigin')
    def check_cartesian_origin_shape(self, change):
        change['value'] = np.array(change['value'], dtype=float).ravel()
        if len(change['value']) != self.dim:
            raise Exception(
                "Dimension mismatch. The mesh dimension is {}, and the "
                "cartesianOrigin provided has length {}".format(
                    self.dim, len(change['value'])
                )
            )

    @property
    def isSymmetric(self):
        """
        Is the mesh cylindrically symmetric?

        Returns
        -------
        bool
            True if the mesh is cylindrically symmetric, False otherwise
        """
        return self.nCy == 1

    @property
    def _ntNx(self):
        """
        Returns
        -------
        int
            Number of total x nodes (prior to deflating)
        """
        if self.isSymmetric:
            return self.nCx
        return self.nCx + 1

    @property
    def nNx(self):
        """
        Returns
        -------
        int
            Number of nodes in the x-direction
        """
        # if self.isSymmetric is True:
        #     return self.nCx
        return self._ntNx

    @property
    def _ntNy(self):
        """
        Returns
        -------
        int
            Number of total y nodes (prior to deflating)
        """
        if self.isSymmetric is True:
            return 1
        return self.nCy + 1

    @property
    def nNy(self):
        """
        Returns
        -------
        int
            Number of nodes in the y-direction
        """
        if self.isSymmetric is True:
            return 0
        return self.nCy

    @property
    def _ntNz(self):
        """
        Returns
        -------
        int
            Number of total z nodes (prior to deflating)
        """
        return self.nNz

    @property
    def _ntN(self):
        """
        Returns
        -------
        int
            Number of total nodes (prior to deflating)
        """
        if self.isSymmetric:
            return 0
        return int(self._ntNx * self._ntNy * self._ntNz)

    @property
    def nN(self):
        """
        Returns
        -------
        int
            Total number of nodes
        """
        if self.isSymmetric:
            return 0
        return (self.nNx - 1) * self.nNy * self.nNz + self.nNz

    @property
    def _vntFx(self):
        """
        vector number of total Fx (prior to deflating)
        """
        # if self.isSymmetric:
        #     return np.r_[self._ntNx, 1, self.nCz]
        return np.r_[self._ntNx, self.nCy, self.nCz]

    @property
    def _ntFx(self):
        """
        number of total Fx (prior to defplating)
        """
        return int(self._vntFx.prod())

    @property
    def _nhFx(self):
        """
        Number of hanging Fx
        """
        return int(self.nCy * self.nCz)

    @property
    def vnFx(self):
        """
        Returns
        -------
        numpy.ndarray
            Number of x-faces in each direction, (dim, )
        """
        return self.vnC

    @property
    def _vntFy(self):
        """
        vector number of total Fy (prior to deflating)
        """
        # if self.isSymmetric:
        #     return np.r_[0, 0, 0]
        return np.r_[self.nCx, self._ntNy, self.nCz]

    @property
    def _ntFy(self):
        """
        number of total Fy (prior to deflating)
        """
        return int(self._vntFy.prod())

    @property
    def _nhFy(self):
        """
        number of hanging y-faces
        """
        return int(self.nCx * self.nCz)

    @property
    def _vntFz(self):
        """
        vector number of total Fz (prior to deflating)
        """
        # if self.isSymmetric:
        #     return np.r_[self.nCx, 1, self._ntNz]
        return np.r_[self.nCx, self.nCy, self._ntNz]

    @property
    def _ntFz(self):
        """
        number of total Fz (prior to deflating)
        """
        return int(self._vntFz.prod())

    @property
    def _nhFz(self):
        """
        number of hanging Fz
        """
        return int(self.nCx * self.nCz)

    @property
    def _vntEx(self):
        """
        vector number of total Ex (prior to deflating)
        """
        return np.r_[self.nCx, self._ntNy, self._ntNz]

    @property
    def _ntEx(self):
        """
        number of total Ex (prior to deflating)
        """
        return int(self._vntEx.prod())

    @property
    def _vntEy(self):
        """
        vector number of total Ey (prior to deflating)
        """
        return np.r_[self._ntNx, self.nCy, self._ntNz]

    @property
    def _ntEy(self):
        """
        number of total Ey (prior to deflating)
        """
        return int(self._vntEy.prod())

    @property
    def vnEy(self):
        """
        Number of y-edges in each direction

        Returns
        -------
        numpy.ndarray
            vnEy or None if dim < 2, (dim, )
        """
        if self.isSymmetric:
            return np.r_[self.nNx, self.nCy, self.nNz]
        return np.r_[self.nNx - 1, self.nCy, self.nNz]

    @property
    def _vntEz(self):
        """
        vector number of total Ez (prior to deflating)
        """
        return np.r_[self._ntNx, self._ntNy, self.nCz]

    @property
    def _ntEz(self):
        """
        number of total Ez (prior to deflating)
        """
        return int(self._vntEz.prod())

    @property
    def vnEz(self):
        """
        Returns
        -------
        numpy.ndarray
            Number of z-edges in each direction or None if nCy > 1, (dim, )
        """
        return np.r_[self.nNx, self.nNy, self.nCz]

    @property
    def nEz(self):
        """
        Returns
        -------
        int
            Number of z-edges
        """
        if self.isSymmetric is True:
            return self.vnEz.prod()
        return (np.r_[self.nNx-1, self.nNy, self.nCz]).prod() + self.nCz

    @property
    def vectorCCx(self):
        """Cell-centered grid vector (1D) in the x direction."""
        return np.r_[0, self.hx[:-1].cumsum()] + self.hx*0.5

    @property
    def vectorCCy(self):
        """Cell-centered grid vector (1D) in the y direction."""
        if self.isSymmetric is True:
            return np.r_[0, self.hy[:-1]]
        return np.r_[0, self.hy[:-1].cumsum()] + self.hy*0.5

    @property
    def vectorNx(self):
        """Nodal grid vector (1D) in the x direction."""
        if self.isSymmetric is True:
            return self.hx.cumsum()
        return np.r_[0, self.hx].cumsum()

    @property
    def _vectorNyFull(self):
        """
        full nodal y vector (prior to deflating)
        """
        if self.isSymmetric:
            return np.r_[0]
        return np.r_[0, self.hy.cumsum()]

    @property
    def vectorNy(self):
        """Nodal grid vector (1D) in the y direction."""
        # if self.isSymmetric is True:
        #     # There aren't really any nodes, but all the grids need
        #     # somewhere to live, why not zero?!
        #     return np.r_[0]
        return np.r_[0, self.hy[:-1].cumsum()]

    @property
    def _edgeExFull(self):
        """
        full x-edge lengths (prior to deflating)
        """
        return np.kron(
            np.ones(self._ntNz), np.kron(np.ones(self._ntNy), self.hx)
        )

    @property
    def edgeEx(self):
        """
        x-edge lengths - these are the radial edges. Radial edges only exist
        for a 3D cyl mesh.

        Returns
        -------
        numpy.ndarray
            vector of radial edge lengths
        """
        if getattr(self, '_edgeEx', None) is None:
            self._edgeEx = self._edgeExFull[~self._ishangingEx]
        return self._edgeEx

    @property
    def _edgeEyFull(self):
        """
        full vector of y-edge lengths (prior to deflating)
        """
        if self.isSymmetric:
            return 2*pi*self.gridN[:, 0]
        return np.kron(
            np.ones(self._ntNz),
            np.kron(self.hy, self.vectorNx)
        )

    @property
    def edgeEy(self):
        """
        y-edge lengths - these are the azimuthal edges. Azimuthal edges exist
        for all cylindrical meshes. These are arc-lengths (:math:`\\theta r`)

        Returns
        -------
        numpy.ndarray
            vector of the azimuthal edges
        """
        if getattr(self, '_edgeEy', None) is None:
            if self.isSymmetric:
                self._edgeEy = self._edgeEyFull
            else:
                self._edgeEy = self._edgeEyFull[~self._ishangingEy]
        return self._edgeEy

    @property
    def _edgeEzFull(self):
        """
        full z-edge lengths (prior to deflation)
        """
        return np.kron(
            self.hz,
            np.kron(np.ones(self._ntNy), np.ones(self._ntNx))
        )

    @property
    def edgeEz(self):
        """
        z-edge lengths - these are the vertical edges. Vertical edges only
        exist for a 3D cyl mesh.

        Returns
        -------
        numpy.ndarray
            vector of the vertical edges
        """
        if getattr(self, '_edgeEz', None) is None:

            self._edgeEz = self._edgeEzFull[~self._ishangingEz]
        return self._edgeEz

    @property
    def _edgeFull(self):
        """
        full edge lengths [r-edges, theta-edgesm z-edges] (prior to
        deflation)
        """
        if self.isSymmetric:
            raise NotImplementedError
        else:
            return np.r_[self._edgeExFull, self._edgeEyFull, self._edgeEzFull]

    @property
    def edge(self):
        """
        Edge lengths

        Returns
        -------
        numpy.ndarray
            vector of edge lengths :math:`(r, \\theta, z)`
        """
        if self.isSymmetric is True:
            return self.edgeEy
            # return 2*pi*self.gridN[:, 0]
        else:
            return np.r_[self.edgeEx, self.edgeEy, self.edgeEz]

    @property
    def _areaFxFull(self):
        """
        area of x-faces prior to deflation
        """
        if self.isSymmetric:
            return np.kron(self.hz, 2*pi*self.vectorNx)
        return np.kron(self.hz, np.kron(self.hy, self.vectorNx))

    @property
    def areaFx(self):
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
        if getattr(self, '_areaFx', None) is None:
            if self.isSymmetric:
                self._areaFx = self._areaFxFull
            else:
                self._areaFx = self._areaFxFull[~self._ishangingFx]
        return self._areaFx

    @property
    def _areaFyFull(self):
        """
        Area of y-faces (Azimuthal faces), prior to deflation.
        """
        return np.kron(self.hz, np.kron(np.ones(self._ntNy), self.hx))

    @property
    def areaFy(self):
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
        if getattr(self, '_areaFy', None) is None:
            if self.isSymmetric is True:
                raise Exception(
                    'There are no y-faces on the Cyl Symmetric mesh'
                )
            self._areaFy = self._areaFyFull[~self._ishangingFy]
        return self._areaFy

    @property
    def _areaFzFull(self):
        """
        area of z-faces prior to deflation
        """
        if self.isSymmetric:
            return np.kron(
                    np.ones_like(self.vectorNz), pi*(
                        self.vectorNx**2 -
                        np.r_[0, self.vectorNx[:-1]]**2
                    )
            )
        return np.kron(
            np.ones(self._ntNz), np.kron(
                self.hy,
                0.5 * (self.vectorNx[1:]**2 - self.vectorNx[:-1]**2)
            )
        )

    @property
    def areaFz(self):
        """
        Area of z-faces.

        .. math::
            A_z = \\frac{\\theta}{2} (r_2^2 - r_1^2)z

        Returns
        -------
        numpy.ndarray
            area of the z-faces
        """
        if getattr(self, '_areaFz', None) is None:
            if self.isSymmetric:
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
    def area(self):
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
        if self.isSymmetric is True:
            return np.r_[self.areaFx, self.areaFz]
        else:
            return np.r_[self.areaFx, self.areaFy, self.areaFz]

    @property
    def vol(self):
        """
        Volume of each cell

        Returns
        -------
        numpy.ndarray
            cell volumes
        """
        if getattr(self, '_vol', None) is None:
            if self.isSymmetric:
                az = pi*(self.vectorNx**2 - np.r_[0, self.vectorNx[:-1]]**2)
                self._vol = np.kron(self.hz, az)
            else:
                self._vol = np.kron(
                    self.hz, np.kron(
                        self.hy,
                        0.5 * (self.vectorNx[1:]**2 - self.vectorNx[:-1]**2)
                    )
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
    #              np.ones(self.nCz, dtype=bool),  # 1 * 0 == 0
    #              np.kron(np.ones(self.nCy, dtype=bool), hang_x)
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
        if getattr(self, '_ishangingFxBool', None) is None:

            # the following is equivalent to
            #     hang_x = np.zeros(self._vntFx, dtype=bool)
            #     hang_x[0, :, :] = True
            #     isHangingFxBool = mkvc(hang_x)
            #
            # but krons of bools is more efficient

            hang_x = np.zeros(self._ntNx, dtype=bool)
            hang_x[0] = True
            self._ishangingFxBool = np.kron(
                np.ones(self.nCz, dtype=bool),  # 1 * 0 == 0
                np.kron(
                    np.ones(self.nCy, dtype=bool),
                    hang_x
                )
            )
        return self._ishangingFxBool

    @property
    def _hangingFx(self):
        """
        dictionary of the indices of the hanging x-faces (keys) and a list
        of indices that the eliminated faces map to (if applicable)
        """
        if getattr(self, '_hangingFxDict', None) is None:
            self._hangingFxDict = dict(zip(
                np.nonzero(self._ishangingFx)[0].tolist(), [None]*self._nhFx
            ))
        return self._hangingFxDict

    @property
    def _ishangingFy(self):
        """
        bool vector indicating if a y-face is hanging or not
        """

        if getattr(self, '_ishangingFyBool', None) is None:
            hang_y = np.zeros(self._ntNy, dtype=bool)
            hang_y[-1] = True
            self._ishangingFyBool = np.kron(
                np.ones(self.nCz, dtype=bool),
                np.kron(
                    hang_y,
                    np.ones(self.nCx, dtype=bool)
                )
            )
        return self._ishangingFyBool

    @property
    def _hangingFy(self):
        """
        dictionary of the indices of the hanging y-faces (keys) and a list
        of indices that the eliminated faces map to (if applicable)
        """
        if getattr(self, '_hangingFyDict', None) is None:
            deflate_y = np.zeros(self._ntNy, dtype=bool)
            deflate_y[0] = True
            deflateFy = np.nonzero(np.kron(
                np.ones(self.nCz, dtype=bool),
                np.kron(
                    deflate_y,
                    np.ones(self.nCx, dtype=bool)
                )
            ))[0].tolist()
            self._hangingFyDict = dict(zip(
                np.nonzero(self._ishangingFy)[0].tolist(),
                deflateFy)
            )
        return self._hangingFyDict

    @property
    def _ishangingFz(self):
        """
        bool vector indicating if a z-face is hanging or not
        """
        if getattr(self, '_ishangingFzBool', None) is None:
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
        if getattr(self, '_ishangingExBool', None) is None:
            hang_y = np.zeros(self._ntNy, dtype=bool)
            hang_y[-1] = True
            self._ishangingExBool = np.kron(
                np.ones(self._ntNz, dtype=bool),
                np.kron(
                    hang_y,
                    np.ones(self.nCx, dtype=bool)
                )
            )
        return self._ishangingExBool

    @property
    def _hangingEx(self):
        """
        dictionary of the indices of the hanging x-edges (keys) and a list
        of indices that the eliminated faces map to (if applicable)
        """
        if getattr(self, '_hangingExDict', None) is None:
            deflate_y = np.zeros(self._ntNy, dtype=bool)
            deflate_y[0] = True
            deflateEx = np.nonzero(np.kron(
                np.ones(self._ntNz, dtype=bool),
                np.kron(
                    deflate_y,
                    np.ones(self.nCx, dtype=bool)
                )
            ))[0].tolist()
            self._hangingExDict = dict(zip(
                np.nonzero(self._ishangingEx)[0].tolist(), deflateEx
            ))
        return self._hangingExDict

    @property
    def _ishangingEy(self):
        """
        bool vector indicating if a y-edge is hanging or not
        """
        if getattr(self, '_ishangingEyBool', None) is None:
            hang_x = np.zeros(self._ntNx, dtype=bool)
            hang_x[0] = True
            self._ishangingEyBool = np.kron(
                np.ones(self._ntNz, dtype=bool),
                np.kron(
                    np.ones(self.nCy, dtype=bool),
                    hang_x
                )
            )
        return self._ishangingEyBool

    @property
    def _hangingEy(self):
        """
        dictionary of the indices of the hanging y-edges (keys) and a list
        of indices that the eliminated faces map to (if applicable)
        """
        if getattr(self, '_hangingEyDict', None) is None:
            self._hangingEyDict = dict(zip(
                np.nonzero(self._ishangingEy)[0].tolist(),
                [None]*len(self._ishangingEyBool))
            )
        return self._hangingEyDict

    @property
    def _axis_of_symmetry_Ez(self):
        """
        bool vector indicating if a z-edge is along the axis of symmetry or not
        """
        if getattr(self, '_axis_of_symmetry_EzBool', None) is None:
            axis_x = np.zeros(self._ntNx, dtype=bool)
            axis_x[0] = True

            axis_y = np.zeros(self._ntNy, dtype=bool)
            axis_y[0] = True
            self._axis_of_symmetry_EzBool = np.kron(
                np.ones(self.nCz, dtype=bool),
                np.kron(
                    axis_y,
                    axis_x
                )
            )
        return self._axis_of_symmetry_EzBool

    @property
    def _ishangingEz(self):
        """
        bool vector indicating if a z-edge is hanging or not
        """
        if getattr(self, '_ishangingEzBool', None) is None:
            if self.isSymmetric:
                self._ishangingEzBool = np.ones(self._ntEz, dtype=bool)
            else:
                hang_x = np.zeros(self._ntNx, dtype=bool)
                hang_x[0] = True

                hang_y = np.zeros(self._ntNy, dtype=bool)
                hang_y[-1] = True

                hangingEz = np.kron(
                    np.ones(self.nCz, dtype=bool),
                    (
                        # True * False = False
                        np.kron(
                            np.ones(self._ntNy, dtype=bool),
                            hang_x
                        ) |
                        np.kron(
                            hang_y,
                            np.ones(self._ntNx, dtype=bool)
                        )
                    )
                )

                self._ishangingEzBool = hangingEz & ~self._axis_of_symmetry_Ez

        return self._ishangingEzBool

    @property
    def _hangingEz(self):
        """
        dictionary of the indices of the hanging z-edges (keys) and a list
        of indices that the eliminated faces map to (if applicable)
        """
        if getattr(self, '_hangingEzDict', None) is None:
            # deflate
            deflateEz = np.hstack([
                np.hstack([
                    np.zeros(self._ntNy-1, dtype=int),
                    np.arange(1, self._ntNx, dtype=int)
                ]) +
                i*int(self._ntNx*self._ntNy)
                for i in range(self.nCz)
            ])
            deflate = zip(
                np.nonzero(self._ishangingEz)[0].tolist(), deflateEz
            )

            self._hangingEzDict = dict(deflate)
        return self._hangingEzDict

    @property
    def _axis_of_symmetry_N(self):
        """
        bool vector indicating if a node is along the axis of symmetry or not
        """
        if getattr(self, '_axis_of_symmetry_NBool', None) is None:
            axis_x = np.zeros(self._ntNx, dtype=bool)
            axis_x[0] = True

            axis_y = np.zeros(self._ntNy, dtype=bool)
            axis_y[0] = True
            self._axis_of_symmetry_NBool = np.kron(
                np.ones(self._ntNz, dtype=bool),
                np.kron(
                    axis_y,
                    axis_x
                )
            )
        return self._axis_of_symmetry_NBool

    @property
    def _ishangingN(self):
        """
        bool vector indicating if a node is hanging or not
        """
        if getattr(self, '_ishangingNBool', None) is None:
            hang_x = np.zeros(self._ntNx, dtype=bool)
            hang_x[0] = True

            hang_y = np.zeros(self._ntNy, dtype=bool)
            hang_y[-1] = True

            hangingN = np.kron(
                np.ones(self._ntNz, dtype=bool),
                (
                    np.kron(
                        np.ones(self._ntNy, dtype=bool),
                        hang_x
                    ) |
                    np.kron(
                        hang_y,
                        np.ones(self._ntNx, dtype=bool)
                    )
                )
            )

            self._ishangingNBool = hangingN & ~self._axis_of_symmetry_N

        return self._ishangingNBool

    @property
    def _hangingN(self):
        """
        dictionary of the indices of the hanging nodes (keys) and a list
        of indices that the eliminated faces map to (if applicable)
        """
        if getattr(self, '_hangingNDict', None) is None:
            # go by layer
            deflateN = np.hstack([
                np.hstack([
                    np.zeros(self._ntNy-1, dtype=int),
                    np.arange(1, self._ntNx, dtype=int)
                ]) +
                i*int(self._ntNx*self._ntNy)
                for i in range(self._ntNz)
            ]).tolist()
            self._hangingNDict = dict(zip(
                np.nonzero(self._ishangingN)[0].tolist(), deflateN
            ))
        return self._hangingNDict

    ####################################################
    # Grids
    ####################################################

    @property
    def _gridNFull(self):
        """
        Full Nodal grid (including hanging nodes)
        """
        return ndgrid([
            self.vectorNx, self._vectorNyFull, self.vectorNz
        ])

    @property
    def gridN(self):
        """
        Nodal grid in cylindrical coordinates :math:`(r, \\theta, z)`.
        Nodes do not exist in a cylindrically symmetric mesh.

        Returns
        -------
        numpy.ndarray
            grid locations of nodes
        """
        if self.isSymmetric:
            self._gridN = self._gridNFull
        if getattr(self, '_gridN', None) is None:
            self._gridN = self._gridNFull[~self._ishangingN, :]
        return self._gridN

    @property
    def _gridFxFull(self):
        """
        Full Fx grid (including hanging faces)
        """
        return ndgrid([
            self.vectorNx, self.vectorCCy, self.vectorCCz
        ])

    @property
    def gridFx(self):
        """
        Grid of x-faces (radial-faces) in cylindrical coordinates
        :math:`(r, \\theta, z)`.

        Returns
        -------
        numpy.ndarray
            grid locations of radial faces
        """
        if getattr(self, '_gridFx', None) is None:
            if self.isSymmetric is True:
                return super(CylMesh, self).gridFx
            else:
                self._gridFx = self._gridFxFull[~self._ishangingFx, :]
        return self._gridFx

    @property
    def _gridEyFull(self):
        """
        Full grid of y-edges (including eliminated edges)
        """
        return super(CylMesh, self).gridEy

    @property
    def gridEy(self):
        """
        Grid of y-edges (azimuthal-faces) in cylindrical coordinates
        :math:`(r, \\theta, z)`.

        Returns
        -------
        numpy.ndarray
            grid locations of azimuthal faces
        """
        if getattr(self, '_gridEy', None) is None:
            if self.isSymmetric is True:
                return self._gridEyFull
            else:
                self._gridEy = self._gridEyFull[~self._ishangingEy, :]
        return self._gridEy

    @property
    def _gridEzFull(self):
        """
        Full z-edge grid (including hanging edges)
        """
        return ndgrid([
            self.vectorNx, self._vectorNyFull, self.vectorCCz
        ])

    @property
    def gridEz(self):
        """
        Grid of z-faces (vertical-faces) in cylindrical coordinates
        :math:`(r, \\theta, z)`.

        Returns
        -------
        numpy.ndarray
            grid locations of radial faces
        """
        if getattr(self, '_gridEz', None) is None:
            if self.isSymmetric is True:
                self._gridEz = None
            else:
                self._gridEz = self._gridEzFull[~self._ishangingEz, :]
        return self._gridEz

    ####################################################
    # Operators
    ####################################################

    @property
    def faceDiv(self):
        """
        Construct divergence operator (faces to cell-centres).
        """
        if getattr(self, '_faceDiv', None) is None:
            # Compute faceDivergence operator on faces
            D1 = self.faceDivx
            D3 = self.faceDivz
            if self.isSymmetric is True:
                D = sp.hstack((D1, D3), format="csr")
            elif self.nCy > 1:
                D2 = self.faceDivy
                D = sp.hstack((D1, D2, D3), format="csr")
            self._faceDiv = D
        return self._faceDiv

    @property
    def faceDivx(self):
        """
        Construct divergence operator in the x component
        (faces to cell-centres).
        """
        if getattr(self, '_faceDivx', None) is None:
            if self.isSymmetric:
                D1 = kron3(
                    speye(self.nCz), speye(self.nCy),
                    ddx(self.nCx)[:, 1:]
                )
            else:
                D1 = super(CylMesh, self)._faceDivStencilx

            S = self._areaFxFull
            V = self.vol
            self._faceDivx = sdiag(1/V)*D1*sdiag(S)

            if not self.isSymmetric:
                self._faceDivx = (
                    self._faceDivx *
                    self._deflationMatrix(
                        'Fx', asOnes=True
                    ).T
                )

        return self._faceDivx

    @property
    def faceDivy(self):
        """
        Construct divergence operator in the y component
        (faces to cell-centres).
        """
        if getattr(self, '_faceDivy', None) is None:
            D2 = super(CylMesh, self)._faceDivStencily
            S = self._areaFyFull  # self.r(self.area, 'F', 'Fy', 'V')
            V = self.vol
            self._faceDivy = (
                sdiag(1/V)*D2*sdiag(S) *
                self._deflationMatrix('Fy', asOnes=True).T
            )
        return self._faceDivy

    @property
    def faceDivz(self):
        """
        Construct divergence operator in the z component
        (faces to cell-centres).
        """
        if getattr(self, '_faceDivz', None) is None:
            D3 = super(CylMesh, self)._faceDivStencilz
            S = self._areaFzFull
            V = self.vol
            self._faceDivz = sdiag(1/V)*D3*sdiag(S)
        return self._faceDivz

    # @property
    # def _cellGradxStencil(self):
    #     n = self.vnC

    #     if self.isSymmetric:
    #         G1 = sp.kron(speye(n[2]), ddxCellGrad(n[0], BC))
    #     else:
    #         G1 = self._deflationMatrix('Fx').T * kron3(
    #             speye(n[2]), speye(n[1]), ddxCellGrad(n[0], BC)
    #         )
    #     return G1

    @property
    def cellGradx(self):
        raise NotImplementedError
        # if getattr(self, '_cellGradx', None) is None:
        #     G1 = super(CylMesh, self)._cellGradxStencil
        #     V = self._deflationMatrix('F', withHanging='True', asOnes='True')*self.aveCC2F*self.vol
        #     A = self.area
        #     L = (A/V)[:self._ntFx]
        #     # L = self.r(L, 'F', 'Fx', 'V')
        #     # L = A[:self.nFx] / V
        #     self._cellGradx = self._deflationMatrix('Fx')*sdiag(L)*G1
        # return self._cellGradx

    @property
    def _cellGradyStencil(self):
        raise NotImplementedError

    @property
    def _cellGradzStencil(self):
        raise NotImplementedError

    @property
    def _cellGradStencil(self):
        raise NotImplementedError

    @property
    def cellGrad(self):
        """The cell centered Gradient, takes you to cell faces."""
        raise NotImplementedError('Cell Grad is not yet implemented.')

    # @property
    # def _nodalGradStencilx(self):
    #     if self.isSymmetric is True:
    #         return None
    #     return kron3(speye(self.nNz), speye(self.nNy), ddx(self.nCx))

    # @property
    # def _nodalGradStencily(self):
    #     if self.isSymmetric is True:
    #         None
    #         # return kron3(speye(self.nNz), ddx(self.nCy), speye(self.nNx)) * self._deflationMatrix('Ey')
    #     return kron3(speye(self.nNz), ddx(self.nCy), speye(self.nNx))

    # @property
    # def _nodalGradStencilz(self):
    #     if self.isSymmetric is True:
    #         return None
    #     return kron3(ddx(self.nCz), speye(self.nNy), speye(self.nNx))

    # @property
    # def _nodalGradStencil(self):
    #     if self.isSymmetric is True:
    #         return None
    #     else:
    #         G = self._deflationMatrix('E').T * sp.vstack((
    #             self._nodalGradStencilx,
    #             self._nodalGradStencily,
    #             self._nodalGradStencilz
    #         ), format="csr") * self._deflationMatrix('N')
    #     return G

    @property
    def nodalGrad(self):
        """Construct gradient operator (nodes to edges)."""
        if self.isSymmetric is True:
            return None
        raise NotImplementedError('nodalGrad not yet implemented')

    @property
    def nodalLaplacian(self):
        """Construct laplacian operator (nodes to edges)."""
        raise NotImplementedError('nodalLaplacian not yet implemented')

    @property
    def edgeCurl(self):
        """
        The edgeCurl (edges to faces)

        Returns
        -------
        scipy.sparse.csr_matrix
            edge curl operator
        """
        if getattr(self, '_edgeCurl', None) is None:
            A = self.area
            E = self.edge

            if self.isSymmetric is True:
                # 1D Difference matricies
                dr = sp.spdiags(
                    (np.ones((self.nCx+1, 1))*[-1, 1]).T, [-1, 0],
                    self.nCx, self.nCx, format="csr"
                )
                dz = sp.spdiags(
                    (np.ones((self.nCz+1, 1))*[-1, 1]).T, [0, 1],
                    self.nCz, self.nCz+1, format="csr"
                )
                # 2D Difference matricies
                Dr = sp.kron(sp.identity(self.nNz), dr)
                Dz = -sp.kron(dz, sp.identity(self.nCx))

                # Edge curl operator
                self._edgeCurl = (
                    sdiag(1/A)*sp.vstack((Dz, Dr)) * sdiag(E)
                )
            else:
                self._edgeCurl = (
                    sdiag(1/self.area) *
                    self._deflationMatrix('F', asOnes=False) *
                    self._edgeCurlStencil *
                    sdiag(self._edgeFull) *
                    self._deflationMatrix('E', asOnes=True).T
                )

        return self._edgeCurl

    @property
    def aveEx2CC(self):
        """
        averaging operator of x-edges (radial) to cell centers

        Returns
        -------
        scipy.sparse.csr_matrix
            matrix that averages from x-edges to cell centers
        """
        if self.isSymmetric:
            raise Exception('There are no x-edges on a cyl symmetric mesh')
        return kron3(
            av(self.vnC[2]),
            av(self.vnC[1]),
            speye(self.vnC[0])
        ) * self._deflationMatrix('Ex', asOnes=True).T

    @property
    def aveEy2CC(self):
        """
        averaging operator of y-edges (azimuthal) to cell centers

        Returns
        -------
        scipy.sparse.csr_matrix
            matrix that averages from y-edges to cell centers
        """
        if self.isSymmetric:
            avR = av(self.vnC[0])[:, 1:]
            return sp.kron(av(self.vnC[2]), avR, format="csr")
        else:
            return kron3(
                av(self.vnC[2]),
                speye(self.vnC[1]),
                av(self.vnC[0])
            )*self._deflationMatrix('Ey', asOnes=True).T

    @property
    def aveEz2CC(self):
        """
        averaging operator of z-edges to cell centers

        Returns
        -------
        scipy.sparse.csr_matrix
            matrix that averages from z-edges to cell centers
        """
        if self.isSymmetric:
            raise Exception('There are no z-edges on a cyl symmetric mesh')
        return kron3(
            speye(self.vnC[2]),
            av(self.vnC[1]),
            av(self.vnC[0])
        ) * self._deflationMatrix('Ez', asOnes=True).T

    @property
    def aveE2CC(self):
        """
        averaging operator of edges to cell centers

        Returns
        -------
        scipy.sparse.csr_matrix
            matrix that averages from edges to cell centers
        """
        if getattr(self, '_aveE2CC', None) is None:
            # The number of cell centers in each direction
            # n = self.vnC
            if self.isSymmetric is True:
                self._aveE2CC = self.aveEy2CC
            else:
                self._aveE2CC = 1./self.dim * sp.hstack(
                    (self.aveEx2CC, self.aveEy2CC, self.aveEz2CC),
                    format="csr"
                )
        return self._aveE2CC

    @property
    def aveE2CCV(self):
        """
        averaging operator of edges to a cell centered vector

        Returns
        -------
        scipy.sparse.csr_matrix
            matrix that averages from edges to cell centered vectors
        """
        if self.isSymmetric is True:
            return self.aveE2CC
        else:
            if getattr(self, '_aveE2CCV', None) is None:
                self._aveE2CCV = sp.block_diag(
                    (self.aveEx2CC, self.aveEy2CC, self.aveEz2CC),
                    format="csr"
                )
        return self._aveE2CCV

    @property
    def aveFx2CC(self):
        """
        averaging operator of x-faces (radial) to cell centers

        Returns
        -------
        scipy.sparse.csr_matrix
            matrix that averages from x-faces to cell centers
        """
        avR = av(self.vnC[0])[:, 1:]  # TODO: this should be handled by a deflation matrix
        return kron3(
            speye(self.vnC[2]), speye(self.vnC[1]), avR
        )

    @property
    def aveFy2CC(self):
        """
        averaging operator of y-faces (azimuthal) to cell centers

        Returns
        -------
        scipy.sparse.csr_matrix
            matrix that averages from y-faces to cell centers
        """
        return kron3(
            speye(self.vnC[2]), av(self.vnC[1]),
            speye(self.vnC[0])
        ) * self._deflationMatrix('Fy', asOnes=True).T

    @property
    def aveFz2CC(self):
        """
        averaging operator of z-faces (vertical) to cell centers

        Returns
        -------
        scipy.sparse.csr_matrix
            matrix that averages from z-faces to cell centers
        """

        return kron3(
            av(self.vnC[2]), speye(self.vnC[1]),
            speye(self.vnC[0])
        )

    @property
    def aveF2CC(self):
        """
        averaging operator of faces to cell centers

        Returns
        -------
        scipy.sparse.csr_matrix
            matrix that averages from faces to cell centers
        """
        if getattr(self, '_aveF2CC', None) is None:
            n = self.vnC
            if self.isSymmetric is True:
                self._aveF2CC = 0.5*(
                    sp.hstack((self.aveFx2CC, self.aveFz2CC), format="csr")
                )
            else:
                self._aveF2CC = 1./self.dim*(
                    sp.hstack(
                        (self.aveFx2CC, self.aveFy2CC, self.aveFz2CC),
                        format="csr"
                    )
                )
        return self._aveF2CC

    @property
    def aveF2CCV(self):
        """
        averaging operator of x-faces (radial) to cell centered vectors

        Returns
        -------
        scipy.sparse.csr_matrix
            matrix that averages from faces to cell centered vectors
        """
        if getattr(self, '_aveF2CCV', None) is None:
            # n = self.vnC
            if self.isSymmetric is True:
                self._aveF2CCV = sp.block_diag(
                    (self.aveFx2CC, self.aveFz2CC), format="csr"
                )
            else:
                self._aveF2CCV = sp.block_diag(
                    (self.aveFx2CC, self.aveFy2CC, self.aveFz2CC),
                    format="csr"
                )
        return self._aveF2CCV

    ####################################################
    # Deflation Matrices
    ####################################################

    def _deflationMatrix(self, location, asOnes=False):
        """
        construct the deflation matrix to remove hanging edges / faces / nodes
        from the operators
        """
        if location not in [
            'N', 'F', 'Fx', 'Fy', 'Fz', 'E', 'Ex', 'Ey', 'Ez', 'CC'
        ]:
            raise AssertionError(
                'Location must be a grid location, not {}'.format(location)
            )
        if location == 'CC':
            return speye(self.nC)

        elif location in ['E', 'F']:
            if self.isSymmetric:
                if location == 'E':
                    return self._deflationMatrix('Ey', asOnes=asOnes)
                elif location == 'F':
                    return sp.block_diag([
                        self._deflationMatrix(location+coord, asOnes=asOnes)
                        for coord in ['x', 'z']
                    ])
            return sp.block_diag([
                self._deflationMatrix(location+coord, asOnes=asOnes)
                for coord in ['x', 'y', 'z']
            ])

        R = speye(getattr(self, '_nt{}'.format(location)))
        hanging_dict = getattr(self, '_hanging{}'.format(location))
        nothanging = ~getattr(self, '_ishanging{}'.format(location))

        # remove eliminated edges / faces (eg. Fx just doesn't exist)
        hang = {k: v for k, v in hanging_dict.items() if v is not None}

        values = list(hang.values())
        entries = np.ones(len(values))

        if asOnes is False and len(hang) > 0:
            repeats = set(values)
            repeat_locs = [
                (np.r_[values] == repeat).nonzero()[0]
                for repeat in repeats
            ]
            for loc in repeat_locs:
                entries[loc] = 1./len(loc)

        Hang = sp.csr_matrix(
            (entries, (values, list(hang.keys()))),
            shape=(
                getattr(self, '_nt{}'.format(location)),
                getattr(self, '_nt{}'.format(location))
            )
        )
        R = R + Hang

        R = R[nothanging, :]

        if not asOnes:
            R = sdiag(1./R.sum(1)) * R

        return R

    ####################################################
    # Interpolation
    ####################################################

    def getInterpolationMat(self, loc, locType='CC', zerosOutside=False):
        """ Produces interpolation matrix

        Parameters
        ----------
        loc : numpy.ndarray
            Location of points to interpolate to

        locType : str
            What to interpolate locType can be::

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
        if self.isSymmetric and locType in ['Ex', 'Ez', 'Fy']:
            raise Exception(
                "Symmetric CylMesh does not support {0!s} interpolation, "
                "as this variable does not exist.".format(locType)
            )

        if locType in ['CCVx', 'CCVy', 'CCVz']:
            Q = interpmat(loc, *self.getTensor('CC'))
            Z = spzeros(loc.shape[0], self.nC)
            if locType == 'CCVx':
                Q = sp.hstack([Q, Z])
            elif locType == 'CCVy':
                Q = sp.hstack([Q])
            elif locType == 'CCVz':
                Q = sp.hstack([Z, Q])

            if zerosOutside:
                Q[indZeros, :] = 0

            return Q.tocsr()

        return self._getInterpolationMat(loc, locType, zerosOutside)

    def cartesianGrid(self, locType='CC', theta_shift=None):
        """
        Takes a grid location ('CC', 'N', 'Ex', 'Ey', 'Ez', 'Fx', 'Fy', 'Fz')
        and returns that grid in cartesian coordinates

        Parameters
        ----------
        locType : str
            grid location

        Returns
        -------
        numpy.ndarray
            cartesian coordinates for the cylindrical grid
        """
        grid = getattr(self, 'grid{}'.format(locType)).copy()
        if theta_shift is not None:
            grid[:, 1] = grid[:, 1] - theta_shift
        return cyl2cart(grid)  # TODO: account for cartesian origin

    def getInterpolationMatCartMesh(self, Mrect, locType='CC', locTypeTo=None):
        """
        Takes a cartesian mesh and returns a projection to translate onto
        the cartesian grid.

        Parameters
        ----------
        Mrect : discretize.base.BaseMesh
            the mesh to interpolate on to

        locType : str
            grid location ('CC', 'N', 'Ex', 'Ey', 'Ez', 'Fx', 'Fy', 'Fz')

        locTypeTo : str
            grid location to interpolate to. If None, the same grid type as `locType` will be assumed

        Returns
        -------
        scipy.sparse.csr_matrix
            M, the interpolation matrix
        """

        if not self.isSymmetric:
            raise AssertionError(
                "Currently we have not taken into account other projections "
                "for more complicated CylMeshes"
            )

        if locTypeTo is None:
            locTypeTo = locType

        if locType == 'F':
            # do this three times for each component
            X = self.getInterpolationMatCartMesh(
                Mrect, locType='Fx', locTypeTo=locTypeTo+'x'
            )
            Y = self.getInterpolationMatCartMesh(
                Mrect, locType='Fy', locTypeTo=locTypeTo+'y'
            )
            Z = self.getInterpolationMatCartMesh(
                Mrect, locType='Fz', locTypeTo=locTypeTo+'z'
            )
            return sp.vstack((X, Y, Z))
        if locType == 'E':
            X = self.getInterpolationMatCartMesh(
                Mrect, locType='Ex', locTypeTo=locTypeTo+'x'
            )
            Y = self.getInterpolationMatCartMesh(
                Mrect, locType='Ey', locTypeTo=locTypeTo+'y'
            )
            Z = spzeros(getattr(Mrect, 'n' + locTypeTo + 'z'), self.nE)
            return sp.vstack((X, Y, Z))

        grid = getattr(Mrect, 'grid' + locTypeTo)
        # This is unit circle stuff, 0 to 2*pi, starting at x-axis, rotating
        # counter clockwise in an x-y slice
        theta = - np.arctan2(
            grid[:, 0] - self.cartesianOrigin[0], grid[:, 1] -
            self.cartesianOrigin[1]
        ) + np.pi/2
        theta[theta < 0] += np.pi*2.0
        r = ((grid[:, 0] - self.cartesianOrigin[0])**2 + (grid[:, 1] -
             self.cartesianOrigin[1])**2)**0.5

        if locType in ['CC', 'N', 'Fz', 'Ez']:
            G, proj = np.c_[r, theta, grid[:, 2]], np.ones(r.size)
        else:
            dotMe = {
                'Fx': Mrect.normals[:Mrect.nFx, :],
                'Fy': Mrect.normals[Mrect.nFx:(Mrect.nFx + Mrect.nFy), :],
                'Fz': Mrect.normals[-Mrect.nFz:, :],
                'Ex': Mrect.tangents[:Mrect.nEx, :],
                'Ey': Mrect.tangents[Mrect.nEx:(Mrect.nEx+Mrect.nEy), :],
                'Ez': Mrect.tangents[-Mrect.nEz:, :],
            }[locTypeTo]
            if 'F' in locType:
                normals = np.c_[
                    np.cos(theta), np.sin(theta), np.zeros(theta.size)
                ]
                proj = (normals * dotMe).sum(axis=1)
            if 'E' in locType:
                tangents = np.c_[
                    -np.sin(theta), np.cos(theta), np.zeros(theta.size)
                ]
                proj = (tangents * dotMe).sum(axis=1)
            G = np.c_[r, theta, grid[:, 2]]

        interpType = locType
        if interpType == 'Fy':
            interpType = 'Fx'
        elif interpType == 'Ex':
            interpType = 'Ey'

        Pc2r = self.getInterpolationMat(G, interpType)
        Proj = sdiag(proj)
        return Proj * Pc2r
