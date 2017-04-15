from __future__ import print_function
import numpy as np
import scipy.sparse as sp
from scipy.constants import pi

from . import utils
from .TensorMesh import BaseTensorMesh, BaseRectangularMesh
from .InnerProducts import InnerProducts
from .View import CylView
from .DiffOperators import DiffOperators, ddxCellGrad


class CylMesh(
    BaseTensorMesh, BaseRectangularMesh, InnerProducts, CylView, DiffOperators
):
    """
        CylMesh is a mesh class for cylindrical problems

        .. note::

            for a cylindrically symmetric mesh use [hx, 1, hz]

        ::

            cs, nc, npad = 20., 30, 8
            hx = utils.meshTensor([(cs,npad+10,-0.7), (cs,nc), (cs,npad,1.3)])
            hz = utils.meshTensor([(cs,npad   ,-1.3), (cs,nc), (cs,npad,1.3)])
            mesh = Mesh.CylMesh([hx,1,hz], [0.,0,-hz.sum()/2.])
    """

    _meshType = 'CYL'

    _unitDimensions = [1, 2*np.pi, 1]

    def __init__(self, h, x0=None, cartesianOrigin=None):
        BaseTensorMesh.__init__(self, h, x0)
        assert np.abs(self.hy.sum() - 2*np.pi) < 1e-10, (
            "The 2nd dimension must sum to 2*pi"
        )
        if self.dim == 2:
            print('Warning, a disk mesh has not been tested thoroughly.')
        cartesianOrigin = (np.zeros(self.dim) if cartesianOrigin is None
                           else cartesianOrigin)
        assert len(cartesianOrigin) == self.dim, (
            "cartesianOrigin must be the same length as the dimension of the "
            "mesh."
        )
        self.cartesianOrigin = np.array(cartesianOrigin, dtype=float)

    @property
    def isSymmetric(self):
        return self.nCy == 1

    @property
    def _ntNx(self):
        return self.nCx + 1

    @property
    def nNx(self):
        """
        Number of nodes in the x-direction

        :rtype: int
        :return: nNx
        """
        if self.isSymmetric is True:
            return self.nCx
        return self._ntNx

    @property
    def _ntNy(self):
        if self.isSymmetric is True:
            return 0
        return self.nCy + 1

    @property
    def nNy(self):
        """
        Number of nodes in the y-direction

        :rtype: int
        :return: nNy
        """
        if self.isSymmetric is True:
            return 0
        return self.nCy

    @property
    def _ntNz(self):
        return self.nNz

    @property
    def _ntN(self):
        if self.isSymmetric:
            return 0
        return int(self._ntNx * self._ntNy * self._ntNz)

    @property
    def nN(self):
        """
        Total number of nodes

        :rtype: int
        :return: nN
        """
        if self.isSymmetric:
            return 0  # there are no nodes on a cylindrically symmetric mesh
        return (self.nNx - 1) * self.nNy * self.nNz + self.nNz

    @property
    def _vntFx(self):
        if self.isSymmetric:
            return np.r_[self._ntNx, 1, self.nCz]
        return np.r_[self._ntNx, self.nCy, self.nCz]

    @property
    def _ntFx(self):
        return int(self._vntFx.prod())

    @property
    def _nhFx(self):
        return int(self.nCy * self.nCz)

    @property
    def vnFx(self):
        """
        Number of x-faces in each direction

        :rtype: numpy.array
        :return: vnFx, (dim, )
        """
        return self.vnC

    @property
    def _vntFy(self):
        if self.isSymmetric:
            return np.r_[0, 0, 0]
        return np.r_[self.nCx, self._ntNy, self.nCz]

    @property
    def _ntFy(self):
        return int(self._vntFy.prod())

    @property
    def _nhFy(self):
        return int(self.nCx * self.nCz)

    @property
    def _vntFz(self):
        if self.isSymmetric:
            return np.r_[self.nCx, 1, self._ntNz]
        return np.r_[self.nCx, self.nCy, self._ntNz]

    @property
    def _ntFz(self):
        return int(self._vntFz.prod())

    @property
    def _nhFz(self):
        return int(self.nCx * self.nCz)

    @property
    def _vntEx(self):
        return np.r_[self.nCx, self._ntNy, self._ntNz]

    @property
    def _ntEx(self):
        return int(self._vntEx.prod())

    @property
    def _vntEy(self):
        return np.r_[self._ntNx, self.nCy, self._ntNz]

    @property
    def _ntEy(self):
        return int(self._vntEy.prod())

    @property
    def vnEy(self):
        """
        Number of y-edges in each direction

        :rtype: numpy.array
        :return: vnEy or None if dim < 2, (dim, )
        """
        nNx = self.nNx if self.isSymmetric else self.nNx - 1
        return np.r_[nNx, self.nCy, self.nNz]

    @property
    def _vntEz(self):
        return np.r_[self._ntNx, self._ntNy, self.nCz]

    @property
    def _ntEz(self):
        return int(self._vntEz.prod())

    @property
    def vnEz(self):
        """
        Number of z-edges in each direction

        :rtype: numpy.array
        :return: vnEz or None if nCy > 1, (dim, )
        """
        return np.r_[self.nNx, self.nNy, self.nCz]

    @property
    def nEz(self):
        """
        Number of z-edges

        :rtype: int
        :return: nEz
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
        return np.r_[0, self.hy.cumsum()]

    @property
    def vectorNy(self):
        """Nodal grid vector (1D) in the y direction."""
        if self.isSymmetric is True:
            # There aren't really any nodes, but all the grids need
            # somewhere to live, why not zero?!
            return np.r_[0]
        return np.r_[0, self.hy[:-1].cumsum()]

    @property
    def _edgeExFull(self):
        return np.kron(
            np.ones(self._ntNz), np.kron(np.ones(self._ntNy), self.hx)
        )

    @property
    def edgeEx(self):
        if getattr(self, '_edgeEx', None) is None:
            self._edgeEx = (
                self._deflationMatrix('Ex', withHanging=False) *
                self._edgeExFull
            )
        return self._edgeEx

    @property
    def _edgeEyFull(self):
        return np.kron(
            np.ones(self._ntNz),
            np.kron(self.hy, self.vectorNx)
        )

    @property
    def edgeEy(self):
        if getattr(self, '_edgeEy', None) is None:
            self._edgeEy = (
                self._deflationMatrix('Ey', withHanging=False) *
                self._edgeEyFull
            )
        return self._edgeEy

    @property
    def _edgeEzFull(self):
        return np.kron(
            self.hz,
            np.kron(np.ones(self._ntNy), np.ones(self._ntNx))
        )

    @property
    def edgeEz(self):
        if getattr(self, '_edgeEz', None) is None:

            self._edgeEz = (
                self._deflationMatrix('Ez', withHanging=False) *
                self._edgeEzFull
            )
        return self._edgeEz

    @property
    def _edgeFull(self):
        if self.isSymmetric:
            raise NotImplementedError
        else:
            return np.r_[self._edgeExFull, self._edgeEyFull, self._edgeEzFull]

    @property
    def edge(self):
        """Edge lengths"""
        if self.isSymmetric is True:
            return 2*pi*self.gridN[:, 0]
        else:
            return np.r_[self.edgeEx, self.edgeEy, self.edgeEz]

    @property
    def _areaFxFull(self):
        return np.kron(self.hz, np.kron(self.hy, self.vectorNx))

    @property
    def areaFx(self):
        if getattr(self, '_areaFx', None) is None:
            if self.isSymmetric:
                self._areaFx = np.kron(self.hz, 2*pi*self.vectorNx)
            else:
                self._areaFx = (
                    self._deflationMatrix('Fx', withHanging=False) *
                    self._areaFxFull
                )
        return self._areaFx

    @property
    def _areaFyFull(self):
        return np.kron(self.hz, np.kron(np.ones(self._ntNy), self.hx))

    @property
    def areaFy(self):
        if getattr(self, '_areaFy', None) is None:
            self._areaFy = (
                self._deflationMatrix('Fy', withHanging=False) *
                self._areaFyFull
            )
        return self._areaFy

    @property
    def _areaFzFull(self):
        if self.isSymmetric:
            return np.kron(
                    np.ones_like(self.vectorNz), pi*(
                        self.vectorNx**2 -
                        np.r_[0, self.vectorNx[:-1]]**2
                    )
                )
        else:
            return np.kron(
                np.ones(self._ntNz), np.kron(
                    self.hy,
                    0.5 * (self.vectorNx[1:]**2 - self.vectorNx[:-1]**2)
                )
            )

    @property
    def areaFz(self):
        if getattr(self, '_areaFz', None) is None:
            self._areaFz = (
                self._deflationMatrix('Fz', withHanging=False) *
                self._areaFzFull
            )
        return self._areaFz

    @property
    def _areaFull(self):
        if self.isSymmetric is True:
            return np.r_[self._areaFxFull, self._areaFzFull]
        else:
            return np.r_[self._areaFxFull, self._areaFyFull, self._areaFzFull]

    @property
    def area(self):
        """Face areas"""
        # if getattr(self, '_area', None) is None:
        if self.isSymmetric is True:
            return np.r_[self.areaFx, self.areaFz]
        else:
            return np.r_[self.areaFx, self.areaFy, self.areaFz]
        # return self._area

    @property
    def vol(self):
        """Volume of each cell"""
        if getattr(self, '_vol', None) is None:
            if self.isSymmetric is True:
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

    ####################################################
    # Active and Hanging Edges and Faces
    ####################################################

    @property
    def _ishangingFx(self):
        if getattr(self, '__ishangingFx', None) is None:
            hang_x = np.zeros(self._ntNx, dtype=bool)
            hang_x[0] = True
            self.__ishangingFx = np.kron(
                np.ones(self.nCz, dtype=bool),
                np.kron(
                    np.ones(self.nCy, dtype=bool),
                    hang_x
                )
            )
        return self.__ishangingFx

    @property
    def _hangingFx(self):
        if getattr(self, '__hangingFx', None) is None:
            self.__hangingFx = dict(zip(
                np.nonzero(self._ishangingFx)[0].tolist(), [None]*self._nhFx
            ))
        return self.__hangingFx

    @property
    def _ishangingFy(self):
        if getattr(self, '__ishangingFy', None) is None:
            hang_y = np.zeros(self._ntNy, dtype=bool)
            hang_y[-1] = True
            self.__ishangingFy = np.kron(
                np.ones(self.nCz, dtype=bool),
                np.kron(
                    hang_y,
                    np.ones(self.nCx, dtype=bool)
                )
            )
        return self.__ishangingFy

    @property
    def _hangingFy(self):
        if getattr(self, '__hangingFy', None) is None:
            deflate_y = np.zeros(self._ntNy, dtype=bool)
            deflate_y[0] = True
            deflateFy = np.nonzero(np.kron(
                np.ones(self.nCz, dtype=bool),
                np.kron(
                    deflate_y,
                    np.ones(self.nCx, dtype=bool)
                )
            ))[0].tolist()
            self.__hangingFy = dict(zip(
                np.nonzero(self._ishangingFy)[0].tolist(),
                deflateFy)
            )
        return self.__hangingFy

    @property
    def _ishangingFz(self):
        if getattr(self, '__ishangingFz', None) is None:
            self.__ishangingFz = np.kron(
                np.zeros(self.nNz, dtype=bool),
                np.kron(
                    np.zeros(self.nCy, dtype=bool),
                    np.zeros(self.nCx, dtype=bool)
                )
            )
        return self.__ishangingFz

    @property
    def _hangingFz(self):
        return {}

    @property
    def _ishangingEx(self):
        if getattr(self, '__ishangingEx', None) is None:
            hang_y = np.zeros(self._ntNy, dtype=bool)
            hang_y[-1] = True
            self.__ishangingEx = np.kron(
                np.ones(self._ntNz, dtype=bool),
                np.kron(
                    hang_y,
                    np.ones(self.nCx, dtype=bool)
                )
            )
        return self.__ishangingEx

    @property
    def _hangingEx(self):
        if getattr(self, '__hangingEx', None) is None:
            deflate_y = np.zeros(self._ntNy, dtype=bool)
            deflate_y[0] = True
            deflateEx = np.nonzero(np.kron(
                np.ones(self._ntNz, dtype=bool),
                np.kron(
                    deflate_y,
                    np.ones(self.nCx, dtype=bool)
                )
            ))[0].tolist()
            self.__hangingEx = dict(zip(
                np.nonzero(self._ishangingEx)[0].tolist(), deflateEx
            ))
        return self.__hangingEx

    @property
    def _ishangingEy(self):
        if getattr(self, '__ishangingEy', None) is None:
            hang_x = np.zeros(self._ntNx, dtype=bool)
            hang_x[0] = True
            self.__ishangingEy = np.kron(
                np.ones(self._ntNz, dtype=bool),
                np.kron(
                    np.ones(self.nCy, dtype=bool),
                    hang_x
                )
            )
        return self.__ishangingEy

    @property
    def _hangingEy(self):
        if getattr(self, '__hangingEy', None) is None:
            self.__hangingEy = dict(zip(
                np.nonzero(self._ishangingEy)[0].tolist(),
                [None]*len(self.__ishangingEy))
            )
        return self.__hangingEy

    @property
    def _axis_of_symmetry_Ez(self):
        if getattr(self, '__axis_of_symmetry_Ez', None) is None:
            axis_x = np.zeros(self._ntNx, dtype=bool)
            axis_x[0] = True

            axis_y = np.zeros(self._ntNy, dtype=bool)
            axis_y[0] = True
            self.__axis_of_symmetry_Ez = np.kron(
                np.ones(self.nCz, dtype=bool),
                np.kron(
                    axis_y,
                    axis_x
                )
            )
        return self.__axis_of_symmetry_Ez

    @property
    def _ishangingEz(self):
        if getattr(self, '__ishangingEz', None) is None:
            hang_x = np.zeros(self._ntNx, dtype=bool)
            hang_x[0] = True

            hang_y = np.zeros(self._ntNy, dtype=bool)
            hang_y[-1] = True

            hangingEz = np.kron(
                np.ones(self.nCz, dtype=bool),
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

            self.__ishangingEz = hangingEz & ~self._axis_of_symmetry_Ez

        return self.__ishangingEz

    @property
    def _hangingEz(self):
        if getattr(self, '__hangingEz', None) is None:
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

            self.__hangingEz = dict(deflate)
        return self.__hangingEz

    @property
    def _axis_of_symmetry_N(self):
        if getattr(self, '__axis_of_symmetry_N', None) is None:
            axis_x = np.zeros(self._ntNx, dtype=bool)
            axis_x[0] = True

            axis_y = np.zeros(self._ntNy, dtype=bool)
            axis_y[0] = True
            self.__axis_of_symmetry_N = np.kron(
                np.ones(self._ntNz, dtype=bool),
                np.kron(
                    axis_y,
                    axis_x
                )
            )
        return self.__axis_of_symmetry_N

    @property
    def _ishangingN(self):
        if getattr(self, '__ishangingN', None) is None:
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

            self.__ishangingN = hangingN & ~self._axis_of_symmetry_N

        return self.__ishangingN

    @property
    def _hangingN(self):
        if getattr(self, '__hangingN', None) is None:
            # go by layer
            deflateN = np.hstack([
                np.hstack([
                    np.zeros(self._ntNy-1, dtype=int),
                    np.arange(1, self._ntNx, dtype=int)
                ]) +
                i*int(self._ntNx*self._ntNy)
                for i in range(self._ntNz)
            ]).tolist()
            self.__hangingN = dict(zip(
                np.nonzero(self._ishangingN)[0].tolist(), deflateN
            ))
        return self.__hangingN

    ####################################################
    # Grids
    ####################################################

    @property
    def _gridNFull(self):
        return utils.ndgrid([
            self.vectorNx, self._vectorNyFull, self.vectorNz
        ])

    @property
    def gridN(self):
        if self.isSymmetric:
            self._gridN = super(CylMesh, self).gridN
        if getattr(self, '_gridN', None) is None:
            self._gridN = (
                self._deflationMatrix('N', withHanging=False) *
                self._gridNFull
                # super(CylMesh, self).gridN
            )
        return self._gridN


    @property
    def gridFx(self):
        if getattr(self, '_gridFx', None) is None:
            if self.isSymmetric is True:
                return super(CylMesh, self).gridFx
            else:
                self._gridFx = (
                    self._deflationMatrix('Fx', withHanging=False) *
                    utils.ndgrid([
                        self.vectorNx, self.vectorCCy, self.vectorCCz
                    ])
                )
        return self._gridFx


    @property
    def gridEy(self):
        if getattr(self, '_gridEy', None) is None:
            if self.isSymmetric is True:
                return super(CylMesh, self).gridEy
            else:
                self._gridEy = (
                    self._deflationMatrix('Ey', withHanging=False) *
                    utils.ndgrid([
                        self.vectorNx, self.vectorCCy, self.vectorNz
                    ])
                )
        return self._gridEy

    @property
    def _gridEzFull(self):
        return utils.ndgrid([
            self.vectorNx, self._vectorNyFull, self.vectorCCz
        ])

    @property
    def gridEz(self):
        if getattr(self, '_gridEz', None) is None:
            if self.isSymmetric is True:
                self._gridEz = super(CylMesh, self).gridEz
            else:
                self._gridEz = (
                    self._deflationMatrix('Ez', withHanging=False) *
                    self._gridEzFull
                )
        return self._gridEz

    ####################################################
    # Operators
    ####################################################

    @property
    def faceDiv(self):
        """Construct divergence operator (faces to cell-centres)."""
        if getattr(self, '_faceDiv', None) is None:
            n = self.vnC
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
            D1 = utils.kron3(
                utils.speye(self.nCz),
                utils.speye(self.nCy),
                utils.ddx(self.nCx)
            ) * self._deflationMatrix('Fx', withHanging=False).T
            S = self.r(self.area, 'F', 'Fx', 'V')
            V = self.vol
            self._faceDivx = utils.sdiag(1/V)*D1*utils.sdiag(S)
        return self._faceDivx

    @property
    def faceDivy(self):
        """
        Construct divergence operator in the y component
        (faces to cell-centres).
        """
        # raise NotImplementedError(
        if getattr(self, '_faceDivy', None) is None:
            # TODO: this needs to wrap to join up faces which are
            # connected in the cylinder
            D2 = utils.kron3(
                utils.speye(self.nCz),
                utils.ddx(self.nCy),
                utils.speye(self.nCx)
            ) * self._deflationMatrix('Fy', withHanging=False).T
            S = self.r(self.area, 'F', 'Fy', 'V')
            V = self.vol
            self._faceDivy = utils.sdiag(1/V)*D2*utils.sdiag(S)
        return self._faceDivy

    @property
    def faceDivz(self):
        """
        Construct divergence operator in the z component
        (faces to cell-centres).
        """
        if getattr(self, '_faceDivz', None) is None:
            D3 = utils.kron3(
                utils.ddx(self.nCz),
                utils.speye(self.nCy),
                utils.speye(self.nCx)
            )
            S = self.r(self.area, 'F', 'Fz', 'V')
            V = self.vol
            self._faceDivz = utils.sdiag(1/V)*D3*utils.sdiag(S)
        return self._faceDivz

    # @property
    # def _cellGradxStencil(self):
    #     n = self.vnC

    #     if self.isSymmetric:
    #         G1 = sp.kron(utils.speye(n[2]), ddxCellGrad(n[0], BC))
    #     else:
    #         G1 = self._deflationMatrix('Fx').T * utils.kron3(
    #             utils.speye(n[2]), utils.speye(n[1]), ddxCellGrad(n[0], BC)
    #         )
    #     return G1

    @property
    def cellGradx(self):
        if getattr(self, '_cellGradx', None) is None:
            G1 = super(CylMesh, self)._cellGradxStencil
            V = self._deflationMatrix('F', withHanging='True', asOnes='True')*self.aveCC2F*self.vol
            A = self.area
            L = (A/V)[:self._ntFx]
            # L = self.r(L, 'F', 'Fx', 'V')
            # L = A[:self.nFx] / V
            self._cellGradx = self._deflationMatrix('Fx')*utils.sdiag(L)*G1
        return self._cellGradx


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

    @property
    def nodalGrad(self):
        """Construct gradient operator (nodes to edges)."""
        # Nodal grad does not make sense for cylindrically symmetric mesh.
        if self.isSymmetric is True:
            return None
        raise NotImplementedError('nodalGrad not yet implemented')

    @property
    def nodalLaplacian(self):
        """Construct laplacian operator (nodes to edges)."""
        raise NotImplementedError('nodalLaplacian not yet implemented')

    @property
    def edgeCurl(self):
        """The edgeCurl property."""
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
                    utils.sdiag(1/A)*sp.vstack((Dz, Dr)) * utils.sdiag(E)
                )
            else:
                self._edgeCurl = (
                    utils.sdiag(1/self.area) *
                    self._deflationMatrix('F', withHanging=True, asOnes=False) *
                    self._edgeCurlStencil *
                    utils.sdiag(self._edgeFull) *
                    self._deflationMatrix('E', withHanging=True, asOnes=True).T
                )

        return self._edgeCurl

    @property
    def aveEx2CC(self):
        "averaging operator of x-faces to cell centers"
        return utils.kron3(
            utils.av(self.vnC[2]),
            utils.av(self.vnC[1]),
            utils.speye(self.vnC[0])
        ) * self._deflationMatrix('Ex', withHanging=True, asOnes=True).T

    @property
    def aveEy2CC(self):
        "averaging from y-faces to cell centers"
        return utils.kron3(
            utils.av(self.vnC[2]),
            utils.speye(self.vnC[1]),
            utils.av(self.vnC[0])
        ) * self._deflationMatrix('Ey', withHanging=True, asOnes=True).T

    @property
    def aveEz2CC(self):
        "averaging from z-faces to cell centers"
        return utils.kron3(
            utils.speye(self.vnC[2]),
            utils.av(self.vnC[1]),
            utils.av(self.vnC[0])
        ) * self._deflationMatrix('Ez', withHanging=True, asOnes=True).T

    @property
    def aveE2CC(self):
        "Construct the averaging operator on cell edges to cell centers."
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
        "Construct the averaging operator on cell edges to cell centers."
        if getattr(self, '_aveE2CCV', None) is None:
            # The number of cell centers in each direction
            n = self.vnC
            if self.isSymmetric is True:
                return self.aveE2CC
            else:
                self._aveE2CCV = sp.block_diag(
                    (self.aveEx2CC, self.aveEy2CC, self.aveEz2CC),
                    format="csr"
                )
        return self._aveE2CCV

    @property
    def aveFx2CC(self):
        "averaging operator of x-faces to cell centers"
        avR = utils.av(self.vnC[0])[:, 1:]
        return utils.kron3(
            utils.speye(self.vnC[2]), utils.speye(self.vnC[1]), avR
        )

    @property
    def aveFy2CC(self):
        "averaging from y-faces to cell centers"

        return utils.kron3(
            utils.speye(self.vnC[2]), utils.av(self.vnC[1]),
            utils.speye(self.vnC[0])
        ) * self._deflationMatrix('Fy', withHanging=True, asOnes=True).T

    @property
    def aveFz2CC(self):
        "averaging from z-faces to cell centers"

        return utils.kron3(
            utils.av(self.vnC[2]), utils.speye(self.vnC[1]),
            utils.speye(self.vnC[0])
        )

    @property
    def aveF2CC(self):
        "Construct the averaging operator on cell faces to cell centers."
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
        "Construct the averaging operator on cell faces to cell centers."
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

    def _deflationMatrix(self, location, withHanging=True, asOnes=False):
        assert(
            location in ['N', 'F', 'Fx', 'Fy', 'Fz', 'E', 'Ex', 'Ey', 'Ez']
        )
        if location in ['E', 'F']:
            return sp.block_diag([
                self._deflationMatrix(
                    location+coord, withHanging=withHanging, asOnes=asOnes
                )
                for coord in ['x', 'y', 'z']
            ])

        R = utils.speye(getattr(self, '_nt{}'.format(location)))
        hanging = getattr(self, '_hanging{}'.format(location))
        nothanging = ~getattr(self, '_ishanging{}'.format(location))

        if withHanging:
            # remove eliminated edges / faces (eg. Fx just doesn't exist)
            hang = {k: v for k, v in hanging.items() if v is not None}

            entries = np.ones(len(hang.values()))

            if asOnes is False and len(hang) > 0:
                repeats = set(hang.values())
                repeat_locs = [
                    (np.r_[hang.values()] == repeat).nonzero()[0] for repeat in repeats
                ]
                for loc in repeat_locs:
                    entries[loc] = 1./len(loc)

            Hang = sp.csr_matrix(
                (entries, (hang.values(), hang.keys())),
                shape=(
                    getattr(self, '_nt{}'.format(location)),
                    getattr(self, '_nt{}'.format(location))
                )
            )
            R = R + Hang

        R = R[nothanging, :]

        if not asOnes:
            R = utils.sdiag(1./R.sum(1)) * R

        return R



    ####################################################
    # Interpolation
    ####################################################

    def getInterpolationMat(self, loc, locType='CC', zerosOutside=False):
        """ Produces interpolation matrix

        :param numpy.ndarray loc: Location of points to interpolate to
        :param str locType: What to interpolate (see below)
        :rtype: scipy.sparse.csr_matrix
        :return: M, the interpolation matrix

        locType can be::

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
        """
        if self.isSymmetric and locType in ['Ex', 'Ez', 'Fy']:
            raise Exception(
                "Symmetric CylMesh does not support {0!s} interpolation, "
                "as this variable does not exist.".format(locType)
            )

        if locType in ['CCVx', 'CCVy', 'CCVz']:
            Q = utils.interpmat(loc, *self.getTensor('CC'))
            Z = utils.spzeros(loc.shape[0], self.nC)
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

    def cartesianGrid(self, locType='CC'):
        """
        Takes a grid location ('CC', 'N', 'Ex', 'Ey', 'Ez', 'Fx', 'Fy', 'Fz')
        and returns that grid in cartesian coordinates
        """
        grid = getattr(self, 'grid{}'.format(locType))
        return utils.cyl2cart(grid)

    def getInterpolationMatCartMesh(self, Mrect, locType='CC', locTypeTo=None):
        """
            Takes a cartesian mesh and returns a projection to translate onto
            the cartesian grid.
        """

        assert self.isSymmetric, (
            "Currently we have not taken into account other projections for "
            "more complicated CylMeshes"
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
            Z = utils.spzeros(getattr(Mrect, 'n' + locTypeTo + 'z'), self.nE)
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
        Proj = utils.sdiag(proj)
        return Proj * Pc2r
