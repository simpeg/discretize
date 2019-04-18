#      ___          ___       ___          ___          ___          ___
#     /\  \        /\  \     /\  \        /\  \        /\  \        /\  \
#    /::\  \      /::\  \    \:\  \      /::\  \      /::\  \      /::\  \
#   /:/\:\  \    /:/\:\  \    \:\  \    /:/\:\  \    /:/\:\  \    /:/\:\  \
#  /:/  \:\  \  /:/  \:\  \   /::\  \  /::\~\:\  \  /::\~\:\  \  /::\~\:\  \
# /:/__/ \:\__\/:/__/ \:\__\ /:/\:\__\/:/\:\ \:\__\/:/\:\ \:\__\/:/\:\ \:\__\
# \:\  \ /:/  /\:\  \  \/__//:/  \/__/\/_|::\/:/  /\:\~\:\ \/__/\:\~\:\ \/__/
#  \:\  /:/  /  \:\  \     /:/  /        |:|::/  /  \:\ \:\__\   \:\ \:\__\
#   \:\/:/  /    \:\  \    \/__/         |:|\/__/    \:\ \/__/    \:\ \/__/
#    \::/  /      \:\__\                 |:|  |       \:\__\       \:\__\
#     \/__/        \/__/                  \|__|        \/__/        \/__/
#
#
#
#                      .----------------.----------------.
#                     /|               /|               /|
#                    / |              / |              / |
#                   /  |      6      /  |     7       /  |
#                  /   |            /   |            /   |
#                 .----------------.----+-----------.    |
#                /|    . ---------/|----.----------/|----.
#               / |   /|         / |   /|         / |   /|
#              /  |  / |  4     /  |  / |   5    /  |  / |
#             /   | /  |       /   | /  |       /   | /  |
#            . -------------- .----------------.    |/   |
#            |    . ---+------|----.----+------|----.    |
#            |   /|    .______|___/|____.______|___/|____.
#            |  / |   /    2  |  / |   /     3 |  / |   /
#            | /  |  /        | /  |  /        | /  |  /
#            . ---+---------- . ---+---------- .    | /
#            |    |/          |    |/          |    |/             z
#            |    . ----------|----.-----------|----.              ^   y
#            |   /      0     |   /       1    |   /               |  /
#            |  /             |  /             |  /                | /
#            | /              | /              | /                 o----> x
#            . -------------- . -------------- .
#
#
# Face Refinement:
#
#      2_______________3                    _______________
#      |               |                   |       |       |
#   ^  |               |                   |   2   |   3   |
#   |  |               |                   |       |       |
#   |  |       x       |        --->       |-------+-------|
#   t1 |               |                   |       |       |
#      |               |                   |   0   |   1   |
#      |_______________|                   |_______|_______|
#      0      t0-->    1
#
#
# Face and Edge naming conventions:
#
#                      fZp
#                       |
#                 6 ------eX3------ 7
#                /|     |         / |
#               /eZ2    .        / eZ3
#             eY2 |        fYp eY3  |
#             /   |            / fXp|
#            4 ------eX2----- 5     |
#            |fXm 2 -----eX1--|---- 3          z
#           eZ0  /            |  eY1           ^   y
#            | eY0   .  fYm  eZ1 /             |  /
#            | /     |        | /              | /
#            0 ------eX0------1                o----> x
#                    |
#                   fZm
#
#
#            fX                                  fY
#      2___________3                       2___________3
#      |     e1    |                       |     e1    |
#      |           |                       |           |
#   e0 |     x     | e2      z          e0 |     x     | e2      z
#      |           |         ^             |           |         ^
#      |___________|         |___> y       |___________|         |___> x
#      0    e3     1                       0    e3     1
#           fZ
#      2___________3
#      |     e1    |
#      |           |
#   e0 |     x     | e2      y
#      |           |         ^
#      |___________|         |___> x
#      0    e3     1
from .base import BaseTensorMesh
from .InnerProducts import InnerProducts
from .MeshIO import TreeMeshIO
from . import utils
from .tree_ext import _TreeMesh
import numpy as np
from scipy.spatial import Delaunay
import scipy.sparse as sp
from six import integer_types

class TreeMesh(_TreeMesh, BaseTensorMesh, InnerProducts, TreeMeshIO):
    """
    TreeMesh is a class for adaptive QuadTree (2D) and OcTree (3D) meshes.
    """
    _meshType = 'TREE'

    #inheriting stuff from BaseTensorMesh that isn't defined in _QuadTree
    def __init__(self, h, x0=None, **kwargs):
        BaseTensorMesh.__init__(self, h, x0)#TODO:, **kwargs) # pass the kwargs for copy/paste

        nx = len(self.h[0])
        ny = len(self.h[1])
        nz = len(self.h[2]) if self.dim == 3 else 2
        def is_pow2(num): return ((num & (num - 1)) == 0) and num != 0
        if not (is_pow2(nx) and is_pow2(ny) and is_pow2(nz)):
            raise ValueError("length of cell width vectors must be a power of 2")
        # Now can initialize cpp tree parent
        _TreeMesh.__init__(self, self.h, self.x0)

    def __str__(self):
        outStr = '  ---- {0!s}TreeMesh ----  '.format(
            ('Oc' if self.dim == 3 else 'Quad')
        )

        def printH(hx, outStr=''):
            i = -1
            while True:
                i = i + 1
                if i > hx.size:
                    break
                elif i == hx.size:
                    break
                h = hx[i]
                n = 1
                for j in range(i+1, hx.size):
                    if hx[j] == h:
                        n = n + 1
                        i = i + 1
                    else:
                        break
                if n == 1:
                    outStr += ' {0:.2f}, '.format(h)
                else:
                    outStr += ' {0:d}*{1:.2f}, '.format(n, h)
            return outStr[:-1]

        if self.dim == 2:
            outStr += '\n   x0: {0:.2f}'.format(self.x0[0])
            outStr += '\n   y0: {0:.2f}'.format(self.x0[1])
            outStr += printH(self.hx, outStr='\n   hx:')
            outStr += printH(self.hy, outStr='\n   hy:')
        elif self.dim == 3:
            outStr += '\n   x0: {0:.2f}'.format(self.x0[0])
            outStr += '\n   y0: {0:.2f}'.format(self.x0[1])
            outStr += '\n   z0: {0:.2f}'.format(self.x0[2])
            outStr += printH(self.hx, outStr='\n   hx:')
            outStr += printH(self.hy, outStr='\n   hy:')
            outStr += printH(self.hz, outStr='\n   hz:')
        outStr += '\n  nC: {0:d}'.format(self.nC)
        outStr += '\n  Fill: {0:2.2f}%'.format((self.fill*100))
        return outStr

    @property
    def vntF(self):
        return [self.ntFx, self.ntFy] + ([] if self.dim == 2 else [self.ntFz])

    @property
    def vntE(self):
        return [self.ntEx, self.ntEy] + ([] if self.dim == 2 else [self.ntEz])

    @property
    def cellGradStencil(self):
        if getattr(self, '_cellGradStencil', None) is None:

            self._cellGradStencil = sp.vstack([
                self._cellGradxStencil, self._cellGradyStencil
            ])
            if self.dim == 3:
                self._cellGradStencil = sp.vstack([
                    self._cellGradStencil, self._cellGradzStencil
                ])

        return self._cellGradStencil

    @property
    def cellGrad(self):
        """
        Cell centered Gradient operator built off of the faceDiv operator.
        Grad =  - (Mf)^{-1} * Div * diag (volume)
        """
        if getattr(self, '_cellGrad', None) is None:

            i_s = self.faceBoundaryInd

            ix = np.ones(self.nFx)
            ix[i_s[0]] = 0.
            ix[i_s[1]] = 0.
            Pafx = sp.diags(ix)

            iy = np.ones(self.nFy)
            iy[i_s[2]] = 0.
            iy[i_s[3]] = 0.
            Pafy = sp.diags(iy)

            MfI = self.getFaceInnerProduct(invMat=True)

            if self.dim == 2:
                Pi = sp.block_diag([Pafx, Pafy])

            elif self.dim == 3:
                iz = np.ones(self.nFz)
                iz[i_s[4]] = 0.
                iz[i_s[5]] = 0.
                Pafz = sp.diags(iz)
                Pi = sp.block_diag([Pafx, Pafy, Pafz])

            self._cellGrad = -Pi * MfI * self.faceDiv.T * sp.diags(self.vol)

        return self._cellGrad

    @property
    def cellGradx(self):
        """
        Cell centered Gradient operator in x-direction (Gradx)
        Grad = sp.vstack((Gradx, Grady, Gradz))
        """
        if getattr(self, '_cellGradx', None) is None:

            nFx = self.nFx
            i_s = self.faceBoundaryInd

            ix = np.ones(self.nFx)
            ix[i_s[0]] = 0.0
            ix[i_s[1]] = 0.0
            Pafx = sp.diags(ix)

            MfI = self.getFaceInnerProduct(invMat=True)
            MfIx = sp.diags(MfI.diagonal()[:nFx])

            self._cellGradx = (
                -Pafx * MfIx * self.faceDivx.T * sp.diags(self.vol)
            )

        return self._cellGradx

    @property
    def cellGrady(self):
        """
        Cell centered Gradient operator in y-direction (Gradx)
        Grad = sp.vstack((Gradx, Grady, Gradz))
        """
        if getattr(self, '_cellGrady', None) is None:

            nFx = self.nFx
            nFy = self.nFy
            i_s = self.faceBoundaryInd

            iy = np.ones(self.nFy)
            iy[i_s[2]] = 0.
            iy[i_s[3]] = 0.
            Pafy = sp.diags(iy)

            MfI = self.getFaceInnerProduct(invMat=True)
            MfIy = sp.diags(MfI.diagonal()[nFx:nFx+nFy])

            self._cellGrady = (
                -Pafy * MfIy * self.faceDivy.T * sp.diags(self.vol)
            )

        return self._cellGrady

    @property
    def cellGradz(self):
        """
        Cell centered Gradient operator in z-direction (Gradz)
        Grad = sp.vstack((Gradx, Grady, Gradz))
        """
        if getattr(self, '_cellGradz', None) is None:

            nFx = self.nFx
            nFy = self.nFy
            i_s = self.faceBoundaryInd

            iz = np.ones(self.nFz)
            iz[i_s[4]] = 0.
            iz[i_s[5]] = 0.
            Pafz = sp.diags(iz)

            MfI = self.getFaceInnerProduct(invMat=True)
            MfIz = sp.diags(MfI.diagonal()[nFx+nFy:])

            self._cellGradz = (
                -Pafz * MfIz * self.faceDivz.T * sp.diags(self.vol)
            )

        return self._cellGradz

    @property
    def faceDivx(self):
        if getattr(self, '_faceDivx', None) is None:
            self._faceDivx = self.faceDiv[:, :self.nFx]
        return self._faceDivx

    @property
    def faceDivy(self):
        if getattr(self, '_faceDivy', None) is None:
            self._faceDivy = self.faceDiv[:, self.nFx:self.nFx+self.nFy]
        return self._faceDivy

    @property
    def faceDivz(self):
        if getattr(self, '_faceDivz', None) is None:
            self._faceDivz = self.faceDiv[:, self.nFx+self.nFy:]
        return self._faceDivz

    def point2index(self, locs):
        locs = utils.asArray_N_x_Dim(locs, self.dim)

        inds = np.empty(locs.shape[0], dtype=np.int64)
        for ind, loc in enumerate(locs):
            inds[ind] = self._get_containing_cell_index(loc)
        return inds

    @property
    def permuteCC(self):
        # TODO: cache these?
        P = np.lexsort(self.gridCC.T) # sort by x, then y, then z
        return sp.identity(self.nC).tocsr()[P]

    @property
    def permuteF(self):
        # TODO: cache these?
        Px = np.lexsort(self.gridFx.T)
        Py = np.lexsort(self.gridFy.T)+self.nFx
        if self.dim == 2:
            P = np.r_[Px, Py]
        else:
            Pz = np.lexsort(self.gridFz.T)+(self.nFx+self.nFy)
            P = np.r_[Px, Py, Pz]
        return sp.identity(self.nF).tocsr()[P]

    @property
    def permuteE(self):
        # TODO: cache these?
        Px = np.lexsort(self.gridEx.T)
        Py = np.lexsort(self.gridEy.T) + self.nEx
        if self.dim == 2:
            P = np.r_[Px, Py]
        if self.dim == 3:
            Pz = np.lexsort(self.gridEz.T) + (self.nEx+self.nEy)
            P = np.r_[Px, Py, Pz]
        return sp.identity(self.nE).tocsr()[P]

    def plotSlice(
        self, v, vType='CC',
        normal='Z', ind=None, grid=False, view='real',
        ax=None, clim=None, showIt=False,
        pcolorOpts=None, streamOpts=None, gridOpts=None,
        range_x=None, range_y=None,
    ):

        if pcolorOpts is None:
            pcolorOpts = {}
        if streamOpts is None:
            streamOpts = {'color': 'k'}
        if gridOpts is None:
            gridOpts = {'color': 'k', 'alpha': 0.5}
        vTypeOpts = ['CC', 'N', 'F', 'E', 'Fx', 'Fy', 'Fz', 'E', 'Ex', 'Ey', 'Ez']
        viewOpts = ['real', 'imag', 'abs']
        normalOpts = ['X', 'Y', 'Z']
        if vType not in vTypeOpts:
            raise ValueError(
                "vType must be in ['{0!s}']".format("', '".join(vTypeOpts))
            )
        if self.dim == 2:
            raise NotImplementedError(
                'Must be a 3D mesh. Use plotImage.'
            )
        if view == 'vec':
            raise NotImplementedError(
                'Vector view plotting is not implemented for TreeMesh (yet)'
            )
        if view not in viewOpts:
            raise ValueError(
                "view must be in ['{0!s}']".format("', '".join(viewOpts))
            )
        normal = normal.upper()
        if normal not in normalOpts:
            raise ValueError(
                "normal must be in ['{0!s}']".format("', '".join(normalOpts))
            )

        if not isinstance(grid, bool):
            raise TypeError('grid must be a boolean')

        import matplotlib.pyplot as plt
        import matplotlib

        normalInd = {'X': 0, 'Y': 1, 'Z': 2}[normal]
        antiNormalInd = {'X': [1, 2], 'Y': [0, 2], 'Z': [0, 1]}[normal]

        h2d = (self.h[antiNormalInd[0]], self.h[antiNormalInd[1]])
        x2d = (self.x0[antiNormalInd[0]], self.x0[antiNormalInd[1]])

        #: Size of the sliced dimension
        szSliceDim = len(self.h[normalInd])
        if ind is None:
            ind = int(szSliceDim//2)

        cc_tensor = [None, None, None]
        for i in range(3):
            cc_tensor[i] = np.cumsum(np.r_[self.x0[i], self.h[i]])
            cc_tensor[i] = (cc_tensor[i][1:] + cc_tensor[i][:-1])*0.5
        slice_loc = cc_tensor[normalInd][ind]

        if type(ind) not in integer_types:
            raise ValueError('ind must be an integer')

        # create a temporary TreeMesh with the slice through
        temp_mesh = TreeMesh(h2d, x2d)
        level_diff = self.max_level - temp_mesh.max_level

        XS = [None, None, None]
        XS[antiNormalInd[0]], XS[antiNormalInd[1]] = np.meshgrid(cc_tensor[antiNormalInd[0]],
                                                                 cc_tensor[antiNormalInd[1]])
        XS[normalInd] = np.ones_like(XS[antiNormalInd[0]])*slice_loc
        loc_grid = np.c_[XS[0].reshape(-1), XS[1].reshape(-1), XS[2].reshape(-1)]
        inds = np.unique(self._get_containing_cell_indexes(loc_grid))

        grid2d = self.gridCC[inds][:, antiNormalInd]
        levels = self._cell_levels_by_indexes(inds) - level_diff
        temp_mesh.insert_cells(grid2d, levels)
        tm_gridboost = np.empty((temp_mesh.nC, 3))
        tm_gridboost[:, antiNormalInd] = temp_mesh.gridCC
        tm_gridboost[:, normalInd] = slice_loc

        # interpolate values to self.gridCC if not 'CC'
        if vType is not 'CC':
            aveOp = 'ave' + vType + '2CC'
            Av = getattr(self, aveOp)
            if v.size == Av.shape[1]:
                v = Av*v
            elif len(vType) == 2:
                # was one of Fx, Fy, Fz, Ex, Ey, Ez
                # assuming v has all three components in these cases
                vec_ind = {'x': 0, 'y': 1, 'z': 2}[vType[1]]
                if vType[0] == 'E':
                    i_s = np.cumsum([0, self.nEx, self.nEy, self.nEz])
                elif vType[0] == 'F':
                    i_s = np.cumsum([0, self.nFx, self.nFy, self.nFz])
                v = v[i_s[vec_ind]:i_s[vec_ind+1]]
                v = Av*v

        # interpolate values from self.gridCC to grid2d
        ind_3d_to_2d = self._get_containing_cell_indexes(tm_gridboost)
        v2d = v[ind_3d_to_2d]

        if ax is None:
            plt.figure()
            ax = plt.subplot(111)
        elif not isinstance(ax, matplotlib.axes.Axes):
            raise Exception("ax must be an matplotlib.axes.Axes")

        out = temp_mesh.plotImage(
            v2d, vType='CC',
            grid=grid, view=view,
            ax=ax, clim=clim, showIt=False,
            pcolorOpts=pcolorOpts,
            gridOpts=gridOpts,
            range_x=range_x,
            range_y=range_y)

        ax.set_xlabel('y' if normal == 'X' else 'x')
        ax.set_ylabel('y' if normal == 'Z' else 'z')
        ax.set_title(
            'Slice {0:d}, {1!s} = {2:4.2f}'.format(ind, normal, slice_loc)
        )
        if showIt:
            plt.show()
        return tuple(out)

    def save(self, *args, **kwargs):
        raise NotImplementedError()

    def load(self, *args, **kwargs):
        raise NotImplementedError()

    def copy(self, *args, **kwargs):
        raise NotImplementedError()

    def __reduce__(self):
        return TreeMesh, (self.h, self.x0), self.__getstate__()
