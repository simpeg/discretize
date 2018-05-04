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
from .TensorMesh import BaseTensorMesh
from .InnerProducts import InnerProducts
from .MeshIO import TreeMeshIO
from .tree_ext import _TreeMesh
import numpy as np
from scipy.spatial import Delaunay
import scipy.sparse as sp
from . import utils
import six

class TreeMesh(_TreeMesh, BaseTensorMesh, InnerProducts, TreeMeshIO):
    _meshType = 'TREE'

    #inheriting stuff from BaseTensorMesh that isn't defined in _QuadTree
    def __init__(self, h, x0=None, levels=None, **kwargs):
        BaseTensorMesh.__init__(self, h, x0, **kwargs)

        if levels is None:
            levels = int(np.log2(len(self.h[0])))

        # Now can initialize cpp tree parent
        _TreeMesh.__init__(self, levels, self.x0, self.h)

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
                self._cellGradxStencil(), self._cellGradyStencil()
            ])
            if self.dim == 3:
                self._cellGradStencil = sp.vstack([
                    self._cellGradStencil, self._cellGradzStencil()
                ])

        return self._cellGradStencil

    @property
    def cellGrad(self):
        """
        Cell centered Gradient operator built off of the faceDiv operator.
        Grad =  - (Mf)^{-1} * Div * diag (volume)
        """
        if getattr(self, '_cellGrad', None) is None:

            indBoundary = np.ones(self.nC, dtype=float)

            indBoundary_Fx = (self.aveFx2CC.T * indBoundary) >= 1
            ix = np.zeros(self.nFx)
            ix[indBoundary_Fx] = 1.
            Pafx = sp.diags(ix)

            indBoundary_Fy = (self.aveFy2CC.T * indBoundary) >= 1
            iy = np.zeros(self.nFy)
            iy[indBoundary_Fy] = 1.
            Pafy = sp.diags(iy)

            MfI = self.getFaceInnerProduct(invMat=True)

            if self.dim == 2:
                Pi = sp.block_diag([Pafx, Pafy])

            elif self.dim == 3:
                indBoundary_Fz = (self.aveFz2CC.T * indBoundary) >= 1
                iz = np.zeros(self.nFz)
                iz[indBoundary_Fz] = 1.
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
            indBoundary = np.ones(self.nC, dtype=float)

            indBoundary_Fx = (self.aveFx2CC.T * indBoundary) >= 1
            ix = np.zeros(self.nFx)
            ix[indBoundary_Fx] = 1.
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
            indBoundary = np.ones(self.nC, dtype=float)

            indBoundary_Fy = (self.aveFy2CC.T * indBoundary) >= 1
            iy = np.zeros(self.nFy)
            iy[indBoundary_Fy] = 1.
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
        Cell centered Gradient operator in y-direction (Gradz)
        Grad = sp.vstack((Gradx, Grady, Gradz))
        """
        if getattr(self, '_cellGradz', None) is None:

            nFx = self.nFx
            nFy = self.nFy
            indBoundary = np.ones(self.nC, dtype=float)

            indBoundary_Fz = (self.aveFz2CC.T * indBoundary) >= 1
            iz = np.zeros(self.nFz)
            iz[indBoundary_Fz] = 1.
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

    @property
    def aveCC2Fx(self):
        "Construct the averaging operator on cell centers to cell x-faces."
        if getattr(self, '_aveCC2Fx', None) is None:
            tri = Delaunay(self.gridCC)
            gridF = self.gridFx
            simplexes = tri.find_simplex(gridF)
            nd = self.dim
            ns = nd+1
            nf = self.nFx
            I = np.zeros(nf*ns, dtype=np.int64)
            J = np.zeros(nf*ns, dtype=np.int64)
            V = np.zeros(nf*ns, dtype=np.float64)

            simp = tri.simplices[simplexes]
            trans = tri.transform[simplexes]
            shift = gridF-trans[:, nd]
            bs = np.einsum('ikj,ij->ik', trans[:, :nd], shift)
            bs = np.c_[bs, 1-bs.sum(axis=1)]

            I = np.zeros(nf*ns, dtype=np.int64)
            J = np.zeros(nf*ns, dtype=np.int64)
            V = np.zeros(nf*ns, dtype=np.float64)
            for i in range(nf):
                I[ns*i: ns*i+ns] = i
                if simplexes[i] == -1:
                    # Extrapolating.... from nearest cell
                    ic = self._get_containing_cell_index(gridF[i])
                    J[ns*i] = ic
                    V[ns*i] = 1.0
                    # rest are zeros from definition of J and V
                else:
                    J[ns*i:ns*i+ns] = simp[i]
                    V[ns*i:ns*i+ns] = bs[i]
            self._aveCC2Fx = sp.csr_matrix((V, (I, J)))
        return self._aveCC2Fx

    @property
    def aveCC2Fy(self):
        "Construct the averaging operator on cell centers to cell y-faces."
        if getattr(self, '_aveCC2Fy', None) is None:
            tri = Delaunay(self.gridCC)
            gridF = self.gridFy
            simplexes = tri.find_simplex(gridF)
            nd = self.dim
            ns = nd+1
            nf = self.nFy
            I = np.zeros(nf*ns, dtype=np.int64)
            J = np.zeros(nf*ns, dtype=np.int64)
            V = np.zeros(nf*ns, dtype=np.float64)

            simp = tri.simplices[simplexes]
            trans = tri.transform[simplexes]
            shift = gridF-trans[:, nd]
            bs = np.einsum('ikj,ij->ik', trans[:, :nd], shift)
            bs = np.c_[bs, 1-bs.sum(axis=1)]

            I = np.zeros(nf*ns, dtype=np.int64)
            J = np.zeros(nf*ns, dtype=np.int64)
            V = np.zeros(nf*ns, dtype=np.float64)
            for i in range(nf):
                I[ns*i: ns*i+ns] = i
                if simplexes[i] == -1:
                    # Extrapolating.... from nearest cell
                    ic = self._get_containing_cell_index(gridF[i])
                    J[ns*i] = ic
                    V[ns*i] = 1.0
                    # rest are zeros from definition of J and V
                else:
                    J[ns*i:ns*i+ns] = simp[i]
                    V[ns*i:ns*i+ns] = bs[i]
            self._aveCC2Fy = sp.csr_matrix((V, (I, J)))
        return self._aveCC2Fy

    @property
    def aveCC2Fz(self):
        "Construct the averaging operator on cell centers to cell z-faces."
        if self.dim == 2:
            raise Exception('TreeMesh has no z-faces in 2D')
        if getattr(self, '_aveCC2Fz', None) is None:
            tri = Delaunay(self.gridCC)
            gridF = self.gridFz
            simplexes = tri.find_simplex(gridF)
            nd = self.dim
            ns = nd+1
            nf = self.nFz
            I = np.zeros(nf*ns, dtype=np.int64)
            J = np.zeros(nf*ns, dtype=np.int64)
            V = np.zeros(nf*ns, dtype=np.float64)

            simp = tri.simplices[simplexes]

            trans = tri.transform[simplexes]
            shift = gridF-trans[:, nd]
            bs = np.einsum('ikj,ij->ik', trans[:, :nd], shift)
            bs = np.c_[bs, 1-bs.sum(axis=1)]

            I = np.zeros(nf*ns, dtype=np.int64)
            J = np.zeros(nf*ns, dtype=np.int64)
            V = np.zeros(nf*ns, dtype=np.float64)
            for i in range(nf):
                I[ns*i: ns*i+ns] = i
                if simplexes[i] == -1:
                    # Extrapolating.... from nearest cell
                    ic = self._get_containing_cell_index(gridF[i])
                    J[ns*i] = ic
                    V[ns*i] = 1.0
                    # rest are zeros from definition of J and V
                else:
                    J[ns*i:ns*i+ns] = simp[i]
                    V[ns*i:ns*i+ns] = bs[i]
            self._aveCC2Fz = sp.csr_matrix((V, (I, J)))
        return self._aveCC2Fz

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

    def __reduce__(self):
        return TreeMesh, (self.h, self.x0), self.__getstate__()
