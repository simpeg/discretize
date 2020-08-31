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

import properties

from .base import BaseTensorMesh
from .InnerProducts import InnerProducts
from .MeshIO import TreeMeshIO
from . import utils
from .tree_ext import _TreeMesh, TreeCell
import numpy as np
from scipy.spatial import Delaunay
import scipy.sparse as sp
from six import integer_types
import warnings

from discretize.utils.codeutils import requires
# matplotlib is a soft dependencies for discretize
try:
    import matplotlib.pyplot as plt
    import matplotlib
except ImportError:
    matplotlib = False


class TreeMesh(_TreeMesh, BaseTensorMesh, InnerProducts, TreeMeshIO):
    """
    TreeMesh is a class for adaptive QuadTree (2D) and OcTree (3D) meshes.
    """
    _meshType = 'TREE'

    #inheriting stuff from BaseTensorMesh that isn't defined in _QuadTree
    def __init__(self, h=None, x0=None, **kwargs):
        if 'h' in kwargs.keys():
            h = kwargs.pop('h')
        if 'x0' in kwargs.keys():
            x0 = kwargs.pop('x0')
        # print(h, x0)
        BaseTensorMesh.__init__(self, h, x0)#TODO:, **kwargs) # pass the kwargs for copy/paste

        nx = len(self.h[0])
        ny = len(self.h[1])
        nz = len(self.h[2]) if self.dim == 3 else 2
        def is_pow2(num): return ((num & (num - 1)) == 0) and num != 0
        if not (is_pow2(nx) and is_pow2(ny) and is_pow2(nz)):
            raise ValueError("length of cell width vectors must be a power of 2")
        # Now can initialize cpp tree parent
        _TreeMesh.__init__(self, self.h, self.x0)

        if 'cell_levels' in kwargs.keys() and 'cell_indexes' in kwargs.keys():
            inds = kwargs.pop('cell_indexes')
            levels = kwargs.pop('cell_levels')
            self.__setstate__((inds, levels))

    def __repr__(self):
        """Plain text representation."""
        mesh_name = '{0!s}TreeMesh'.format(('Oc' if self.dim==3 else 'Quad'))

        top = "\n"+mesh_name+": {0:2.2f}% filled\n\n".format(self.fill*100)

        # Number of cells per level
        level_count = self._count_cells_per_index()
        non_zero_levels = np.nonzero(level_count)[0]
        cell_display = ["Level : Number of cells"]
        cell_display.append("-----------------------")
        for level in non_zero_levels:
            cell_display.append("{:^5} : {:^15}".format(level, level_count[level]))
        cell_display.append("-----------------------")
        cell_display.append("Total : {:^15}".format(self.nC))

        extent_display =     ["            Mesh Extent       "]
        extent_display.append("        min     ,     max     ")
        extent_display.append("   ---------------------------")
        dim_label = {0:'x',1:'y',2:'z'}
        for dim in range(self.dim):
            n_vector = getattr(self, 'vectorN'+dim_label[dim])
            extent_display.append("{}: {:^13},{:^13}".format(dim_label[dim], n_vector[0], n_vector[-1]))

        for i, line in enumerate(extent_display):
            if i==len(cell_display):
                cell_display.append(" "*(len(cell_display[0])-3-len(line)))
            cell_display[i] += 3*" " + line

        h_display =     ['     Cell Widths    ']
        h_display.append("    min   ,   max   ")
        h_display.append("-"*(len(h_display[0])))
        h_gridded = self.h_gridded
        mins = np.min(h_gridded,axis=0)
        maxs = np.max(h_gridded,axis=0)
        for dim in range(self.dim):
            h_display.append("{:^10}, {:^10}".format(mins[dim], maxs[dim]))

        for i, line in enumerate(h_display):
            if i==len(cell_display):
                cell_display.append(" "*len(cell_display[0]))
            cell_display[i] += 3*" " + line

        return top+"\n".join(cell_display)

    def _repr_html_(self):
        """html representation"""
        mesh_name = '{0!s}TreeMesh'.format(('Oc' if self.dim==3 else 'Quad'))
        level_count = self._count_cells_per_index()
        non_zero_levels = np.nonzero(level_count)[0]
        dim_label = {0:'x',1:'y',2:'z'}
        h_gridded = self.h_gridded
        mins = np.min(h_gridded,axis=0)
        maxs = np.max(h_gridded,axis=0)

        style = " style='padding: 5px 20px 5px 20px;'"
        #Cell level table:
        cel_tbl =  "<table>\n"
        cel_tbl += "<tr>\n"
        cel_tbl += "<th"+style+">Level</th>\n"
        cel_tbl += "<th"+style+">Number of cells</th>\n"
        cel_tbl += "</tr>\n"
        for level in non_zero_levels:
            cel_tbl += "<tr>\n"
            cel_tbl += "<td"+style+">{}</td>\n".format(level)
            cel_tbl += "<td"+style+">{}</td>\n".format(level_count[level])
            cel_tbl += "</tr>\n"
        cel_tbl += "<tr>\n"
        cel_tbl += "<td style='font-weight: bold; padding: 5px 20px 5px 20px;'> Total </td>\n"
        cel_tbl += "<td"+style+"> {} </td>\n".format(self.nC)
        cel_tbl += "</tr>\n"
        cel_tbl += "</table>\n"

        det_tbl =  "<table>\n"
        det_tbl += "<tr>\n"
        det_tbl += "<th></th>\n"
        det_tbl += "<th"+style+" colspan='2'>Mesh extent</th>\n"
        det_tbl += "<th"+style+" colspan='2'>Cell widths</th>\n"
        det_tbl += "</tr>\n"

        det_tbl += "<tr>\n"
        det_tbl += "<th></th>\n"
        det_tbl += "<th"+style+">min</th>\n"
        det_tbl += "<th"+style+">max</th>\n"
        det_tbl += "<th"+style+">min</th>\n"
        det_tbl += "<th"+style+">max</th>\n"
        det_tbl += "</tr>\n"
        for dim in range(self.dim):
            n_vector = getattr(self, 'vectorN'+dim_label[dim])
            det_tbl += "<tr>\n"
            det_tbl += "<td"+style+">{}</td>\n".format(dim_label[dim])
            det_tbl += "<td"+style+">{}</td>\n".format(n_vector[0])
            det_tbl += "<td"+style+">{}</td>\n".format(n_vector[-1])
            det_tbl += "<td"+style+">{}</td>\n".format(mins[dim])
            det_tbl += "<td"+style+">{}</td>\n".format(maxs[dim])
            det_tbl += "</tr>\n"
        det_tbl += "</table>\n"

        full_tbl =  "<table>\n"
        full_tbl += "<tr>\n"
        full_tbl += "<td style='font-weight: bold; font-size: 1.2em; text-align: center;'>{}</td>\n".format(mesh_name)
        full_tbl += "<td style='font-size: 1.2em; text-align: center;' colspan='2'>{0:2.2f}% filled</td>\n".format(100*self.fill)
        full_tbl += "</tr>\n"
        full_tbl += "<tr>\n"

        full_tbl += "<td>\n"
        full_tbl += cel_tbl
        full_tbl += "</td>\n"

        full_tbl += "<td>\n"
        full_tbl += det_tbl
        full_tbl += "</td>\n"

        full_tbl += "</tr>\n"
        full_tbl += "</table>\n"

        return full_tbl

    @properties.validator('x0')
    def _x0_validator(self, change):
        self._set_x0(change['value'])

    @property
    def vntF(self):
        """Total number of hanging and non-hanging faces in a [nx,ny,nz] form"""
        return [self.ntFx, self.ntFy] + ([] if self.dim == 2 else [self.ntFz])

    @property
    def vntE(self):
        """Total number of hanging and non-hanging edges in a [nx,ny,nz] form"""
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
        Cell centered Gradient operator in y-direction (Grady)
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
        if self.dim == 2:
            raise TypeError("z derivative not defined in 2D")
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
        """Finds cells that contain the given points.
        Returns an array of index values of the cells that contain the given
        points

        Parameters
        ----------
        locs: array_like of shape (N, dim)
            points to search for the location of

        Returns
        -------
        numpy.array of integers of length(N)
            Cell indices that contain the points
        """
        locs = utils.asArray_N_x_Dim(locs, self.dim)
        inds = self._get_containing_cell_indexes(locs)
        return inds

    def cell_levels_by_index(self, indices):
        """Fast function to return a list of levels for the given cell indices

        Parameters
        ----------
        index: array_like of length (N)
            Cell indexes to query

        Returns
        -------
        numpy.array of length (N)
            Levels for the cells.
        """

        return self._cell_levels_by_indexes(indices)


    def getInterpolationMat(self, locs, locType, zerosOutside=False):
        """ Produces interpolation matrix

        Parameters
        ----------
        loc : numpy.ndarray
            Location of points to interpolate to

        locType: str
            What to interpolate

            locType can be::

                'Ex'    -> x-component of field defined on edges
                'Ey'    -> y-component of field defined on edges
                'Ez'    -> z-component of field defined on edges
                'Fx'    -> x-component of field defined on faces
                'Fy'    -> y-component of field defined on faces
                'Fz'    -> z-component of field defined on faces
                'N'     -> scalar field defined on nodes
                'CC'    -> scalar field defined on cell centers

        Returns
        -------
        scipy.sparse.csr_matrix
            M, the interpolation matrix

        """
        locs = utils.asArray_N_x_Dim(locs, self.dim)
        if locType not in ['N', 'CC', "Ex", "Ey", "Ez", "Fx", "Fy", "Fz"]:
            raise Exception('locType must be one of N, CC, Ex, Ey, Ez, Fx, Fy, or Fz')

        if self.dim == 2 and locType in ['Ez', 'Fz']:
            raise Exception('Unable to interpolate from Z edges/face in 2D')

        locs = np.require(np.atleast_2d(locs), dtype=np.float64, requirements='C')

        if locType == 'N':
            Av = self._getNodeIntMat(locs, zerosOutside)
        elif locType in ['Ex', 'Ey', 'Ez']:
            Av = self._getEdgeIntMat(locs, zerosOutside, locType[1])
        elif locType in ['Fx', 'Fy', 'Fz']:
            Av = self._getFaceIntMat(locs, zerosOutside, locType[1])
        elif locType in ['CC']:
            Av = self._getCellIntMat(locs, zerosOutside)
        return Av

    @property
    def permuteCC(self):
        """Permutation matrix re-ordering of cells sorted by x, then y, then z"""
        # TODO: cache these?
        P = np.lexsort(self.gridCC.T) # sort by x, then y, then z
        return sp.identity(self.nC).tocsr()[P]

    @property
    def permuteF(self):
        """Permutation matrix re-ordering of faces sorted by x, then y, then z"""
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
        """Permutation matrix re-ordering of edges sorted by x, then y, then z"""
        # TODO: cache these?
        Px = np.lexsort(self.gridEx.T)
        Py = np.lexsort(self.gridEy.T) + self.nEx
        if self.dim == 2:
            P = np.r_[Px, Py]
        if self.dim == 3:
            Pz = np.lexsort(self.gridEz.T) + (self.nEx+self.nEy)
            P = np.r_[Px, Py, Pz]
        return sp.identity(self.nE).tocsr()[P]

    @requires({'matplotlib': matplotlib})
    def plotSlice(
        self, v, v_type='CC',
        normal='Z', ind=None, grid=False, view='real',
        ax=None, clim=None, show_it=False,
        pcolor_opts=None, stream_opts=None, grid_opts=None,
        range_x=None, range_y=None, **kwargs
    ):
        if "pcolorOpts" in kwargs:
            pcolor_opts = kwargs["pcolorOpts"]
            warnings.warn("pcolorOpts has been deprecated, please use pcolor_opts", DeprecationWarning)
        if "streamOpts" in kwargs:
            stream_opts = kwargs["streamOpts"]
            warnings.warn("streamOpts has been deprecated, please use stream_opts", DeprecationWarning)
        if "gridOpts" in kwargs:
            grid_opts = kwargs["gridOpts"]
            warnings.warn("gridOpts has been deprecated, please use grid_opts", DeprecationWarning)
        if "showIt" in kwargs:
            show_it = kwargs["showIt"]
            warnings.warn("showIt has been deprecated, please use show_it", DeprecationWarning)
        if "vType" in kwargs:
            v_type = kwargs["vType"]
            warnings.warn("vType has been deprecated, please use v_type", DeprecationWarning)

        if pcolor_opts is None:
            pcolor_opts = {}
        if stream_opts is None:
            stream_opts = {'color': 'k'}
        if grid_opts is None:
            grid_opts = {'color': 'k', 'alpha': 0.5}
        v_typeOpts = ['CC', 'N', 'F', 'E', 'Fx', 'Fy', 'Fz', 'E', 'Ex', 'Ey', 'Ez']
        viewOpts = ['real', 'imag', 'abs']
        normalOpts = ['X', 'Y', 'Z']
        if v_type not in v_typeOpts:
            raise ValueError(
                "v_type must be in ['{0!s}']".format("', '".join(v_typeOpts))
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
        if v_type is not 'CC':
            aveOp = 'ave' + v_type + '2CC'
            Av = getattr(self, aveOp)
            if v.size == Av.shape[1]:
                v = Av*v
            elif len(v_type) == 2:
                # was one of Fx, Fy, Fz, Ex, Ey, Ez
                # assuming v has all three components in these cases
                vec_ind = {'x': 0, 'y': 1, 'z': 2}[v_type[1]]
                if v_type[0] == 'E':
                    i_s = np.cumsum([0, self.nEx, self.nEy, self.nEz])
                elif v_type[0] == 'F':
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
            v2d, v_type='CC',
            grid=grid, view=view,
            ax=ax, clim=clim, show_it=False,
            pcolor_opts=pcolor_opts,
            grid_opts=grid_opts,
            range_x=range_x,
            range_y=range_y)

        ax.set_xlabel('y' if normal == 'X' else 'x')
        ax.set_ylabel('y' if normal == 'Z' else 'z')
        ax.set_title(
            'Slice {0:d}, {1!s} = {2:4.2f}'.format(ind, normal, slice_loc)
        )
        if show_it:
            plt.show()
        return tuple(out)

    def serialize(self):
        serial = BaseTensorMesh.serialize(self)
        inds, levels = self.__getstate__()
        serial['cell_indexes'] = inds.tolist()
        serial['cell_levels'] = levels.tolist()
        return serial

    @classmethod
    def deserialize(cls, serial):
        mesh = cls(**serial)
        return mesh

    def __reduce__(self):
        return TreeMesh, (self.h, self.x0), self.__getstate__()
