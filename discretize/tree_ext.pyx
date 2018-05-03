# distutils: language=c++
cimport cython
cimport numpy as np
from libc.math cimport sqrt, abs, cbrt

from tree cimport int_t, Tree as c_Tree, PyWrapper, Node, Edge, Face, Cell as c_Cell

import scipy.sparse as sp
from scipy.spatial import Delaunay, cKDTree
from six import integer_types
import numpy as np

cdef class Cell:
    cdef double _x, _y, _z, _x0, _y0, _z0, _wx, _wy, _wz
    cdef int_t _dim
    cdef c_Cell* _cell
    cdef void _set(self, c_Cell* cell):
        self._cell = cell
        self._dim = cell.n_dim
        self._x = cell.location[0]
        self._x0 = cell.points[0].location[0]

        self._y = cell.location[1]
        self._y0 = cell.points[0].location[1]

        self._wx = 2*(self._x-self._x0)
        self._wy = 2*(self._y-self._y0)
        if(self._dim>2):
            self._z = cell.location[2]
            self._z0 = cell.points[0].location[2]

            self._wz = 2*(self._z-self._z0)

    @property
    def nodes(self):
        cdef c_Cell* cell = self._cell
        if self._dim>2:
            return tuple((cell.points[0].index, cell.points[1].index,
                          cell.points[2].index, cell.points[3].index,
                          cell.points[4].index, cell.points[5].index,
                          cell.points[6].index, cell.points[7].index))
        return tuple((cell.points[0].index, cell.points[1].index,
                      cell.points[2].index, cell.points[3].index))

    @property
    def center(self):
        if self._dim==2: return np.array(tuple((self._x, self._y)))
        return np.array(tuple((self._x, self._y, self._z)))

    @property
    def x0(self):
        if self._dim==2: return np.array(tuple((self._x0, self._y0)))
        return np.array(tuple((self._x0, self._y0, self._z0)))

    @property
    def h(self):
        if self._dim==2: return np.array(tuple((self._wx, self._wy)))
        return np.array(tuple((self._wx, self._wy, self._wz)))

    @property
    def dim(self):
        return self._dim

    @property
    def index(self):
        return self._cell.index

    @property
    def _index_loc(self):
        if self._dim==2:
            return tuple((self._cell.location_ind[0], self._cell.location_ind[1]))
        return tuple((self._cell.location_ind[0], self._cell.location_ind[1],
                      self._cell.location_ind[2]))
    @property
    def _level(self):
        return self._cell.level

cdef int_t _evaluate_func(void* function, c_Cell* cell) with gil:
    func = <object> function
    pycell = Cell()
    pycell._set(cell)
    return <int_t> func(pycell)

cdef inline int sign(double val):
    return (0<val)-(val<0)

cdef class _TreeMesh:
    cdef c_Tree *tree
    cdef PyWrapper *wrapper
    cdef int_t _nx, _ny, _nz, max_level
    cdef double[3] _xc, _xf

    cdef double[:] _xs, _ys, _zs

    cdef object _gridCC, _gridN, _gridEx, _gridEy
    cdef object _gridhN, _gridhEx, _gridhEy
    cdef object _h_gridded
    cdef object _vol, _area, _edge
    cdef object _aveFx2CC, _aveFy2CC, _aveFz2CC, _aveF2CC, _aveF2CCV,
    cdef object _aveN2CC
    cdef object _aveEx2CC, _aveEy2CC, _aveEz2CC,_aveE2CC,_aveE2CCV
    cdef object _faceDiv
    cdef object _edgeCurl, _nodalGrad

    cdef object __ubc_order, __ubc_indArr

    def __cinit__(self, *args, **kwargs):
        self.wrapper = new PyWrapper()
        self.tree = new c_Tree()

    def __init__(self, max_level, x0, h):
        self.max_level = max_level
        self._nx = 2<<max_level
        self._ny = 2<<max_level

        xs = np.empty(self._nx+1, dtype=float)
        xs[::2] = np.cumsum(np.r_[x0[0],h[0]])
        xs[1::2] = (xs[:-1:2]+xs[2::2])/2
        self._xs = xs

        ys = np.empty(self._ny+1, dtype=float)
        ys[::2] = np.cumsum(np.r_[x0[1],h[1]])
        ys[1::2] = (ys[:-1:2]+ys[2::2])/2
        self._ys = ys

        if self.dim>2:
            self._nz = 2<<max_level

            zs = np.empty(self._nz+1, dtype=float)
            zs[::2] = np.cumsum(np.r_[x0[2],h[2]])
            zs[1::2] = (zs[:-1:2]+zs[2::2])/2
            self._zs = zs
        else:
            self._zs = np.zeros(1, dtype=float)

        self.tree.set_dimension(self.dim)
        self.tree.set_level(self.max_level)
        self.tree.set_xs(&self._xs[0], &self._ys[0], &self._zs[0])

        self._gridCC = None
        self._gridN = None
        self._gridhN = None
        self._h_gridded = None

        self._gridEx = None
        self._gridEy = None
        self._gridEz = None
        self._gridhEx = None
        self._gridhEy = None
        self._gridhEz = None

        self._gridFx = None
        self._gridFy = None
        self._gridFz = None
        self._gridhFx = None
        self._gridhFy = None
        self._gridhFz = None

        self._vol = None
        self._area = None
        self._edge = None

        self._aveFx2CC = None
        self._aveFy2CC = None
        self._aveFz2CC = None
        self._aveF2CC = None
        self._aveF2CCV = None

        self._aveEx2CC = None
        self._aveEy2CC = None
        self._aveEz2CC = None
        self._aveE2CC = None
        self._aveE2CCV = None

        self._aveN2CC = None

        self._faceDiv = None
        self._nodalGrad = None
        self._edgeCurl = None

        self.__ubc_order = None
        self.__ubc_indArr = None

    def refine(self, function, **kwargs):
        if type(function) in integer_types:
            level = function
            function = lambda cell: level

        #Wrapping function so it can be called in c++
        cdef void * self_ptr
        cdef void * func_ptr;
        func_ptr = <void *> function;
        self.wrapper.set(func_ptr, _evaluate_func)

        #Then tell c++ to build the tree
        self.tree.build_tree_from_function(self.wrapper)
        self.number()

    def _insert_cells(self, double[:, :] cells, long[:] levels):
        cdef int_t i
        for i in range(levels.shape[0]):
            self.tree.insert_cell(&cells[i, 0], levels[i])
        self.tree.finalize_lists()

    def _get_xs(self):
        return np.array(self._xs), np.array(self._ys), np.array(self._zs)

    def number(self):
        self.tree.number()

    @property
    def xC(self):
        return self._xc

    @property
    def fill(self):
        """
        How filled is the mesh compared to a TensorMesh?
        As a fraction: [0, 1].
        """
        return float(self.nC)/((2**self.maxLevel)**self.dim)

    @property
    def maxLevel(self):
        """
        The maximum level used, which may be
        less than `levels`.
        """
        cdef int level = 0
        for cell in self.tree.cells:
            level = max(level, cell.level)
        return level

    @property
    def nC(self):
        return self.tree.cells.size()

    @property
    def nN(self):
        return self.ntN-self.nhN

    @property
    def ntN(self):
        return self.tree.nodes.size()

    @property
    def nhN(self):
        return self.tree.hanging_nodes.size()

    @property
    def nE(self):
        return self.nEx+self.nEy+self.nEz

    @property
    def nhE(self):
        return self.nhEx+self.nhEy+self.nhEz

    @property
    def ntE(self):
        return self.nE+self.nhE

    @property
    def nEx(self):
        return self.ntEx-self.nhEx

    @property
    def nEy(self):
        return self.ntEy-self.nhEy

    @property
    def nEz(self):
        return self.ntEz-self.nhEz

    @property
    def ntEx(self):
        return self.tree.edges_x.size()

    @property
    def ntEy(self):
        return self.tree.edges_y.size()

    @property
    def ntEz(self):
        return self.tree.edges_z.size()

    @property
    def nhEx(self):
        return self.tree.hanging_edges_x.size()

    @property
    def nhEy(self):
        return self.tree.hanging_edges_y.size()

    @property
    def nhEz(self):
        return self.tree.hanging_edges_z.size()

    @property
    def nF(self):
        return self.nFx+self.nFy+self.nFz

    @property
    def nhF(self):
        return self.nhFx+self.nhFy+self.nhFz

    @property
    def ntF(self):
        return self.nF+self.nhF

    @property
    def nFx(self):
        return self.ntFx-self.nhFx

    @property
    def nFy(self):
        return self.ntFy-self.nhFy

    @property
    def nFz(self):
        return self.ntFz-self.nhFz

    @property
    def ntFx(self):
        if(self.dim==2): return self.ntEy
        return self.tree.faces_x.size()

    @property
    def ntFy(self):
        if(self.dim==2): return self.ntEx
        return self.tree.faces_y.size()

    @property
    def ntFz(self):
        if(self.dim==2): return 0
        return self.tree.faces_z.size()

    @property
    def nhFx(self):
        if(self.dim==2): return self.nhEy
        return self.tree.hanging_faces_x.size()

    @property
    def nhFy(self):
        if(self.dim==2): return self.nhEx
        return self.tree.hanging_faces_y.size()

    @property
    def nhFz(self):
        if(self.dim==2): return 0
        return self.tree.hanging_faces_z.size()

    @property
    def gridCC(self):
        """
        Returns an M by N numpy array with the center locations of all cells
        in order. M is the number of cells and N=2,3 is the dimension of the
        mesh.
        """
        cdef np.float64_t[:, :] gridCC
        cdef np.int64_t ii, ind, dim
        if self._gridCC is None:
            dim = self.dim
            self._gridCC = np.empty((self.nC, dim), dtype=np.float64)
            gridCC = self._gridCC
            for cell in self.tree.cells:
                ind = cell.index
                for ii in range(dim):
                    gridCC[ind, ii] = cell.location[ii]
        return self._gridCC

    @property
    def gridN(self):
        """
        Returns an M by N numpy array with the widths of all cells in order.
        M is the number of nodes and N=2,3 is the dimension of the mesh.
        """
        cdef np.float64_t[:, :] gridN
        cdef Node *node
        cdef np.int64_t ii, ind, dim
        if self._gridN is None:
            dim = self.dim
            self._gridN = np.empty((self.nN, dim) ,dtype=np.float64)
            gridN = self._gridN
            for it in self.tree.nodes:
                node = it.second
                if not node.hanging:
                    ind = node.index
                    for ii in range(dim):
                        gridN[ind, ii] = node.location[ii]
        return self._gridN

    @property
    def gridhN(self):
        cdef np.float64_t[:, :] gridN
        cdef Node *node
        cdef np.int64_t ii, ind, dim
        if self._gridhN is None:
            dim = self.dim
            self._gridhN = np.empty((self.nhN, dim), dtype=np.float64)
            gridhN = self._gridhN
            for node in self.tree.hanging_nodes:
                ind = node.index-self.nN
                for ii in range(dim):
                    gridhN[ind, ii] = node.location[ii]
        return self._gridhN

    @property
    def h_gridded(self):
        """
        Returns an (nC, dim) numpy array with the widths of all cells in order
        """
        # TODO
        cdef np.float64_t[:, :] gridCH
        cdef np.int64_t ii, ind, dim
        cdef np.float64_t len
        if self._h_gridded is None:
            dim = self.dim
            self._h_gridded = np.empty((self.nC, dim), dtype=np.float64)
            h_gridded = self._h_gridded
            for cell in self:
                ind = cell.index
                for ii in range(dim):
                    h_gridded[ind, ii] = cell.h[ii]

        return self._h_gridded

    @property
    def gridEx(self):
        cdef np.float64_t[:, :] gridEx
        cdef Edge *edge
        cdef np.int64_t ii, ind, dim
        if self._gridEx is None:
            dim = self.dim
            self._gridEx = np.empty((self.nEx, dim), dtype=np.float64)
            gridEx = self._gridEx
            for it in self.tree.edges_x:
                edge = it.second
                if not edge.hanging:
                    ind = edge.index
                    for ii in range(dim):
                        gridEx[ind, ii] = edge.location[ii]
        return self._gridEx

    @property
    def gridhEx(self):
        cdef np.float64_t[:, :] gridhEx
        cdef Edge *edge
        cdef np.int64_t ii, ind, dim
        if self._gridhEx is None:
            dim = self.dim
            self._gridhEx = np.empty((self.nhEx, dim), dtype=np.float64)
            gridhEx = self._gridhEx
            for edge in self.tree.hanging_edges_x:
                ind = edge.index-self.nEx
                for ii in range(dim):
                    gridhEx[ind,ii] = edge.location[ii]
        return self._gridhEx

    @property
    def gridEy(self):
        cdef np.float64_t[:, :] gridEy
        cdef Edge *edge
        cdef np.int64_t ii, ind, dim
        if self._gridEy is None:
            dim = self.dim
            self._gridEy = np.empty((self.nEy, dim), dtype=np.float64)
            gridEy = self._gridEy
            for it in self.tree.edges_y:
                edge = it.second
                if not edge.hanging:
                    ind = edge.index
                    for ii in range(dim):
                        gridEy[ind,ii] = edge.location[ii]
        return self._gridEy

    @property
    def gridhEy(self):
        cdef np.float64_t[:,:] gridhEy
        cdef Edge *edge
        cdef np.int64_t ii, ind, dim
        if self._gridhEy is None:
            dim = self.dim
            self._gridhEy = np.empty((self.nhEy, dim), dtype=np.float64)
            gridhEy = self._gridhEy
            for edge in self.tree.hanging_edges_y:
                ind = edge.index-self.nEy
                for ii in range(dim):
                    gridhEy[ind, ii] = edge.location[ii]
        return self._gridhEy

    @property
    def gridEz(self):
        cdef np.float64_t[:, :] gridEz
        cdef Edge *edge
        cdef np.int64_t ii, ind, dim
        if self._gridEz is None:
            dim = self.dim
            self._gridEz = np.empty((self.nEz, dim), dtype=np.float64)
            gridEz = self._gridEz
            for it in self.tree.edges_z:
                edge = it.second
                if not edge.hanging:
                    ind = edge.index
                    for ii in range(dim):
                        gridEz[ind,ii] = edge.location[ii]
        return self._gridEz

    @property
    def gridhEz(self):
        cdef np.float64_t[:,:] gridhEz
        cdef Edge *edge
        cdef np.int64_t ii, ind, dim
        if self._gridhEz is None:
            dim = self.dim
            self._gridhEz = np.empty((self.nhEz, dim), dtype=np.float64)
            gridhEz = self._gridhEz
            for edge in self.tree.hanging_edges_z:
                ind = edge.index-self.nEz
                for ii in range(dim):
                    gridhEz[ind, ii] = edge.location[ii]
        return self._gridhEz

    @property
    def gridFx(self):
        if(self.dim==2): return self.gridEy

        cdef np.float64_t[:,:] gridFx
        cdef Face *face
        cdef np.int64_t ii, ind, dim
        if self._gridFx is None:
            dim = self.dim
            self._gridFx = np.empty((self.nFx, dim), dtype=np.float64)
            gridFx = self._gridFx
            for it in self.tree.faces_x:
                face = it.second
                if not face.hanging:
                    ind = face.index
                    for ii in range(dim):
                        gridFx[ind, ii] = face.location[ii]
        return self._gridFx

    @property
    def gridFy(self):
        if(self.dim==2): return self.gridEx
        cdef np.float64_t[:,:] gridFy
        cdef Face *face
        cdef np.int64_t ii, ind, dim
        if self._gridFy is None:
            dim = self.dim
            self._gridFy = np.empty((self.nFy, dim), dtype=np.float64)
            gridFy = self._gridFy
            for it in self.tree.faces_y:
                face = it.second
                if not face.hanging:
                    ind = face.index
                    for ii in range(dim):
                        gridFy[ind, ii] = face.location[ii]
        return self._gridFy

    @property
    def gridFz(self):
        if(self.dim==2): return self.gridCC

        cdef np.float64_t[:,:] gridFz
        cdef Face *face
        cdef np.int64_t ii, ind, dim
        if self._gridFz is None:
            dim = self.dim
            self._gridFz = np.empty((self.nFz, dim), dtype=np.float64)
            gridFz = self._gridFz
            for it in self.tree.faces_z:
                face = it.second
                if not face.hanging:
                    ind = face.index
                    for ii in range(dim):
                        gridFz[ind, ii] = face.location[ii]
        return self._gridFz

    @property
    def gridhFx(self):
        if(self.dim==2): return self.gridhEy

        cdef np.float64_t[:,:] gridFx
        cdef Face *face
        cdef np.int64_t ii, ind, dim
        if self._gridhFx is None:
            dim = self.dim
            self._gridhFx = np.empty((self.nhFx, dim), dtype=np.float64)
            gridhFx = self._gridhFx
            for face in self.tree.hanging_faces_x:
                ind = face.index-self.nFx
                for ii in range(dim):
                    gridhFx[ind, ii] = face.location[ii]
        return self._gridhFx

    @property
    def gridhFy(self):
        if(self.dim==2): return self.gridhEx

        cdef np.float64_t[:,:] gridhFy
        cdef Face *face
        cdef np.int64_t ii, ind, dim
        if self._gridhFy is None:
            dim = self.dim
            self._gridhFy = np.empty((self.nhFy, dim), dtype=np.float64)
            gridhFy = self._gridhFy
            for face in self.tree.hanging_faces_y:
                ind = face.index-self.nFy
                for ii in range(dim):
                    gridhFy[ind, ii] = face.location[ii]
        return self._gridhFy

    @property
    def gridhFz(self):
        if(self.dim==2): return np.array([])

        cdef np.float64_t[:,:] gridhFz
        cdef Face *face
        cdef np.int64_t ii, ind, dim
        if self._gridhFz is None:
            dim = self.dim
            self._gridhFz = np.empty((self.nhFz, dim), dtype=np.float64)
            gridhFz = self._gridhFz
            for face in self.tree.hanging_faces_z:
                ind = face.index-self.nFz
                for ii in range(dim):
                    gridhFz[ind, ii] = face.location[ii]
        return self._gridhFz

    @property
    def vol(self):
        cdef np.float64_t[:] vol
        if self._vol is None:
            self._vol = np.empty(self.nC, dtype=np.float64)
            vol = self._vol
            for cell in self.tree.cells:
                vol[cell.index] = cell.volume
        return self._vol

    @property
    def area(self):
        if self.dim == 2 and self._area is None:
            self._area = np.r_[self.edge[self.nEx:], self.edge[:self.nEx]]
        cdef np.float64_t[:] area
        cdef int_t ind, offset = 0
        cdef Face *face
        if self._area is None:
            self._area = np.empty(self.nF, dtype=np.float64)
            area = self._area

            for it in self.tree.faces_x:
                face = it.second
                if face.hanging: continue
                area[face.index] = face.area

            offset = self.nFx
            for it in self.tree.faces_y:
                face = it.second
                if face.hanging: continue
                area[face.index+offset] = face.area

            offset = self.nFx + self.nFy
            for it in self.tree.faces_z:
                face = it.second
                if face.hanging: continue
                area[face.index+offset] = face.area
        return self._area

    @property
    def edge(self):
        cdef np.float64_t[:] edge_l
        cdef Edge *edge
        cdef int_t ind, offset
        if self._edge is None:
            self._edge = np.empty(self.nE, dtype=np.float64)
            edge_l = self._edge

            for it in self.tree.edges_x:
                edge = it.second
                if edge.hanging: continue
                edge_l[edge.index] = edge.length

            offset = self.nEx
            for it in self.tree.edges_y:
                edge = it.second
                if edge.hanging: continue
                edge_l[edge.index+offset] = edge.length

            if self.dim>2:
                offset = self.nEx + self.nEy
                for it in self.tree.edges_z:
                    edge = it.second
                    if edge.hanging: continue
                    edge_l[edge.index+offset] = edge.length
        return self._edge

    @property
    def faceDiv(self):
        if(self._faceDiv is not None):
            return self._faceDiv
        if(self.dim==2):
            D = self._faceDiv2D() # Because it uses edges instead of faces
        else:
            D = self._faceDiv3D()
        R = self._deflate_faces()
        self._faceDiv = D*R
        return self._faceDiv

    @cython.cdivision(True)
    @cython.boundscheck(False)
    def _faceDiv2D(self):
        cdef np.int64_t[:] I = np.empty(self.nC*4, dtype=np.int64)
        cdef np.int64_t[:] J = np.empty(self.nC*4, dtype=np.int64)
        cdef np.float64_t[:] V = np.empty(self.nC*4, dtype=np.float64)

        cdef np.int64_t i = 0
        cdef Edge *edges[4]
        cdef np.int64_t offset = self.tree.edges_y.size()
        cdef double volume

        for cell in self.tree.cells:
            edges = cell.edges
            i = cell.index
            I[i*4:i*4+4] = i
            J[i*4  ] = edges[0].index+offset #x edge, y face (add offset)
            J[i*4+1] = edges[1].index+offset #x edge, y face (add offset)
            J[i*4+2] = edges[2].index #y edge, x face
            J[i*4+3] = edges[3].index #y edge, x face

            volume = cell.volume
            V[i*4  ] = -(edges[0].length/volume)
            V[i*4+1] = edges[1].length/volume
            V[i*4+2] = -(edges[2].length/volume)
            V[i*4+3] = edges[3].length/volume
        return sp.csr_matrix((V, (I, J)))

    @cython.cdivision(True)
    @cython.boundscheck(False)
    def _faceDiv3D(self):
        cdef:
            np.int64_t[:] I = np.empty(self.nC*6, dtype=np.int64)
            np.int64_t[:] J = np.empty(self.nC*6, dtype=np.int64)
            np.float64_t[:] V = np.empty(self.nC*6, dtype=np.float64)

            np.int64_t i = 0
            Face *faces[6]
            np.int64_t offset1 = self.tree.faces_x.size()
            np.int64_t offset2 = offset1+self.tree.faces_y.size()
            double volume, fx_area, fy_area, fz_area

        for cell in self.tree.cells:
            faces = cell.faces
            i = cell.index
            I[i*6:i*6+6] = i
            J[i*6  ] = faces[0].index #x1 face
            J[i*6+1] = faces[1].index #x2 face
            J[i*6+2] = faces[2].index+offset1 #y face (add offset1)
            J[i*6+3] = faces[3].index+offset1 #y face (add offset1)
            J[i*6+4] = faces[4].index+offset2 #z face (add offset2)
            J[i*6+5] = faces[5].index+offset2 #z face (add offset2)

            volume = cell.volume
            fx_area = faces[0].area
            fy_area = faces[2].area
            fz_area = faces[4].area
            V[i*6  ] = -(fx_area)/volume
            V[i*6+1] =  (fx_area)/volume
            V[i*6+2] = -(fy_area)/volume
            V[i*6+3] =  (fy_area)/volume
            V[i*6+4] = -(fz_area)/volume
            V[i*6+5] =  (fz_area)/volume
        return sp.csr_matrix((V, (I, J)))

    @property
    @cython.cdivision(True)
    @cython.boundscheck(False)
    def edgeCurl(self):
        if self._edgeCurl is not None:
            return self._edgeCurl
        cdef:
            int_t dim = self.dim
            np.int64_t[:] I = np.empty(4*self.nF, dtype=np.int64)
            np.int64_t[:] J = np.empty(4*self.nF, dtype=np.int64)
            np.float64_t[:] V = np.empty(4*self.nF, dtype=np.float64)
            Face *face
            int_t ii
            int_t face_offset_y = self.nFx
            int_t face_offset_z = self.nFx+self.nFy
            int_t edge_offset_y = self.ntEx
            int_t edge_offset_z = self.ntEx+self.ntEy
            double area

        for it in self.tree.faces_x:
            face = it.second
            if face.hanging:
                continue
            ii = face.index
            I[4*ii:4*ii+4] = ii
            J[4*ii  ] = face.edges[0].index+edge_offset_z
            J[4*ii+1] = face.edges[1].index+edge_offset_y
            J[4*ii+2] = face.edges[2].index+edge_offset_z
            J[4*ii+3] = face.edges[3].index+edge_offset_y

            area = face.area
            V[4*ii  ] = -(face.edges[0].length/area)
            V[4*ii+1] = -(face.edges[1].length/area)
            V[4*ii+2] = (face.edges[2].length/area)
            V[4*ii+3] = (face.edges[3].length/area)

        for it in self.tree.faces_y:
            face = it.second
            if face.hanging:
                continue
            ii = face.index+face_offset_y
            I[4*ii:4*ii+4] = ii
            J[4*ii  ] = face.edges[0].index+edge_offset_z
            J[4*ii+1] = face.edges[1].index
            J[4*ii+2] = face.edges[2].index+edge_offset_z
            J[4*ii+3] = face.edges[3].index

            area = face.area
            V[4*ii  ] = (face.edges[0].length/area)
            V[4*ii+1] = (face.edges[1].length/area)
            V[4*ii+2] = -(face.edges[2].length/area)
            V[4*ii+3] = -(face.edges[3].length/area)

        for it in self.tree.faces_z:
            face = it.second
            if face.hanging:
                continue
            ii = face.index+face_offset_z
            I[4*ii:4*ii+4] = ii
            J[4*ii  ] = face.edges[0].index+edge_offset_y
            J[4*ii+1] = face.edges[1].index
            J[4*ii+2] = face.edges[2].index+edge_offset_y
            J[4*ii+3] = face.edges[3].index

            area = face.area
            V[4*ii  ] = -(face.edges[0].length/area)
            V[4*ii+1] = -(face.edges[1].length/area)
            V[4*ii+2] = (face.edges[2].length/area)
            V[4*ii+3] = (face.edges[3].length/area)

        C = sp.csr_matrix((V, (I, J)),shape=(self.nF, self.ntE))
        R = self._deflate_edges()
        self._edgeCurl = C*R
        return self._edgeCurl

    @property
    @cython.cdivision(True)
    @cython.boundscheck(False)
    def nodalGrad(self):
        if self._nodalGrad is not None:
            return self._nodalGrad
        cdef:
            int_t dim = self.dim
            np.int64_t[:] I = np.empty(2*self.nE, dtype=np.int64)
            np.int64_t[:] J = np.empty(2*self.nE, dtype=np.int64)
            np.float64_t[:] V = np.empty(2*self.nE, dtype=np.float64)
            Edge *edge
            double length
            int_t ii
            np.int64_t offset1 = self.nEx
            np.int64_t offset2 = offset1+self.nEy

        for it in self.tree.edges_x:
            edge = it.second
            if edge.hanging: continue
            ii = edge.index
            I[ii*2:ii*2+2] = ii
            J[ii*2  ] = edge.points[0].index
            J[ii*2+1] = edge.points[1].index

            length = edge.length
            V[ii*2  ] = -1.0/length
            V[ii*2+1] = 1.0/length

        for it in self.tree.edges_y:
            edge = it.second
            if edge.hanging: continue
            ii = edge.index+offset1
            I[ii*2:ii*2+2] = ii
            J[ii*2] = edge.points[0].index
            J[ii*2+1] = edge.points[1].index

            length = edge.length
            V[ii*2  ] = -1.0/length
            V[ii*2+1] = 1.0/length

        if(dim>2):
            for it in self.tree.edges_z:
                edge = it.second
                if edge.hanging: continue
                ii = edge.index+offset2
                I[ii*2:ii*2+2] = ii
                J[ii*2  ] = edge.points[0].index
                J[ii*2+1] = edge.points[1].index

                length = edge.length
                V[ii*2  ] = -1.0/length
                V[ii*2+1] = 1.0/length


        Rn = self._deflate_nodes()
        G = sp.csr_matrix((V, (I, J)), shape=(self.nE, self.ntN))
        self._nodalGrad = G*Rn
        return self._nodalGrad

    @cython.boundscheck(False)
    def _cellGradxStencil(self):
        cdef np.int64_t[:] I = np.zeros(2*self.ntFx, dtype=np.int64)
        cdef np.int64_t[:] J = np.zeros(2*self.ntFx, dtype=np.int64)
        cdef np.float64_t[:] V = np.zeros(2*self.ntFx, dtype=np.float64)
        cdef int dim = self.dim
        cdef int_t ind

        for cell in self.tree.cells :
            next_cell = cell.neighbors[1]
            if next_cell==NULL:
                continue
            if dim==2:
                if next_cell.is_leaf():
                    ind = cell.edges[3].index
                    I[2*ind  ] = ind
                    I[2*ind+1] = ind
                    J[2*ind  ] = cell.index
                    J[2*ind+1] = next_cell.index
                    V[2*ind  ] = -1.0
                    V[2*ind+1] = 1.0
                else:
                    for i in range(2): # two neighbors in +x direction
                        ind = next_cell.children[2*i].edges[2].index
                        I[2*ind  ] = ind
                        I[2*ind+1] = ind
                        J[2*ind  ] = cell.index
                        J[2*ind+1] = next_cell.children[2*i].index
                        V[2*ind  ] = -1.0
                        V[2*ind+1] = 1.0
            else:
                if cell.neighbors[1].is_leaf():
                    ind = cell.faces[1].index
                    I[2*ind  ] = ind
                    I[2*ind+1] = ind
                    J[2*ind  ] = cell.index
                    J[2*ind+1] = next_cell.index
                    V[2*ind  ] = -1.0
                    V[2*ind+1] = 1.0
                else:
                    for i in range(4): # four neighbors in +x direction
                        ind = next_cell.children[2*i].faces[0].index
                        I[2*ind  ] = ind
                        I[2*ind+1] = ind
                        J[2*ind  ] = cell.index
                        J[2*ind+1] = next_cell.children[2*i].index
                        V[2*ind  ] = -1.0
                        V[2*ind+1] = 1.0

        return sp.csr_matrix((V, (I,J)), shape=(self.ntFx, self.nC))

    @cython.boundscheck(False)
    def _cellGradyStencil(self):
        cdef np.int64_t[:] I = np.zeros(2*self.ntFy, dtype=np.int64)
        cdef np.int64_t[:] J = np.zeros(2*self.ntFy, dtype=np.int64)
        cdef np.float64_t[:] V = np.zeros(2*self.ntFy, dtype=np.float64)
        cdef int dim = self.dim
        cdef int_t ind

        for cell in self.tree.cells :
            next_cell = cell.neighbors[3]
            if next_cell==NULL:
                continue
            if dim==2:
                if next_cell.is_leaf():
                    ind = cell.edges[1].index
                    I[2*ind  ] = ind
                    I[2*ind+1] = ind
                    J[2*ind  ] = cell.index
                    J[2*ind+1] = next_cell.index
                    V[2*ind  ] = -1.0
                    V[2*ind+1] = 1.0
                else:
                    for i in range(2): # two neighbors in +y direction
                        ind = next_cell.children[i].edges[0].index
                        I[2*ind  ] = ind
                        I[2*ind+1] = ind
                        J[2*ind  ] = cell.index
                        J[2*ind+1] = next_cell.children[i].index
                        V[2*ind  ] = -1.0
                        V[2*ind+1] = 1.0
            else:
                if next_cell.is_leaf():
                    ind = cell.faces[3].index
                    I[2*ind  ] = ind
                    I[2*ind+1] = ind
                    J[2*ind  ] = cell.index
                    J[2*ind+1] = next_cell.index
                    V[2*ind  ] = -1.0
                    V[2*ind+1] = 1.0
                else:
                    for i in range(4): # four neighbors in +x direction
                        ind = next_cell.children[i].faces[2].index
                        I[2*ind  ] = ind
                        I[2*ind+1] = ind
                        J[2*ind  ] = cell.index
                        J[2*ind+1] = next_cell.children[i].index
                        V[2*ind  ] = -1.0
                        V[2*ind+1] = 1.0

        return sp.csr_matrix((V, (I,J)), shape=(self.ntFy, self.nC))

    @cython.boundscheck(False)
    def _cellGradzStencil(self):
        cdef np.int64_t[:] I = np.zeros(2*self.ntFz, dtype=np.int64)
        cdef np.int64_t[:] J = np.zeros(2*self.ntFz, dtype=np.int64)
        cdef np.float64_t[:] V = np.zeros(2*self.ntFz, dtype=np.float64)
        cdef int_t ind

        for cell in self.tree.cells :
            next_cell = cell.neighbors[5]
            if next_cell==NULL:
                continue
            if next_cell.is_leaf():
                ind = cell.faces[5].index
                I[2*ind  ] = ind
                I[2*ind+1] = ind
                J[2*ind  ] = cell.index
                J[2*ind+1] = next_cell.index
                V[2*ind  ] = -1.0
                V[2*ind+1] = 1.0
            else:
                for i in range(4): # four neighbors in +z direction
                    ind = next_cell.children[i].faces[4].index
                    I[2*ind  ] = ind
                    I[2*ind+1] = ind
                    J[2*ind  ] = cell.index
                    J[2*ind+1] = next_cell.children[i].index
                    V[2*ind  ] = -1.0
                    V[2*ind+1] = 1.0

        return sp.csr_matrix((V, (I,J)), shape=(self.ntFz, self.nC))

    @cython.boundscheck(False)
    def _deflate_edges_x(self):
        #I is output index (with hanging)
        #J is input index (without hanging)
        cdef np.int64_t[:] I = np.empty(2*self.ntEx, dtype=np.int64)
        cdef np.int64_t[:] J = np.empty(2*self.ntEx, dtype=np.int64)
        cdef np.float64_t[:] V = np.empty(2*self.ntEx, dtype=np.float64)
        cdef Edge *edge
        cdef np.int64_t ii
        #x edges:
        for it in self.tree.edges_x:
            edge = it.second
            ii = edge.index
            I[2*ii  ] = ii
            I[2*ii+1] = ii
            if edge.hanging:
                J[2*ii  ] = edge.parents[0].index
                J[2*ii+1] = edge.parents[1].index
            else:
                J[2*ii  ] = ii
                J[2*ii+1] = ii
            V[2*ii  ] = 0.5
            V[2*ii+1] = 0.5
        Rh = sp.csr_matrix((V, (I, J)), shape=(self.ntEx, self.ntEx))
        # Test if it needs to be deflated again, (if any parents were also hanging)
        last_ind = max(np.nonzero(Rh.getnnz(0)>0)[0][-1], self.nEx)
        while(last_ind > self.nEx):
            Rh = Rh*Rh
            last_ind = max(np.nonzero(Rh.getnnz(0)>0)[0][-1], self.nEx)
        Rh = Rh[:,:last_ind]
        return Rh

    @cython.boundscheck(False)
    def _deflate_edges_y(self):
        #I is output index (with hanging)
        #J is input index (without hanging)
        cdef int_t dim = self.dim
        cdef np.int64_t[:] I = np.empty(2*self.ntEy, dtype=np.int64)
        cdef np.int64_t[:] J = np.empty(2*self.ntEy, dtype=np.int64)
        cdef np.float64_t[:] V = np.empty(2*self.ntEy, dtype=np.float64)
        cdef Edge *edge
        cdef np.int64_t ii
        #x edges:
        for it in self.tree.edges_y:
            edge = it.second
            ii = edge.index
            I[2*ii  ] = ii
            I[2*ii+1] = ii
            if edge.hanging:
                J[2*ii  ] = edge.parents[0].index
                J[2*ii+1] = edge.parents[1].index
            else:
                J[2*ii  ] = ii
                J[2*ii+1] = ii
            V[2*ii  ] = 0.5
            V[2*ii+1] = 0.5
        Rh = sp.csr_matrix((V, (I, J)), shape=(self.ntEy, self.ntEy))
        # Test if it needs to be deflated again, (if any parents were also hanging)
        last_ind = max(np.nonzero(Rh.getnnz(0)>0)[0][-1], self.nEy)
        while(last_ind > self.nEy):
            Rh = Rh*Rh
            last_ind = max(np.nonzero(Rh.getnnz(0)>0)[0][-1], self.nEy)
        Rh = Rh[:,:last_ind]
        return Rh

    @cython.boundscheck(False)
    def _deflate_edges_z(self):
        #I is output index (with hanging)
        #J is input index (without hanging)
        cdef int_t dim = self.dim
        cdef np.int64_t[:] I = np.empty(2*self.ntEz, dtype=np.int64)
        cdef np.int64_t[:] J = np.empty(2*self.ntEz, dtype=np.int64)
        cdef np.float64_t[:] V = np.empty(2*self.ntEz, dtype=np.float64)
        cdef Edge *edge
        cdef np.int64_t ii
        #x edges:
        for it in self.tree.edges_z:
            edge = it.second
            ii = edge.index
            I[2*ii  ] = ii
            I[2*ii+1] = ii
            if edge.hanging:
                J[2*ii  ] = edge.parents[0].index
                J[2*ii+1] = edge.parents[1].index
            else:
                J[2*ii  ] = ii
                J[2*ii+1] = ii
            V[2*ii  ] = 0.5
            V[2*ii+1] = 0.5
        Rh = sp.csr_matrix((V, (I, J)), shape=(self.ntEz, self.ntEz))
        # Test if it needs to be deflated again, (if any parents were also hanging)
        last_ind = max(np.nonzero(Rh.getnnz(0)>0)[0][-1], self.nEz)
        while(last_ind > self.nEz):
            Rh = Rh*Rh
            last_ind = max(np.nonzero(Rh.getnnz(0)>0)[0][-1], self.nEz)
        Rh = Rh[:,:last_ind]
        return Rh

    def _deflate_edges(self):
        Rx = self._deflate_edges_x()
        Ry = self._deflate_edges_y()
        Rz = self._deflate_edges_z()
        return sp.block_diag((Rx, Ry, Rz))

    def _deflate_faces(self):
        if(self.dim==2):
            Rx = self._deflate_edges_x()
            Ry = self._deflate_edges_y()
            return sp.block_diag((Ry, Rx))
        else:
            Rx = self._deflate_faces_x()
            Ry = self._deflate_faces_y()
            Rz = self._deflate_faces_z()
            return sp.block_diag((Rx, Ry, Rz))

    @cython.boundscheck(False)
    def _deflate_faces_x(self):
        #I is output index (with hanging)
        #J is input index (without hanging)
        cdef np.int64_t[:] I = np.empty(self.ntFx, dtype=np.int64)
        cdef np.int64_t[:] J = np.empty(self.ntFx, dtype=np.int64)
        cdef np.float64_t[:] V = np.empty(self.ntFx, dtype=np.float64)
        cdef Face *face
        cdef np.int64_t ii;

        for it in self.tree.faces_x:
            face = it.second
            ii = face.index
            I[ii] = ii
            if face.hanging:
                J[ii] = face.parent.index
            else:
                J[ii] = ii
            V[ii] = 1.0
        return sp.csr_matrix((V, (I, J)))

    @cython.boundscheck(False)
    def _deflate_faces_y(self):
        #I is output index (with hanging)
        #J is input index (without hanging)
        cdef np.int64_t[:] I = np.empty(self.ntFy, dtype=np.int64)
        cdef np.int64_t[:] J = np.empty(self.ntFy, dtype=np.int64)
        cdef np.float64_t[:] V = np.empty(self.ntFy, dtype=np.float64)
        cdef Face *face
        cdef np.int64_t ii;

        for it in self.tree.faces_y:
            face = it.second
            ii = face.index
            I[ii] = ii
            if face.hanging:
                J[ii] = face.parent.index
            else:
                J[ii] = ii
            V[ii] = 1.0
        return sp.csr_matrix((V, (I, J)))

    @cython.boundscheck(False)
    def _deflate_faces_z(self):
        #I is output index (with hanging)
        #J is input index (without hanging)
        cdef np.int64_t[:] I = np.empty(self.ntFz, dtype=np.int64)
        cdef np.int64_t[:] J = np.empty(self.ntFz, dtype=np.int64)
        cdef np.float64_t[:] V = np.empty(self.ntFz, dtype=np.float64)
        cdef Face *face
        cdef np.int64_t ii;

        for it in self.tree.faces_z:
            face = it.second
            ii = face.index
            I[ii] = ii
            if face.hanging:
                J[ii] = face.parent.index
            else:
                J[ii] = ii
            V[ii] = 1.0
        return sp.csr_matrix((V, (I, J)))

    @cython.boundscheck(False)
    def _deflate_nodes(self):
        cdef np.int64_t[:] I = np.empty(4*self.ntN, dtype=np.int64)
        cdef np.int64_t[:] J = np.empty(4*self.ntN, dtype=np.int64)
        cdef np.float64_t[:] V = np.empty(4*self.ntN, dtype=np.float64)

        # I is output index
        # J is input index
        cdef Node *node
        cdef np.int64_t ii, i, offset
        offset = self.nN
        cdef double[4] weights

        for it in self.tree.nodes:
            node = it.second
            ii = node.index
            I[4*ii:4*ii+4] = ii
            if node.hanging:
                J[4*ii  ] = node.parents[0].index
                J[4*ii+1] = node.parents[1].index
                J[4*ii+2] = node.parents[2].index
                J[4*ii+3] = node.parents[3].index
            else:
                J[4*ii:4*ii+4] = ii
            V[4*ii:4*ii+4] = 0.25;

        Rh = sp.csr_matrix((V, (I, J)), shape=(self.ntN, self.ntN))
        # Test if it needs to be deflated again, (if any parents were also hanging)
        last_ind = max(np.nonzero(Rh.getnnz(0)>0)[0][-1], self.nN)
        while(last_ind > self.nN):
            Rh = Rh*Rh
            last_ind = max(np.nonzero(Rh.getnnz(0)>0)[0][-1], self.nN)
        Rh = Rh[:,:last_ind]
        return Rh

    @property
    @cython.boundscheck(False)
    def aveEx2CC(self):
        if self._aveEx2CC is not None:
            return self._aveEx2CC
        cdef np.int64_t[:] I,J
        cdef np.float64_t[:] V
        cdef np.int64_t ind, ii, n_epc
        cdef double scale

        n_epc = 2*(self.dim-1)
        I = np.empty(self.nC*n_epc, dtype=np.int64)
        J = np.empty(self.nC*n_epc, dtype=np.int64)
        V = np.empty(self.nC*n_epc, dtype=np.float64)
        scale = 1.0/n_epc
        for cell in self.tree.cells:
            ind = cell.index
            for ii in range(n_epc):
                I[ind*n_epc+ii] = ind
                J[ind*n_epc+ii] = cell.edges[ii].index
                V[ind*n_epc+ii] = scale

        Rex = self._deflate_edges_x()
        self._aveEx2CC = sp.csr_matrix((V, (I, J)))*Rex
        return self._aveEx2CC

    @property
    @cython.boundscheck(False)
    def aveEy2CC(self):
        if self._aveEy2CC is not None:
            return self._aveEy2CC
        cdef np.int64_t[:] I,J
        cdef np.float64_t[:] V
        cdef np.int64_t ind, ii, n_epc
        cdef double scale

        n_epc = 2*(self.dim-1)
        I = np.empty(self.nC*n_epc, dtype=np.int64)
        J = np.empty(self.nC*n_epc, dtype=np.int64)
        V = np.empty(self.nC*n_epc, dtype=np.float64)
        scale = 1.0/n_epc
        for cell in self.tree.cells:
            ind = cell.index
            for ii in range(n_epc):
                I[ind*n_epc+ii] = ind
                J[ind*n_epc+ii] = cell.edges[n_epc+ii].index #y edges
                V[ind*n_epc+ii] = scale

        Rey = self._deflate_edges_y()
        self._aveEy2CC = sp.csr_matrix((V, (I, J)))*Rey
        return self._aveEy2CC

    @property
    @cython.boundscheck(False)
    def aveEz2CC(self):
        if self._aveEz2CC is not None:
            return self._aveEz2CC
        if self.dim == 2:
            raise Exception('There are no z-edges in 2D')
        cdef np.int64_t[:] I,J
        cdef np.float64_t[:] V
        cdef np.int64_t ind, ii, n_epc
        cdef double scale

        n_epc = 2*(self.dim-1)
        I = np.empty(self.nC*n_epc, dtype=np.int64)
        J = np.empty(self.nC*n_epc, dtype=np.int64)
        V = np.empty(self.nC*n_epc, dtype=np.float64)
        scale = 1.0/n_epc
        for cell in self.tree.cells:
            ind = cell.index
            for ii in range(n_epc):
                I[ind*n_epc+ii] = ind
                J[ind*n_epc+ii] = cell.edges[ii+2*n_epc].index
                V[ind*n_epc+ii] = scale

        Rez = self._deflate_edges_z()
        self._aveEz2CC = sp.csr_matrix((V, (I, J)))*Rez
        return self._aveEz2CC

    @property
    def aveE2CC(self):
        if self._aveE2CC is None:
            stacks = [self.aveEx2CC, self.aveEy2CC]
            if self.dim==3:
                stacks += [self.aveEz2CC]
            self._aveE2CC = 1.0/self.dim * sp.hstack(stacks).tocsr()
        return self._aveE2CC

    @property
    def aveE2CCV(self):
        if self._aveE2CCV is None:
            stacks = [self.aveEx2CC, self.aveEy2CC]
            if self.dim==3:
                stacks += [self.aveEz2CC]
            self._aveE2CCV = sp.block_diag(stacks).tocsr()
        return self._aveE2CCV

    @property
    @cython.boundscheck(False)
    def aveFx2CC(self):
        if self._aveFx2CC is not None:
            return self._aveFx2CC
        if self.dim == 2:
            return self.aveEy2CC

        cdef np.int64_t[:] I,J
        cdef np.float64_t[:] V
        cdef Face *face1
        cdef Face *face2
        cdef np.int64_t ii
        I = np.empty(self.nC*2, dtype=np.int64)
        J = np.empty(self.nC*2, dtype=np.int64)
        V = np.empty(self.nC*2, dtype=np.float64)
        for cell in self.tree.cells:
            face1 = cell.faces[0] #y edge/ x face
            face2 = cell.faces[1] #y edge/ x face
            ii = cell.index
            I[ii*2:ii*2+2] = ii
            J[ii*2] = face1.index
            J[ii*2+1] = face2.index
            V[ii*2:ii*2+2] = 0.5
        Rfx = self._deflate_faces_x()
        self._aveFx2CC = sp.csr_matrix((V, (I, J)))*Rfx
        return self._aveFx2CC

    @property
    @cython.boundscheck(False)
    def aveFy2CC(self):
        if self._aveFy2CC is not None:
            return self._aveFy2CC
        if self.dim == 2:
            return self.aveEx2CC

        cdef np.int64_t[:] I,J
        cdef np.float64_t[:] V
        cdef Face *face1
        cdef Face *face2
        cdef np.int64_t ii
        I = np.empty(self.nC*2, dtype=np.int64)
        J = np.empty(self.nC*2, dtype=np.int64)
        V = np.empty(self.nC*2, dtype=np.float64)
        for cell in self.tree.cells:
            face1 = cell.faces[2] #x edge/ y face
            face2 = cell.faces[3] #x edge/ y face
            ii = cell.index
            I[ii*2:ii*2+2] = ii
            J[ii*2] = face1.index
            J[ii*2+1] = face2.index
            V[ii*2:ii*2+2] = 0.5
        Rfy = self._deflate_faces_y()
        self._aveFy2CC = sp.csr_matrix((V, (I, J)))*Rfy
        return self._aveFy2CC

    @property
    @cython.boundscheck(False)
    def aveFz2CC(self):
        if self._aveFz2CC is not None:
            return self._aveFz2CC
        if self.dim == 2:
            raise Exception('There are no z-faces in 2D')
        cdef np.int64_t[:] I,J
        cdef np.float64_t[:] V
        cdef Face *face1
        cdef Face *face2
        cdef np.int64_t ii
        I = np.empty(self.nC*2, dtype=np.int64)
        J = np.empty(self.nC*2, dtype=np.int64)
        V = np.empty(self.nC*2, dtype=np.float64)
        for cell in self.tree.cells:
            face1 = cell.faces[4]
            face2 = cell.faces[5]
            ii = cell.index
            I[ii*2:ii*2+2] = ii
            J[ii*2] = face1.index
            J[ii*2+1] = face2.index
            V[ii*2:ii*2+2] = 0.5
        Rfy = self._deflate_faces_z()
        self._aveFz2CC = sp.csr_matrix((V, (I, J)))*Rfy
        return self._aveFz2CC

    @property
    def aveF2CC(self):
        "Construct the averaging operator on cell faces to cell centers."
        if self._aveF2CC is None:
            stacks = [self.aveFx2CC, self.aveFy2CC]
            if self.dim == 3:
                stacks += [self.aveFz2CC]

            self._aveF2CC = 1./self.dim*sp.hstack(stacks).tocsr()
        return self._aveF2CC

    @property
    def aveF2CCV(self):
        "Construct the averaging operator on cell faces to cell centers."
        if self._aveF2CCV is None:
            stacks = [self.aveFx2CC, self.aveFy2CC]
            if self.dim == 3:
                stacks += [self.aveFz2CC]

            self._aveF2CCV = sp.block_diag(stacks).tocsr()
        return self._aveF2CCV

    @property
    @cython.boundscheck(False)
    def aveN2CC(self):
        cdef np.int64_t[:] I, J
        cdef np.float64_t[:] V
        cdef np.int64_t ii, id, n_ppc
        cdef double scale
        if self._aveN2CC is None:
            n_ppc = 1<<self.dim
            scale = 1.0/n_ppc
            I = np.empty(self.nC*n_ppc, dtype=np.int64)
            J = np.empty(self.nC*n_ppc, dtype=np.int64)
            V = np.empty(self.nC*n_ppc, dtype=np.float64)
            for cell in self.tree.cells:
                ii = cell.index
                for id in range(n_ppc):
                    I[ii*n_ppc+id] = ii
                    J[ii*n_ppc+id] = cell.points[id].index
                    V[ii*n_ppc+id] = scale
            Rn = self._deflate_nodes()
            self._aveN2CC = sp.csr_matrix((V, (I, J)))*Rn
        return self._aveN2CC

    def _get_containing_cell_index(self, loc):
        cdef double x,y,z
        x = loc[0]
        y = loc[1]
        if self.dim==3:
            z = loc[2]
        else:
            z = 0
        return self.tree.containing_cell(x, y, z).index

    def _getFaceP(self, xFace, yFace, zFace):
        cdef int dim = self.dim
        cdef int_t ind, id

        cdef np.int64_t[:] I, J, J1, J2, J3
        cdef np.float64_t[:] V

        J1 = np.empty(self.nC, dtype=np.int64)
        J2 = np.empty(self.nC, dtype=np.int64)
        if dim==3:
            J3 = np.empty(self.nC, dtype=np.int64)

        cdef int[3] faces
        cdef np.int64_t[:] offsets = np.empty(self.dim, dtype=np.int64)
        faces[0] = (xFace == 'fXp')
        faces[1] = (yFace == 'fYp')
        if dim == 3:
            faces[2] = (zFace == 'fZp')

        if dim == 2:
            offsets[0] = 0
            offsets[1] = self.ntFx
        else:
            offsets[0] = 0
            offsets[1] = self.ntFx
            offsets[2] = self.ntFx+self.ntFy

        for cell in self.tree.cells:
            ind = cell.index
            if dim==2:
                J1[ind] = cell.edges[2+faces[0]].index
                J2[ind] = cell.edges[  faces[1]].index + offsets[1]
            else:
                J1[ind] = cell.faces[  faces[0]].index
                J2[ind] = cell.faces[2+faces[1]].index + offsets[1]
                J3[ind] = cell.faces[4+faces[2]].index + offsets[2]

        I = np.arange(dim*self.nC, dtype=np.int64)
        if dim==2:
            J = np.r_[J1, J2]
        else:
            J = np.r_[J1, J2, J3]
        V = np.ones(self.nC*dim, dtype=np.float64)

        P = sp.csr_matrix((V, (I, J)), shape=(self.dim*self.nC, self.ntF))
        Rf = self._deflate_faces()
        return P*Rf

    def _getFacePxx(self):
        def Pxx(xFace, yFace):
            return self._getFaceP(xFace, yFace, None)
        return Pxx

    def _getFacePxxx(self):
        def Pxxx(xFace, yFace, zFace):
            return self._getFaceP(xFace, yFace, zFace)
        return Pxxx

    def _getEdgeP(self, xEdge, yEdge, zEdge):
        cdef int dim = self.dim
        cdef int_t ind, id
        cdef int epc = 1<<(dim-1) #edges per cell 2/4

        cdef np.int64_t[:] I, J, J1, J2, J3
        cdef np.float64_t[:] V

        J1 = np.empty(self.nC, dtype=np.int64)
        J2 = np.empty(self.nC, dtype=np.int64)
        if dim==3:
            J3 = np.empty(self.nC, dtype=np.int64)

        cdef int[3] edges
        cdef np.int64_t[:] offsets = np.empty(self.dim, dtype=np.int64)
        try:
            edges[0] = int(xEdge[-1]) #0, 1, 2, 3
            edges[1] = int(yEdge[-1]) #0, 1, 2, 3
            if dim==3:
                edges[2] = int(zEdge[-1]) #0, 1, 2, 3
        except ValueError:
            raise Exception('Last character of edge string must be 0, 1, 2, or 3')

        offsets[0] = 0
        offsets[1] = self.ntEx
        if dim==3:
            offsets[2] = self.ntEx+self.ntEy

        for cell in self.tree.cells:
            ind = cell.index
            J1[ind] = cell.edges[0*epc + edges[0]].index + offsets[0]
            J2[ind] = cell.edges[1*epc + edges[1]].index + offsets[1]
            if dim==3:
                J3[ind] = cell.edges[2*epc + edges[2]].index + offsets[2]

        I = np.arange(dim*self.nC, dtype=np.int64)
        if dim==2:
            J = np.r_[J1, J2]
        else:
            J = np.r_[J1, J2, J3]
        V = np.ones(self.nC*dim, dtype=np.float64)

        P = sp.csr_matrix((V, (I, J)), shape=(self.dim*self.nC, self.ntE))
        Rf = self._deflate_edges()
        return P*Rf

    def _getEdgePxx(self):
        def Pxx(xEdge, yEdge):
            return self._getEdgeP(xEdge, yEdge, None)
        return Pxx

    def _getEdgePxxx(self):
        def Pxxx(xEdge, yEdge, zEdge):
            return self._getEdgeP(xEdge, yEdge, zEdge)
        return Pxxx

    def _cull_outer_simplices(self, np.float64_t[:,:] points, int[:,:] simplices,
                       cull_dir='xyz'):
        cdef int cull_x, cull_y, cull_z
        cdef int i, ii, id, n_simps, dim

        n_simps = simplices.shape[0]
        dim = self.dim

        cull_x = 1 if 'x' in cull_dir else 0
        cull_y = 1 if 'y' in cull_dir else 0
        cull_z = 1 if (dim==3 and 'z' in cull_dir) else 0
        cdef np.float64_t[:] center = np.zeros(dim)
        cdef c_Cell *cell
        is_inside = np.ones(n_simps, dtype=np.bool)
        cdef double diff

        for i in range(n_simps):
            for ii in range(dim+1):
                for id in range(dim):
                    center[id] += points[simplices[i,ii],id]

            for id in range(dim):
                center[id] *= 1.0/(dim+1.0)

            if dim==2:
                cell = self.tree.containing_cell(center[0], center[1], 0.0)
                iz = 0
            else:
                cell = self.tree.containing_cell(center[0], center[1], center[2])

            if cull_x:
                diff = center[0]-cell.location[0]
                if ((diff<0 and cell.neighbors[0]==NULL) or
                    (diff>0 and cell.neighbors[1]==NULL)):
                    is_inside[i] = False
            if cull_y:
                diff = center[1]-cell.location[1]
                if ((diff<0 and cell.neighbors[2]==NULL) or
                    (diff>0 and cell.neighbors[3]==NULL)):
                    is_inside[i] = False
            if cull_z:
                diff = center[2]-cell.location[2]
                if ((diff<0 and cell.neighbors[4]==NULL) or
                    (diff>0 and cell.neighbors[5]==NULL)):
                    is_inside[i] = False
            center[:] = 0.0
        return is_inside

    def _get_grid_triang(self, grid='CC', double eps=1E-8):

        if grid=='CC':
            points = self.gridCC
            cull = 'xyz'
        elif grid=='Fx':
            points = self.gridFx
            cull = 'yz'
        elif grid=='Fy':
            points = self.gridFy
            cull = 'xz'
        elif grid=='Fz':
            points = self.gridFz
            cull = 'xy'
        elif grid=='Ex':
            points = self.gridEx
            cull = 'x'
        elif grid=='Ey':
            points = self.gridEy
            cull = 'y'
        elif grid=='Ez':
            points = self.gridEz
            cull = 'z'

        triang = Delaunay(points)

        is_inside = self._cull_outer_simplices(points, triang.simplices, cull)

        p = np.full(triang.nsimplex+1,-1, dtype=triang.neighbors.dtype)

        triang.simplices = triang.simplices[is_inside].copy()
        triang.equations = triang.equations[is_inside].copy()
        triang.nsimplex = triang.simplices.shape[0]

        p[:-1][is_inside] = np.arange(triang.nsimplex, dtype=p.dtype)

        #neighbors need to be renumbered
        neighbors = triang.neighbors[is_inside].copy()
        neighbors = p[neighbors]

        triang.neighbors = neighbors
        triang._transform = None
        triang._vertex_to_simplex = None
        triang._vertex_neighbor_vertices = None

        # Backwards compatibility (Scipy < 0.12.0)
        triang.vertices = triang.simplices

        return triang

    def getInterpolationMat(self, locs, locType, zerosOutside=False):
        if locType not in ['N', 'CC', "Ex", "Ey", "Ez", "Fx", "Fy", "Fz"]:
            raise Exception('locType must be one of N, CC, Ex, Ey, Ez, Fx, Fy, or Fz')
        if locType=='N':
            return self._getNodeIntMat(locs, zerosOutside)
        tri = self._get_grid_triang(grid=locType)

        if self.dim==2 and locType in ['Ez','Fz']:
            raise Exception('Unable to interpolate from Z edges/face in 2D')

        cdef int[:,:] simplices = tri.simplices

        cdef int i_out, i_p, i, dim, n_points, npi, n_grid
        cdef int found = 0
        dim = self.dim

        cdef double[:,:] points = np.atleast_2d(locs).copy()
        cdef double[:,:] grid_points = tri.points
        cdef double[:] point, proj_point
        cdef double *p0
        cdef double *p1
        cdef double *p2
        cdef double d, dsq, inf=np.inf
        cdef int contains_point

        n_points = points.shape[0]
        n_grid = grid_points.shape[0]
        npsimps = tri.find_simplex(locs)
        cdef int[:] simps = npsimps
        cdef int[:] simplex
        cdef int[:, :] hull
        cdef int[:] hull_points
        cdef int[:] npis

        proj_point = np.empty_like(points[0])
        point = np.empty_like(points[0])

        np_outside_points = np.where(npsimps==-1)[0].astype(np.int)
        cdef int[:] outside_points = np_outside_points
        n_outside = outside_points.shape[0]
        cdef double[:] barys = np.empty(dim, dtype=float)

        trans = tri.transform[npsimps]
        shift = np.array(points)-trans[:, dim]
        bs = np.einsum('ikj,ij->ik', trans[:, :dim], shift)
        bs = np.c_[bs, 1-bs.sum(axis=1)]

        I = np.column_stack((dim+1)*[np.arange(n_points)])
        J = tri.simplices[npsimps]
        V = bs
        cdef int[:,:] Js = J
        cdef double[:, :] Vs = V

        if n_outside > 0 and not zerosOutside:
            #oh boy got some points that need to be extrapolated

            hull = tri.convex_hull
            np_hull_points = np.unique(hull)
            hull_points = np_hull_points


            n_hull_simps = hull.shape[0]

            kdtree = cKDTree(tri.points[np_hull_points])
            npis = kdtree.query(locs[np_outside_points])[1]

            for i_out in range(n_outside):
                i_p = outside_points[i_out]
                point[:] = points[i_p]
                npi = hull_points[npis[i_out]]

                d = inf
                for i_s in range(n_hull_simps):
                    simplex = hull[i_s]
                    contains_point = 0
                    for i in range(dim):
                        contains_point = contains_point or simplex[i]==npi
                    if not contains_point:
                        continue
                    p0 = &grid_points[simplex[0], 0]
                    p1 = &grid_points[simplex[1], 0]
                    if dim==2:
                        _project_point_to_edge(&point[0], p0, p1, dim, &proj_point[0])
                    else:
                        p2 = &grid_points[simplex[2], 0]
                        _project_point_to_triangle(&point[0], p0, p1, p2, dim, &proj_point[0])

                    dsq = 0
                    for i in range(dim):
                        dsq += (point[i]-proj_point[i])*(point[i]-proj_point[i])
                    if dsq < d:
                        d = dsq
                        if dim==2:
                            _barycentric_edge(&proj_point[0], p0, p1, &barys[0], dim)
                        else:
                            _barycentric_triangle(&proj_point[0], p0, p1, p2, &barys[0], dim)
                        Js[i_p,0:dim] = simplex[:]
                        Vs[i_p,0:dim] = barys[:]
                        Vs[i_p,dim] = 0.0

        if zerosOutside:
                V[outside_points] = 0.0

        if locType[0] == 'F':
            n_grid = self.nF
            if locType[-1] == 'y':
                J += self.nFx
            elif locType[-1] == 'z':
                J += self.nFx + self.nFy
        elif locType[0] == 'E':
            n_grid = self.nE
            if locType[-1] == 'y':
                J += self.nEx
            elif locType[-1] == 'z':
                J += self.nEx + self.nEy

        return sp.csr_matrix((V.reshape(-1), (I.reshape(-1), J.reshape(-1))),
                             shape=(n_points,n_grid))

    def _getNodeIntMat(self, locs, zerosOutside):
        cdef:
            double[:, :] locations = locs
            int_t dim = self.dim
            int_t n_loc = locs.shape[0]
            int_t n_nodes = 1<<dim
            np.int64_t[:] I = np.empty(n_loc*n_nodes, dtype=np.int64)
            np.int64_t[:] J = np.empty(n_loc*n_nodes, dtype=np.int64)
            np.float64_t[:] V = np.empty(n_loc*n_nodes, dtype=np.float64)

            int_t ii,i
            c_Cell *cell
            double x, y, z
            double wx, wy, wz
            double eps = 100*np.finfo(float).eps
            int zeros_out = zerosOutside

        for i in range(n_loc):
            x = locations[i, 0]
            y = locations[i, 1]
            if dim==3:
                z = locations[i, 1]
            else:
                z = 0.0
            #get containing (or closest) cell
            cell = self.tree.containing_cell(x,y,z)
            #calculate weights
            wx = ((cell.points[3].location[0]-x)/
                  (cell.points[3].location[0]-cell.points[0].location[0]))
            wy = ((cell.points[3].location[1]-y)/
                  (cell.points[3].location[1]-cell.points[0].location[1]))
            if dim==3:
                wz = ((cell.points[7].location[2]-z)/
                      (cell.points[7].location[2]-cell.points[0].location[2]))
            else:
                wz = 1.0


            I[n_nodes*i:n_nodes*i+n_nodes] = i


            if zeros_out:
                if (wx<-eps or wy<-eps or wz<-eps or
                    wx>1+eps or wy>1+eps or wz>1+eps):
                    for ii in range(n_nodes):
                        J[n_nodes*i+ii] = 0
                        V[n_nodes*i+ii] = 0.0
                    continue

            wx = _clip01(wx)
            wy = _clip01(wy)
            wz = _clip01(wz)
            for ii in range(n_nodes):
                J[n_nodes*i+ii] = cell.points[ii].index

            V[n_nodes*i  ] = wx*wy*wz
            V[n_nodes*i+1] = (1-wx)*wy*wz
            V[n_nodes*i+2] = wx*(1-wy)*wz
            V[n_nodes*i+3] = (1-wx)*(1-wy)*wz
            if dim==3:
                V[n_nodes*i+4] = wx*wy*(1-wz)
                V[n_nodes*i+5] = (1-wx)*wy*(1-wz)
                V[n_nodes*i+6] = wx*(1-wy)*(1-wz)
                V[n_nodes*i+7] = (1-wx)*(1-wy)*(1-wz)

        Rn = self._deflate_nodes()
        return sp.csr_matrix((V, (I, J)), shape=(locs.shape[0],self.ntN))*Rn

    def plotGrid(self, ax=None, showIt=False,
        grid=True,
        cells=False, cellLine=False,
        nodes = False,
        facesX = False, facesY = False, facesZ = False,
        edgesX = False, edgesY = False, edgesZ = False):

        import matplotlib
        if ax is None:
            import matplotlib.pyplot as plt
            import matplotlib.colors as colors
            import matplotlib.cm as cmx
            if(self.dim==2):
                ax = plt.subplot(111)
            else:
                from mpl_toolkits.mplot3d import Axes3D
                ax = plt.subplot(111, projection='3d')
        else:
            assert isinstance(ax,matplotlib.axes.Axes), "ax must be an Axes!"
            fig = ax.figure

        cdef:
            int_t i, offset
            Node *p1
            Node *p2
            Edge *edge

        if grid:
            if(self.dim)==2:
                X = np.empty((self.nE*3,))
                Y = np.empty((self.nE*3,))
                for it in self.tree.edges_x:
                    edge = it.second
                    if(edge.hanging): continue
                    i = edge.index*3
                    p1 = edge.points[0]
                    p2 = edge.points[1]
                    X[i:i+3] = [p1.location[0],p2.location[0],np.nan]
                    Y[i:i+3] = [p1.location[1],p2.location[1],np.nan]

                offset = self.nEx
                for it in self.tree.edges_y:
                    edge = it.second
                    if(edge.hanging): continue
                    i = (edge.index+offset)*3
                    p1 = edge.points[0]
                    p2 = edge.points[1]
                    X[i:i+3] = [p1.location[0],p2.location[0],np.nan]
                    Y[i:i+3] = [p1.location[1],p2.location[1],np.nan]

                ax.plot(X, Y, 'b-')
            else:
                X = np.empty((self.nE*3,))
                Y = np.empty((self.nE*3,))
                Z = np.empty((self.nE*3,))
                for it in self.tree.edges_x:
                    edge = it.second
                    if(edge.hanging): continue
                    i = edge.index*3
                    p1 = edge.points[0]
                    p2 = edge.points[1]
                    X[i:i+3] = [p1.location[0], p2.location[0], np.nan]
                    Y[i:i+3] = [p1.location[1], p2.location[1], np.nan]
                    Z[i:i+3] = [p1.location[2], p2.location[2], np.nan]

                offset = self.nEx
                for it in self.tree.edges_y:
                    edge = it.second
                    if(edge.hanging): continue
                    i = (edge.index+offset)*3
                    p1 = edge.points[0]
                    p2 = edge.points[1]
                    X[i:i+3] = [p1.location[0], p2.location[0], np.nan]
                    Y[i:i+3] = [p1.location[1], p2.location[1], np.nan]
                    Z[i:i+3] = [p1.location[2], p2.location[2], np.nan]

                offset += self.nEy
                for it in self.tree.edges_z:
                    edge = it.second
                    if(edge.hanging): continue
                    i = (edge.index+offset)*3
                    p1 = edge.points[0]
                    p2 = edge.points[1]
                    X[i:i+3] = [p1.location[0], p2.location[0], np.nan]
                    Y[i:i+3] = [p1.location[1], p2.location[1], np.nan]
                    Z[i:i+3] = [p1.location[2], p2.location[2], np.nan]

                ax.plot(X, Y, 'b-', zs=Z)

        if cells:
            ax.plot(*self.gridCC.T, 'r.')
        if cellLine:
            ax.plot(*self.gridCC.T, 'r:')
            ax.plot(self.gridCC[[0,-1],0], self.gridCC[[0,-1],1], 'ro')
        if nodes:
            ax.plot(*self.gridN.T, 'ms')
            # Hanging Nodes
            ax.plot(*self.gridhN.T, 'ms')
            ax.plot(*self.gridhN.T, 'ms', ms=10, mfc='none', mec='m')
        if facesX:
            ax.plot(*self.gridFx.T, 'g>')
            # Hanging Faces x
            ax.plot(*self.gridhFx.T, 'g>')
            ax.plot(*self.gridhFx.T, 'gs', ms=10, mfc='none', mec='g')
        if facesY:
            ax.plot(*self.gridFy.T, 'g^')
            # Hanging Faces y
            ax.plot(*self.gridhFy.T, 'g^')
            ax.plot(*self.gridhFy.T, 'gs', ms=10, mfc='none', mec='g')
        if facesZ:
            ax.plot(*self.gridFz.T, 'g^')
            #Hanging
            ax.plot(*self.gridhFz.T, 'gs')
            ax.plot(*self.gridhFz.T, 'gs', ms=10, mfc='none', mec='g')
        if edgesX:
            ax.plot(*self.gridEx.T, 'k>')
            ax.plot(*self.gridhEx.T, 'k>')
            ax.plot(*self.gridhEx.T, 'ks', ms=10, mfc='none', mec='k')
        if edgesY:
            ax.plot(*self.gridEy.T, 'k>')
            ax.plot(*self.gridhEy.T, 'k>')
            ax.plot(*self.gridhEy.T, 'ks', ms=10, mfc='none', mec='k')
        if edgesZ:
            ax.plot(*self.gridEz.T, 'k>')
            ax.plot(*self.gridhEz.T, 'k>')
            ax.plot(*self.gridhEz.T, 'ks', ms=10, mfc='none', mec='k')

        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        if self.dim==3:
            ax.set_zlabel('x3')

        ax.grid(True)
        if showIt:
            plt.show()

    def __len__(self):
        return self.nC

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Get the start, stop, and step from the slice
            return [self[ii] for ii in range(*key.indices(len(self)))]
        elif isinstance(key, int):
            if key < 0:  # Handle negative indices
                key += len(self)
            if key >= len(self):
                raise IndexError(
                    "The index ({0:d}) is out of range.".format(key)
                )
            pycell = Cell()
            pycell._set(self.tree.cells[key])
            return pycell
        else:
            raise TypeError("Invalid argument type.")

    @property
    def _ubc_indArr(self):
        if self.__ubc_indArr is not None:
            return self.__ubc_indArr
        indArr = np.empty((self.nC, 4), dtype=np.int)
        cdef np.int_t[:,:] _indArr = indArr
        cdef int id, dim = self.dim

        for cell in self.tree.cells:
            for id in range(dim):
                _indArr[cell.index, id] = cell.location_ind[id]
            _indArr[cell.index, 3] = cell.level

        max_level = self.max_level

        indArr[:, -1] = 1<<(max_level - indArr[:, -1])
        indArr[:, 2] = (2<<max_level) - indArr[:, 2]

        indArr[:, :-1] = (indArr[:, :-1] - indArr[:, -1, None])//2
        indArr[:, :-1] += 1

        self.__ubc_indArr = indArr
        return self.__ubc_indArr

    @property
    def _ubc_order(self):
        if self.__ubc_order is not None:
            return self.__ubc_order
        indArr = self._ubc_indArr
        self.__ubc_order = np.lexsort((indArr[:, 0], indArr[:, 1], indArr[:, 2]))
        return self.__ubc_order

    def __dealloc__(self):
        del self.tree
        del self.wrapper

cdef inline double _clip01(double x) nogil:
    return min(1, max(x, 0))

cdef inline double _sctp(double *v0, double *v1, double *v2) nogil:
    #dim = 3
    return (v0[0]*(v1[1]*v2[2] - v1[2]*v2[1])
            - v0[1]*(v1[0]*v2[2] - v1[2]*v2[0])
            + v0[2]*(v1[0]*v2[1] - v1[1]*v2[0]))

@cython.cdivision(True)
cdef void _project_point_to_edge(double *p, double *p0, double *p1,
                                 int dim, double *out) nogil:
    # dim can equal 2 or 3
    """Projects a point onto a line segment."""
    # Projects a point onto a line segment

    cdef double v1=0.0, v2=0.0, t = 0.0
    cdef int i

    for i in range(dim):
        v1 += (p[i]-p0[i])*(p1[i]-p0[i])
        v2 += (p1[i]-p0[i])*(p1[i]-p0[i])
    t = _clip01(v1/v2)
    for i in range(dim):
        out[i] = (1.0-t)*p0[i] + t*p1[i]

@cython.cdivision(True)
cdef void _project_point_to_triangle(double *p, double *p0, double *p1, double *p2,
                                    int dim, double *out) nogil:
    #dim can equal 2 or 3
    cdef double[3] bary
    cdef int i
    _barycentric_triangle(p, p0, p1, p2, bary, dim)
    if bary[0] < 0:
        _project_point_to_edge(p, p1, p2, dim, out)
    elif bary[1] < 0:
        _project_point_to_edge(p, p0, p2, dim, out)
    elif bary[2] < 0:
        _project_point_to_edge(p, p0, p1, dim, out)
    else:
        for i in range(dim):
            out[i] = bary[0]*p0[i] + bary[1]*p1[i] + bary[2]*p2[i]

@cython.cdivision(True)
cdef void _project_point_to_tetrahedron(double *p, double *p0, double *p1,
                                       double *p2, double *p3, double *out) nogil:
    cdef double[4] bary

    _barycentric_tetrahedron(p, p0, p1, p2, p3, bary)

    if bary[0] < 0:
        _project_point_to_triangle(p, p1, p2, p3, 3, out)
    elif bary[1] < 0:
        _project_point_to_triangle(p, p0, p2, p3, 3, out)
    elif bary[2] < 0:
        _project_point_to_triangle(p, p0, p1, p3, 3, out)
    elif bary[3] < 0:
        _project_point_to_triangle(p, p0, p1, p2, 3, out)
    else:
        out[0] = p[0]
        out[1] = p[1]
        out[2] = p[2]


@cython.cdivision(True)
cdef void _barycentric_edge(double *p, double *p0, double *p1, double *bary, int dim) nogil:
    cdef double v1=0.0, v2=0.0
    cdef int i

    for i in range(dim):
        v1 += (p[i]-p0[i])*(p1[i]-p0[i])
        v2 += (p1[i]-p0[i])*(p1[i]-p0[i])
    bary[1] = v1/v2
    bary[0] = 1-bary[1]

@cython.cdivision(True)
cdef void _barycentric_triangle(double *p, double *p0, double *p1, double *p2,
                            double *bary, int dim) nogil:
    cdef double d00=0, d01=0, d11=0, d20=0, d21=0
    cdef int i
    for i in range(dim):
        bary[0] = p1[i]-p0[i]
        bary[1] = p2[i]-p0[i]
        bary[2] = p[i]-p0[i]

        d00 += bary[0]*bary[0]
        d01 += bary[0]*bary[1]
        d11 += bary[1]*bary[1]
        d20 += bary[2]*bary[0]
        d21 += bary[2]*bary[1]

    bary[0] = 1.0/(d00*d11 - d01*d01)
    bary[1] = (d11 * d20 - d01 * d21) * bary[0]
    bary[2] = (d00 * d21 - d01 * d20) * bary[0]
    bary[0] = 1.0 - bary[1] - bary[2]

@cython.cdivision(True)
cdef void _barycentric_tetrahedron(double *p, double *p0, double *p1,
                               double *p2, double *p3, double *bary) nogil:
    cdef double[3] vap, vab, vac, vad
    cdef int i
    for i in range(3):
        vap[i] = p[i]-p0[i]

        vab[i] = p1[i]-p0[i] #
        vac[i] = p2[i]-p0[i] #
        vad[i] = p3[i]-p0[i] #

    bary[0] = 1.0/_sctp(vab, vac, vad)

    bary[1] = _sctp(vap, vac, vad)*bary[0]
    bary[2] = _sctp(vap, vad, vab)*bary[0]
    bary[3] = _sctp(vap, vab, vac)*bary[0]
    bary[0] = 1.0 - bary[1] - bary[2] - bary[3]
