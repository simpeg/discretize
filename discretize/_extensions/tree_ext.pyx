# distutils: language=c++
# cython: embedsignature=True, language_level=3
cimport cython
cimport numpy as np
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from numpy.math cimport INFINITY

from tree cimport int_t, Tree as c_Tree, PyWrapper, Node, Edge, Face, Cell as c_Cell

import scipy.sparse as sp
import numpy as np
from discretize._extensions.interputils_cython cimport _bisect_left, _bisect_right


cdef class TreeCell:
    """A Cell of the `TreeMesh`

    This cannot be created in python, it can only be accessed by indexing the
    `TreeMesh` object.

    This is also the object that is passed to the user defined refine function
    when calling TreeMesh.refine(func).

    Notes
    -----
    When called as part of the `refine` function, only the origin, center, and h
    properties are valid.
    """
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

        self._wx = cell.points[3].location[0] - self._x0
        self._wy = cell.points[3].location[1] - self._y0
        if(self._dim > 2):
            self._z = cell.location[2]
            self._z0 = cell.points[0].location[2]
            self._wz = cell.points[7].location[2] - self._z0

    @property
    def nodes(self):
        """indexes of this cell's nodes

        Returns
        -------
        list of ints
        """
        cdef Node *points[8]
        points = self._cell.points
        if self._dim == 3:
            return [points[0].index, points[1].index,
                    points[2].index, points[3].index,
                    points[4].index, points[5].index,
                    points[6].index, points[7].index]
        return [points[0].index, points[1].index,
                points[2].index, points[3].index]

    @property
    def edges(self):
        """indexes of this cell's edges

        Returns
        -------
        list of ints
        """
        cdef Edge *edges[12]
        edges = self._cell.edges
        if self._dim == 2:
            return [edges[0].index, edges[1].index,
                    edges[2].index, edges[3].index]
        return [
            edges[0].index, edges[1].index, edges[2].index, edges[3].index,
            edges[4].index, edges[5].index, edges[6].index, edges[7].index,
            edges[8].index, edges[9].index, edges[10].index, edges[11].index,
        ]

    @property
    def faces(self):
        """indexes of this cell's faces
        Returns
        -------
        list of ints
        """
        cdef Face *faces[6]
        faces = self._cell.faces
        if self._ == 3:
            return [
                faces[0].index, faces[1].index,
                faces[2].index, faces[3].index,
                faces[4].index, faces[5].index
            ]
        cdef Edge *edges[12]
        edges = self._cell.edges
        return [edges[2].index, edges[3].index,
                edges[0].index, edges[1].index]

    @property
    def center(self):
        """numpy.array of length dim"""
        if self._dim == 2: return np.array([self._x, self._y])
        return np.array([self._x, self._y, self._z])

    @property
    def origin(self):
        """numpy.array of length dim"""
        if self._dim == 2: return np.array([self._x0, self._y0])
        return np.array([self._x0, self._y0, self._z0])

    @property
    def x0(self):
        return self.origin

    @property
    def h(self):
        """ numpy.array of length dim
        width of this cell
        """
        if self._dim == 2: return np.array([self._wx, self._wy])
        return np.array([self._wx, self._wy, self._wz])

    @property
    def dim(self):
        """"int dimension of cell"""
        return self._dim

    @property
    def index(self):
        """integer index of this cell"""
        return self._cell.index

    @property
    def neighbors(self):
        """ The indexes of this cell's neighbors

        Indexes of this cell's neighbors. If a cell has more than one neighbor
        in a certain direction (i.e. when a level changes between adjacent cells),
        then that entry will also be a list of all of those neighbor indices.
        The list is order -x, +x, -y, +y, -z, +z. If a cell has no neighbor in
        that direction, the value will be -1.

        Returns
        -------
        list of ints or list of ints
        """
        neighbors = [-1]*self._dim*2

        for i in range(self._dim*2):
            if self._cell.neighbors[i] is NULL:
                continue
            elif self._cell.neighbors[i].is_leaf():
                neighbors[i] = self._cell.neighbors[i].index
            else:
                if self._dim==2:
                    if i==0:
                        neighbors[i] = [self._cell.neighbors[i].children[1].index,
                                        self._cell.neighbors[i].children[3].index]
                    elif i==1:
                        neighbors[i] = [self._cell.neighbors[i].children[0].index,
                                        self._cell.neighbors[i].children[2].index]
                    elif i==2:
                        neighbors[i] = [self._cell.neighbors[i].children[2].index,
                                        self._cell.neighbors[i].children[3].index]
                    else:
                        neighbors[i] = [self._cell.neighbors[i].children[0].index,
                                        self._cell.neighbors[i].children[1].index]
                else:
                    if i==0:
                        neighbors[i] = [self._cell.neighbors[i].children[1].index,
                                        self._cell.neighbors[i].children[3].index,
                                        self._cell.neighbors[i].children[5].index,
                                        self._cell.neighbors[i].children[7].index]
                    elif i==1:
                        neighbors[i] = [self._cell.neighbors[i].children[0].index,
                                        self._cell.neighbors[i].children[2].index,
                                        self._cell.neighbors[i].children[4].index,
                                        self._cell.neighbors[i].children[6].index]
                    elif i==2:
                        neighbors[i] = [self._cell.neighbors[i].children[2].index,
                                        self._cell.neighbors[i].children[3].index,
                                        self._cell.neighbors[i].children[6].index,
                                        self._cell.neighbors[i].children[7].index]
                    elif i==3:
                        neighbors[i] = [self._cell.neighbors[i].children[0].index,
                                        self._cell.neighbors[i].children[1].index,
                                        self._cell.neighbors[i].children[4].index,
                                        self._cell.neighbors[i].children[5].index]
                    elif i==4:
                        neighbors[i] = [self._cell.neighbors[i].children[4].index,
                                        self._cell.neighbors[i].children[5].index,
                                        self._cell.neighbors[i].children[6].index,
                                        self._cell.neighbors[i].children[7].index]
                    else:
                        neighbors[i] = [self._cell.neighbors[i].children[0].index,
                                        self._cell.neighbors[i].children[1].index,
                                        self._cell.neighbors[i].children[2].index,
                                        self._cell.neighbors[i].children[3].index]
        return neighbors

    @property
    def _index_loc(self):
        if self._dim == 2:
            return tuple((self._cell.location_ind[0], self._cell.location_ind[1]))
        return tuple((self._cell.location_ind[0], self._cell.location_ind[1],
                      self._cell.location_ind[2]))

    @property
    def _level(self):
        return self._cell.level

cdef int_t _evaluate_func(void* function, c_Cell* cell) with gil:
    # Wraps a function to be called in C++
    func = <object> function
    pycell = TreeCell()
    pycell._set(cell)
    return <int_t> func(pycell)

cdef class _TreeMesh:
    cdef c_Tree *tree
    cdef PyWrapper *wrapper
    cdef int_t _dim
    cdef int_t[3] ls
    cdef int _finalized

    cdef double[:] _xs, _ys, _zs
    cdef double[:] _origin

    cdef object _cell_centers, _nodes, _hanging_nodes
    cdef object _edges_x, _edges_y, _edges_z, _hanging_edges_x, _hanging_edges_y, _hanging_edges_z
    cdef object _faces_x, _faces_y, _faces_z, _hanging_faces_x, _hanging_faces_y, _hanging_faces_z

    cdef object _h_gridded
    cdef object _cell_volumes, _face_areas, _edge_lengths
    cdef object _average_face_x_to_cell, _average_face_y_to_cell, _average_face_z_to_cell, _average_face_to_cell, _average_face_to_cell_vector,
    cdef object _average_node_to_cell, _average_node_to_edge, _average_node_to_edge_x, _average_node_to_edge_y, _average_node_to_edge_z
    cdef object _average_node_to_face, _average_node_to_face_x, _average_node_to_face_y, _average_node_to_face_z
    cdef object _average_edge_x_to_cell, _average_edge_y_to_cell, _average_edge_z_to_cell, _average_edge_to_cell, _average_edge_to_cell_vector
    cdef object _average_cell_to_face, _average_cell_vector_to_face, _average_cell_to_face_x, _average_cell_to_face_y, _average_cell_to_face_z
    cdef object _face_divergence
    cdef object _edge_curl, _nodal_gradient

    cdef object __ubc_order, __ubc_indArr

    def __cinit__(self, *args, **kwargs):
        self.wrapper = new PyWrapper()
        self.tree = new c_Tree()

    def __init__(self, h, origin):
        nx2 = 2*len(h[0])
        ny2 = 2*len(h[1])
        self._dim = len(origin)
        self._origin = origin

        xs = np.empty(nx2 + 1, dtype=float)
        xs[::2] = np.cumsum(np.r_[origin[0], h[0]])
        xs[1::2] = (xs[:-1:2] + xs[2::2])/2
        self._xs = xs
        self.ls[0] = int(np.log2(len(h[0])))

        ys = np.empty(ny2 + 1, dtype=float)
        ys[::2] = np.cumsum(np.r_[origin[1],h[1]])
        ys[1::2] = (ys[:-1:2] + ys[2::2])/2
        self._ys = ys
        self.ls[1] = int(np.log2(len(h[1])))

        if self._dim > 2:
            nz2 = 2*len(h[2])

            zs = np.empty(nz2 + 1, dtype=float)
            zs[::2] = np.cumsum(np.r_[origin[2],h[2]])
            zs[1::2] = (zs[:-1:2] + zs[2::2])/2
            self._zs = zs
            self.ls[2] = int(np.log2(len(h[2])))
        else:
            self._zs = np.zeros(1, dtype=float)
            self.ls[2] = 1

        self.tree.set_dimension(self._dim)
        self.tree.set_levels(self.ls[0], self.ls[1], self.ls[2])
        self.tree.set_xs(&self._xs[0], &self._ys[0], &self._zs[0])
        self.tree.initialize_roots()
        self._finalized = False
        self._clear_cache()

    def _clear_cache(self):
        self._cell_centers = None
        self._nodes = None
        self._hanging_nodes = None
        self._h_gridded = None

        self._edges_x = None
        self._edges_y = None
        self._edges_z = None
        self._hanging_edges_x = None
        self._hanging_edges_y = None
        self._hanging_edges_z = None

        self._faces_x = None
        self._faces_y = None
        self._faces_z = None
        self._hanging_faces_x = None
        self._hanging_faces_y = None
        self._hanging_faces_z = None

        self._cell_volumes = None
        self._face_areas = None
        self._edge_lengths = None

        self._average_cell_to_face = None
        self._average_cell_to_face_x = None
        self._average_cell_to_face_y = None
        self._average_cell_to_face_z = None

        self._average_face_x_to_cell = None
        self._average_face_y_to_cell = None
        self._average_face_z_to_cell = None
        self._average_face_to_cell = None
        self._average_face_to_cell_vector = None

        self._average_edge_x_to_cell = None
        self._average_edge_y_to_cell = None
        self._average_edge_z_to_cell = None
        self._average_edge_to_cell = None
        self._average_edge_to_cell_vector = None

        self._average_node_to_cell = None
        self._average_node_to_edge = None
        self._average_node_to_face = None
        self._average_node_to_edge_x = None
        self._average_node_to_edge_y = None
        self._average_node_to_edge_z = None
        self._average_node_to_face_x = None
        self._average_node_to_face_y = None
        self._average_node_to_face_z = None

        self._face_divergence = None
        self._nodal_gradient = None
        self._edge_curl = None

        self.__ubc_order = None
        self.__ubc_indArr = None

    def refine(self, function, finalize=True):
        """ Refine a TreeMesh using a user supplied function.

        Refines the TreeMesh using a function that is recursively called on each
        cell of the mesh. It must accept an object of type `discretize.TreeMesh.Cell`
        and return an integer like object defining the desired level.
        The function can also simply be an integer, which will then cause all
        cells to be at least that level.

        Parameters
        ----------
        function : callable | int
            a function describing the desired level,
            or an integer to refine all cells to at least that level.
        finalize : bool, optional
            Whether to finalize the mesh

        Examples
        --------
        >>> from discretize import TreeMesh
        >>> mesh = TreeMesh([32,32])
        >>> def func(cell):
        >>>     r = np.linalg.norm(cell.center-0.5)
        >>>     return mesh.max_level if r<0.2 else mesh.max_level-1
        >>> mesh.refine(func)
        >>> mesh
        ---- QuadTreeMesh ----
         origin: 0.00, 0.00
         hx: 32*0.03,
         hy: 32*0.03,
        n_cells: 352
        Fill: 34.38%

        See Also
        --------
        discretize.TreeMesh.TreeCell : a description of the TreeCell object
        """
        if isinstance(function, int):
            level = function
            function = lambda cell: level

        #Wrapping function so it can be called in c++
        cdef void * func_ptr = <void *> function
        self.wrapper.set(func_ptr, _evaluate_func)
        #Then tell c++ to build the tree
        self.tree.build_tree_from_function(self.wrapper)
        if finalize:
            self.finalize()

    def insert_cells(self, points, levels, finalize=True):
        """Insert cells into the TreeMesh that contain given points

        Insert cell(s) into the TreeMesh that contain the given point(s) at the
        assigned level(s).

        Parameters
        ----------
        points : array_like with shape (N, dim)
        levels : array_like of integers with shape (N)
        finalize : bool, optional
            Whether to finalize after inserting point(s)

        Examples
        --------
        >>> from discretize import TreeMesh
        >>> mesh = TreeMesh([32,32])
        >>> mesh.insert_cells([0.5, 0.5], mesh.max_level)
        >>> print(mesh)
        ---- QuadTreeMesh ----
         origin: 0.00, 0.00
         hx: 32*0.03,
         hy: 32*0.03,
        n_cells: 40
        Fill: 3.91%
        """
        points = np.require(np.atleast_2d(points), dtype=np.float64,
                            requirements='C')
        cdef double[:, :] cs = points
        cdef int[:] ls = np.require(np.atleast_1d(levels), dtype=np.int32,
                                    requirements='C')
        cdef int_t i
        for i in range(ls.shape[0]):
            self.tree.insert_cell(&cs[i, 0], ls[i])
        if finalize:
            self.finalize()

    def finalize(self):
        """Finalize the TreeMesh
        Called after finished cronstruction of the mesh. Can only be called once.
        After finalize is called, all other attributes and functions are valid.
        """
        if not self._finalized:
            self.tree.finalize_lists()
            self.tree.number()
            self._finalized=True

    def number(self):
        """Number the cells, nodes, faces, and edges of the TreeMesh"""
        self.tree.number()

    def _set_origin(self, origin):
        if not isinstance(origin, (list, tuple, np.ndarray)):
            raise ValueError('origin must be a list, tuple or numpy array')
        self._origin = np.asarray(origin, dtype=np.float64)
        cdef int_t dim = self._origin.shape[0]
        cdef double[:] shift
        #cdef c_Cell *cell
        cdef Node *node
        cdef Edge *edge
        cdef Face *face
        if self.tree.n_dim > 0: # Will only happen if __init__ has been called
            shift = np.empty(dim, dtype=np.float64)

            shift[0] = self._origin[0] - self._xs[0]
            shift[1] = self._origin[1] - self._ys[0]
            if dim == 3:
                shift[2] = self._origin[2] - self._zs[0]

            for i in range(self._xs.shape[0]):
                self._xs[i] += shift[0]
            for i in range(self._ys.shape[0]):
                self._ys[i] += shift[1]
            if dim == 3:
                for i in range(self._zs.shape[0]):
                    self._zs[i] += shift[2]

            #update the locations of all of the items
            self.tree.shift_cell_centers(&shift[0])

            for itN in self.tree.nodes:
                node = itN.second
                for i in range(dim):
                    node.location[i] += shift[i]

            for itE in self.tree.edges_x:
                edge = itE.second
                for i in range(dim):
                    edge.location[i] += shift[i]

            for itE in self.tree.edges_y:
                edge = itE.second
                for i in range(dim):
                    edge.location[i] += shift[i]

            if dim == 3:
                for itE in self.tree.edges_z:
                    edge = itE.second
                    for i in range(dim):
                        edge.location[i] += shift[i]

                for itF in self.tree.faces_x:
                    face = itF.second
                    for i in range(dim):
                        face.location[i] += shift[i]

                for itF in self.tree.faces_y:
                    face = itF.second
                    for i in range(dim):
                        face.location[i] += shift[i]

                for itF in self.tree.faces_z:
                    face = itF.second
                    for i in range(dim):
                        face.location[i] += shift[i]
            #clear out all cached grids
            self._cell_centers = None
            self._nodes = None
            self._hanging_nodes = None
            self._edges_x = None
            self._hanging_edges_x = None
            self._edges_y = None
            self._hanging_edges_y = None
            self._edges_z = None
            self._hanging_edges_z = None
            self._faces_x = None
            self._hanging_faces_x = None
            self._faces_y = None
            self._hanging_faces_y = None
            self._faces_z = None
            self._hanging_faces_z = None

    @property
    def fill(self):
        """
        How filled is the mesh compared to a TensorMesh?
        As a fraction: [0, 1].
        """
        #Tensor mesh cells:
        cdef int_t nxc, nyc, nzc;
        nxc = (self._xs.shape[0]-1)//2
        nyc = (self._ys.shape[0]-1)//2
        nzc = (self._zs.shape[0]-1)//2 if self._dim==3 else 1
        return float(self.n_cells)/(nxc * nyc * nzc)

    @property
    def max_used_level(self):
        """
        The maximum level used, which may be
        less than `max_level`.
        """
        cdef int level = 0
        for cell in self.tree.cells:
            level = max(level, cell.level)
        return level

    @property
    def max_level(self):
        """The maximum possible level for a cell on this mesh"""
        return self.tree.max_level

    @property
    def n_cells(self):
        """Number of cells"""
        return self.tree.cells.size()

    @property
    def n_nodes(self):
        """Number of non-hanging nodes"""
        return self.n_total_nodes - self.n_hanging_nodes

    @property
    def n_total_nodes(self):
        """Number of non-hanging and hanging nodes"""
        return self.tree.nodes.size()

    @property
    def n_hanging_nodes(self):
        """Number of hanging nodes"""
        return self.tree.hanging_nodes.size()

    @property
    def n_edges(self):
        """Total number of non-hanging edges amongst all dimensions"""
        return self.n_edges_x + self.n_edges_y + self.n_edges_z

    @property
    def n_hanging_edges(self):
        """Total number of hanging edges amongst all dimensions"""
        return self.n_hanging_edges_x + self.n_hanging_edges_y + self.n_hanging_edges_z

    @property
    def n_total_edges(self):
        """Total number of non-hanging and hanging edges amongst all dimensions"""
        return self.n_edges + self.n_hanging_edges

    @property
    def n_edges_x(self):
        """Number of non-hanging edges oriented along the first dimension"""
        return self.n_total_edges_x - self.n_hanging_edges_x

    @property
    def n_edges_y(self):
        """Number of non-hanging edges oriented along the second dimension"""
        return self.n_total_edges_y - self.n_hanging_edges_y

    @property
    def n_edges_z(self):
        """Number of non-hanging edges oriented along the third dimension"""
        return self.n_total_edges_z - self.n_hanging_edges_z

    @property
    def n_total_edges_x(self):
        """Number of non-hanging and hanging edges oriented along the first dimension"""
        return self.tree.edges_x.size()

    @property
    def n_total_edges_y(self):
        """Number of non-hanging and hanging edges oriented along the second dimension"""
        return self.tree.edges_y.size()

    @property
    def n_total_edges_z(self):
        """Number of non-hanging and hanging edges oriented along the third dimension"""
        return self.tree.edges_z.size()

    @property
    def n_hanging_edges_x(self):
        """Number of hanging edges oriented along the first dimension"""
        return self.tree.hanging_edges_x.size()

    @property
    def n_hanging_edges_y(self):
        """Number of hanging edges oriented along the second dimension"""
        return self.tree.hanging_edges_y.size()

    @property
    def n_hanging_edges_z(self):
        """Number of hanging edges oriented along the third dimension"""
        return self.tree.hanging_edges_z.size()

    @property
    def n_faces(self):
        """Total number of non-hanging faces amongst all dimensions"""
        return self.n_faces_x + self.n_faces_y + self.n_faces_z

    @property
    def n_hanging_faces(self):
        """Total number of hanging faces amongst all dimensions"""
        return self.n_hanging_faces_x + self.n_hanging_faces_y + self.n_hanging_faces_z

    @property
    def n_total_faces(self):
        """Total number of hanging and non-hanging faces amongst all dimensions"""
        return self.n_faces + self.n_hanging_faces

    @property
    def n_faces_x(self):
        """Number of non-hanging faces oriented along the first dimension"""
        return self.n_total_faces_x - self.n_hanging_faces_x

    @property
    def n_faces_y(self):
        """Number of non-hanging faces oriented along the second dimension"""
        return self.n_total_faces_y - self.n_hanging_faces_y

    @property
    def n_faces_z(self):
        """Number of non-hanging faces oriented along the third dimension"""
        return self.n_total_faces_z - self.n_hanging_faces_z

    @property
    def n_total_faces_x(self):
        """Number of non-hanging and hanging faces oriented along the first dimension"""
        if(self._dim == 2): return self.n_total_edges_y
        return self.tree.faces_x.size()

    @property
    def n_total_faces_y(self):
        """Number of non-hanging and hanging faces oriented along the second dimension"""
        if(self._dim == 2): return self.n_total_edges_x
        return self.tree.faces_y.size()

    @property
    def n_total_faces_z(self):
        """Number of non-hanging and hanging faces oriented along the third dimension"""
        if(self._dim == 2): return 0
        return self.tree.faces_z.size()

    @property
    def n_hanging_faces_x(self):
        """Number of hanging faces oriented along the first dimension"""
        if(self._dim == 2): return self.n_hanging_edges_y
        return self.tree.hanging_faces_x.size()

    @property
    def n_hanging_faces_y(self):
        """Number of hanging faces oriented along the second dimension"""
        if(self._dim == 2): return self.n_hanging_edges_x
        return self.tree.hanging_faces_y.size()

    @property
    def n_hanging_faces_z(self):
        """Number of hanging faces oriented along the third dimension"""
        if(self._dim == 2): return 0
        return self.tree.hanging_faces_z.size()

    @property
    def cell_centers(self):
        """
        Returns a numpy arrayof shape (n_cells, dim) with the center locations of all cells
        in order.
        """
        cdef np.float64_t[:, :] gridCC
        cdef np.int64_t ii, ind, dim
        if self._cell_centers is None:
            dim = self._dim
            self._cell_centers = np.empty((self.n_cells, self._dim), dtype=np.float64)
            gridCC = self._cell_centers
            for cell in self.tree.cells:
                ind = cell.index
                for ii in range(dim):
                    gridCC[ind, ii] = cell.location[ii]
        return self._cell_centers

    @property
    def nodes(self):
        """
        Returns a numpy array of shape (n_nodes, dim) with the locations of all
        non-hanging nodes in order.
        """
        cdef np.float64_t[:, :] gridN
        cdef Node *node
        cdef np.int64_t ii, ind, dim
        if self._nodes is None:
            dim = self._dim
            self._nodes = np.empty((self.n_nodes, dim) ,dtype=np.float64)
            gridN = self._nodes
            for it in self.tree.nodes:
                node = it.second
                if not node.hanging:
                    ind = node.index
                    for ii in range(dim):
                        gridN[ind, ii] = node.location[ii]
        return self._nodes

    @property
    def hanging_nodes(self):
        """
        Returns a numpy array of shape (n_nodes, dim) with the locations of all
        hanging nodes in order.
        """
        cdef np.float64_t[:, :] gridN
        cdef Node *node
        cdef np.int64_t ii, ind, dim
        if self._hanging_nodes is None:
            dim = self._dim
            self._hanging_nodes = np.empty((self.n_hanging_nodes, dim), dtype=np.float64)
            gridhN = self._hanging_nodes
            for node in self.tree.hanging_nodes:
                ind = node.index-self.n_nodes
                for ii in range(dim):
                    gridhN[ind, ii] = node.location[ii]
        return self._hanging_nodes

    @property
    def h_gridded(self):
        """
        Returns an (n_cells, dim) numpy array with the widths of all cells in order
        """
        if self._h_gridded is not None:
            return self._h_gridded
        cdef np.float64_t[:, :] gridCH
        cdef np.int64_t ii, ind, dim
        cdef np.float64_t len
        cdef int epc = 4 if self._dim==3 else 2
        dim = self._dim
        self._h_gridded = np.empty((self.n_cells, dim), dtype=np.float64)
        gridCH = self._h_gridded
        for cell in self.tree.cells:
            ind = cell.index
            for ii in range(dim):
                gridCH[ind, ii] = cell.edges[ii*epc].length

        return self._h_gridded

    @property
    def edges_x(self):
        """
        Returns a numpy array of shape (n_edges_x, dim) with the centers of all
        non-hanging edges along the first dimension in order.
        """
        cdef np.float64_t[:, :] gridEx
        cdef Edge *edge
        cdef np.int64_t ii, ind, dim
        if self._edges_x is None:
            dim = self._dim
            self._edges_x = np.empty((self.n_edges_x, dim), dtype=np.float64)
            gridEx = self._edges_x
            for it in self.tree.edges_x:
                edge = it.second
                if not edge.hanging:
                    ind = edge.index
                    for ii in range(dim):
                        gridEx[ind, ii] = edge.location[ii]
        return self._edges_x

    @property
    def hanging_edges_x(self):
        """
        Returns a numpy array of shape (n_hanging_edges_x, dim) with the centers of all
        hanging edges along the first dimension in order.
        """
        cdef np.float64_t[:, :] gridhEx
        cdef Edge *edge
        cdef np.int64_t ii, ind, dim
        if self._hanging_edges_x is None:
            dim = self._dim
            self._hanging_edges_x = np.empty((self.n_hanging_edges_x, dim), dtype=np.float64)
            gridhEx = self._hanging_edges_x
            for edge in self.tree.hanging_edges_x:
                ind = edge.index-self.n_edges_x
                for ii in range(dim):
                    gridhEx[ind, ii] = edge.location[ii]
        return self._hanging_edges_x

    @property
    def edges_y(self):
        """
        Returns a numpy array of shape (n_edges_y, dim) with the centers of all
        non-hanging edges along the second dimension in order.
        """
        cdef np.float64_t[:, :] gridEy
        cdef Edge *edge
        cdef np.int64_t ii, ind, dim
        if self._edges_y is None:
            dim = self._dim
            self._edges_y = np.empty((self.n_edges_y, dim), dtype=np.float64)
            gridEy = self._edges_y
            for it in self.tree.edges_y:
                edge = it.second
                if not edge.hanging:
                    ind = edge.index
                    for ii in range(dim):
                        gridEy[ind, ii] = edge.location[ii]
        return self._edges_y

    @property
    def hanging_edges_y(self):
        """
        Returns a numpy array of shape (n_hanging_edges_y, dim) with the centers of all
        hanging edges along the second dimension in order.
        """
        cdef np.float64_t[:, :] gridhEy
        cdef Edge *edge
        cdef np.int64_t ii, ind, dim
        if self._hanging_edges_y is None:
            dim = self._dim
            self._hanging_edges_y = np.empty((self.n_hanging_edges_y, dim), dtype=np.float64)
            gridhEy = self._hanging_edges_y
            for edge in self.tree.hanging_edges_y:
                ind = edge.index-self.n_edges_y
                for ii in range(dim):
                    gridhEy[ind, ii] = edge.location[ii]
        return self._hanging_edges_y

    @property
    def edges_z(self):
        """
        Returns a numpy array of shape (n_edges_z, dim) with the centers of all
        non-hanging edges along the third dimension in order.
        """
        cdef np.float64_t[:, :] gridEz
        cdef Edge *edge
        cdef np.int64_t ii, ind, dim
        if self._edges_z is None:
            dim = self._dim
            self._edges_z = np.empty((self.n_edges_z, dim), dtype=np.float64)
            gridEz = self._edges_z
            for it in self.tree.edges_z:
                edge = it.second
                if not edge.hanging:
                    ind = edge.index
                    for ii in range(dim):
                        gridEz[ind, ii] = edge.location[ii]
        return self._edges_z

    @property
    def hanging_edges_z(self):
        """
        Returns a numpy array of shape (n_hanging_edges_z, dim) with the centers of all
        hanging edges along the third dimension in order.
        """
        cdef np.float64_t[:, :] gridhEz
        cdef Edge *edge
        cdef np.int64_t ii, ind, dim
        if self._hanging_edges_z is None:
            dim = self._dim
            self._hanging_edges_z = np.empty((self.n_hanging_edges_z, dim), dtype=np.float64)
            gridhEz = self._hanging_edges_z
            for edge in self.tree.hanging_edges_z:
                ind = edge.index-self.n_edges_z
                for ii in range(dim):
                    gridhEz[ind, ii] = edge.location[ii]
        return self._hanging_edges_z

    @property
    def faces_x(self):
        """
        Returns a numpy array of shape (n_faces_x, dim) with the centers of all
        non-hanging faces along the first dimension in order.
        """
        if(self._dim == 2): return self.edges_y

        cdef np.float64_t[:, :] gridFx
        cdef Face *face
        cdef np.int64_t ii, ind, dim
        if self._faces_x is None:
            dim = self._dim
            self._faces_x = np.empty((self.n_faces_x, dim), dtype=np.float64)
            gridFx = self._faces_x
            for it in self.tree.faces_x:
                face = it.second
                if not face.hanging:
                    ind = face.index
                    for ii in range(dim):
                        gridFx[ind, ii] = face.location[ii]
        return self._faces_x

    @property
    def faces_y(self):
        """
        Returns a numpy array of shape (n_faces_y, dim) with the centers of all
        non-hanging faces along the second dimension in order.
        """
        if(self._dim == 2): return self.edges_x
        cdef np.float64_t[:, :] gridFy
        cdef Face *face
        cdef np.int64_t ii, ind, dim
        if self._faces_y is None:
            dim = self._dim
            self._faces_y = np.empty((self.n_faces_y, dim), dtype=np.float64)
            gridFy = self._faces_y
            for it in self.tree.faces_y:
                face = it.second
                if not face.hanging:
                    ind = face.index
                    for ii in range(dim):
                        gridFy[ind, ii] = face.location[ii]
        return self._faces_y

    @property
    def faces_z(self):
        """
        Returns a numpy array of shape (n_faces_z, dim) with the centers of all
        non-hanging faces along the third dimension in order.
        """
        if(self._dim == 2): return self.cell_centers

        cdef np.float64_t[:, :] gridFz
        cdef Face *face
        cdef np.int64_t ii, ind, dim
        if self._faces_z is None:
            dim = self._dim
            self._faces_z = np.empty((self.n_faces_z, dim), dtype=np.float64)
            gridFz = self._faces_z
            for it in self.tree.faces_z:
                face = it.second
                if not face.hanging:
                    ind = face.index
                    for ii in range(dim):
                        gridFz[ind, ii] = face.location[ii]
        return self._faces_z

    @property
    def hanging_faces_x(self):
        """
        Returns a numpy array of shape (n_hanging_faces_x, dim) with the centers of all
        hanging faces along the first dimension in order.
        """
        if(self._dim == 2): return self.hanging_edges_y

        cdef np.float64_t[:, :] gridFx
        cdef Face *face
        cdef np.int64_t ii, ind, dim
        if self._hanging_faces_x is None:
            dim = self._dim
            self._hanging_faces_x = np.empty((self.n_hanging_faces_x, dim), dtype=np.float64)
            gridhFx = self._hanging_faces_x
            for face in self.tree.hanging_faces_x:
                ind = face.index-self.n_faces_x
                for ii in range(dim):
                    gridhFx[ind, ii] = face.location[ii]
        return self._hanging_faces_x

    @property
    def hanging_faces_y(self):
        """
        Returns a numpy array of shape (n_hanging_faces_y, dim) with the centers of all
        hanging faces along the second dimension in order.
        """
        if(self._dim == 2): return self.hanging_edges_x

        cdef np.float64_t[:, :] gridhFy
        cdef Face *face
        cdef np.int64_t ii, ind, dim
        if self._hanging_faces_y is None:
            dim = self._dim
            self._hanging_faces_y = np.empty((self.n_hanging_faces_y, dim), dtype=np.float64)
            gridhFy = self._hanging_faces_y
            for face in self.tree.hanging_faces_y:
                ind = face.index-self.n_faces_y
                for ii in range(dim):
                    gridhFy[ind, ii] = face.location[ii]
        return self._hanging_faces_y

    @property
    def hanging_faces_z(self):
        """
        Returns a numpy array of shape (n_hanging_faces_z, dim) with the centers of all
        hanging faces along the third dimension in order.
        """
        if(self._dim == 2): return np.array([])

        cdef np.float64_t[:, :] gridhFz
        cdef Face *face
        cdef np.int64_t ii, ind, dim
        if self._hanging_faces_z is None:
            dim = self._dim
            self._hanging_faces_z = np.empty((self.n_hanging_faces_z, dim), dtype=np.float64)
            gridhFz = self._hanging_faces_z
            for face in self.tree.hanging_faces_z:
                ind = face.index-self.n_faces_z
                for ii in range(dim):
                    gridhFz[ind, ii] = face.location[ii]
        return self._hanging_faces_z

    @property
    def cell_volumes(self):
        """
        Returns a numpy array of length n_cells with the volumes (areas in 2D) of all
        cells in order.
        """
        cdef np.float64_t[:] vol
        if self._cell_volumes is None:
            self._cell_volumes = np.empty(self.n_cells, dtype=np.float64)
            vol = self._cell_volumes
            for cell in self.tree.cells:
                vol[cell.index] = cell.volume
        return self._cell_volumes

    @property
    def face_areas(self):
        """
        Returns a numpy array of length n_faces with the area (length in 2D) of all
        faces ordered by x, then y, then z.
        """
        if self._dim == 2 and self._face_areas is None:
            self._face_areas = np.r_[self.edge_lengths[self.n_edges_x:], self.edge_lengths[:self.n_edges_x]]
        cdef np.float64_t[:] area
        cdef int_t ind, offset = 0
        cdef Face *face
        if self._face_areas is None:
            self._face_areas = np.empty(self.n_faces, dtype=np.float64)
            area = self._face_areas

            for it in self.tree.faces_x:
                face = it.second
                if face.hanging: continue
                area[face.index] = face.area

            offset = self.n_faces_x
            for it in self.tree.faces_y:
                face = it.second
                if face.hanging: continue
                area[face.index + offset] = face.area

            offset = self.n_faces_x + self.n_faces_y
            for it in self.tree.faces_z:
                face = it.second
                if face.hanging: continue
                area[face.index + offset] = face.area
        return self._face_areas

    @property
    def edge_lengths(self):
        """
        Returns a numpy array of length n_edges with the length of all edges ordered
        by x, then y, then z.
        """
        cdef np.float64_t[:] edge_l
        cdef Edge *edge
        cdef int_t ind, offset
        if self._edge_lengths is None:
            self._edge_lengths = np.empty(self.n_edges, dtype=np.float64)
            edge_l = self._edge_lengths

            for it in self.tree.edges_x:
                edge = it.second
                if edge.hanging: continue
                edge_l[edge.index] = edge.length

            offset = self.n_edges_x
            for it in self.tree.edges_y:
                edge = it.second
                if edge.hanging: continue
                edge_l[edge.index + offset] = edge.length

            if self._dim > 2:
                offset = self.n_edges_x + self.n_edges_y
                for it in self.tree.edges_z:
                    edge = it.second
                    if edge.hanging: continue
                    edge_l[edge.index + offset] = edge.length
        return self._edge_lengths

    @property
    def cell_boundary_indices(self):
        """Returns a tuple of arrays of indexes for boundary cells in each direction
        xdown, xup, ydown, yup, zdown, zup
        """
        cdef np.int64_t[:] indxu, indxd, indyu, indyd, indzu, indzd
        indxu = np.empty(self.n_cells, dtype=np.int64)
        indxd = np.empty(self.n_cells, dtype=np.int64)
        indyu = np.empty(self.n_cells, dtype=np.int64)
        indyd = np.empty(self.n_cells, dtype=np.int64)
        if self._dim == 3:
            indzu = np.empty(self.n_cells, dtype=np.int64)
            indzd = np.empty(self.n_cells, dtype=np.int64)
        cdef int_t nxu, nxd, nyu, nyd, nzu, nzd
        nxu = 0
        nxd = 0
        nyu = 0
        nyd = 0
        nzu = 0
        nzd = 0
        for cell in self.tree.cells:
            if cell.neighbors[0] == NULL:
                indxd[nxd] = cell.index
                nxd += 1
            if cell.neighbors[1] == NULL:
                indxu[nxu] = cell.index
                nxu += 1
            if cell.neighbors[2] == NULL:
                indyd[nyd] = cell.index
                nyd += 1
            if cell.neighbors[3] == NULL:
                indyu[nyu] = cell.index
                nyu += 1
            if self._dim == 3:
                if cell.neighbors[4] == NULL:
                    indzd[nzd] = cell.index
                    nzd += 1
                if cell.neighbors[5] == NULL:
                    indzu[nzu] = cell.index
                    nzu += 1
        ixd = np.array(indxd)[:nxd]
        ixu = np.array(indxu)[:nxu]
        iyd = np.array(indyd)[:nyd]
        iyu = np.array(indyu)[:nyu]
        if self._dim == 3:
            izd = np.array(indzd)[:nzd]
            izu = np.array(indzu)[:nzu]
            return ixd, ixu, iyd, iyu, izd, izu
        else:
            return ixd, ixu, iyd, iyu

    @property
    def face_boundary_indices(self):
        """Returns a tuple of arrays of indexes for boundary faces in each direction
        xdown, xup, ydown, yup, zdown, zup
        """
        cell_boundary_inds = self.cell_boundary_indices
        cdef np.int64_t[:] c_indxu, c_indxd, c_indyu, c_indyd, c_indzu, c_indzd
        cdef np.int64_t[:] f_indxu, f_indxd, f_indyu, f_indyd, f_indzu, f_indzd
        if self._dim == 2:
            c_indxd, c_indxu, c_indyd, c_indyu = cell_boundary_inds
        else:
            c_indxd, c_indxu, c_indyd, c_indyu, c_indzd, c_indzu = cell_boundary_inds

        f_indxd = np.empty(c_indxd.shape[0], dtype=np.int64)
        f_indxu = np.empty(c_indxu.shape[0], dtype=np.int64)
        f_indyd = np.empty(c_indyd.shape[0], dtype=np.int64)
        f_indyu = np.empty(c_indyu.shape[0], dtype=np.int64)
        if self._dim == 2:
            for i in range(f_indxd.shape[0]):
                f_indxd[i] = self.tree.cells[c_indxd[i]].edges[2].index
            for i in range(f_indxu.shape[0]):
                f_indxu[i] = self.tree.cells[c_indxu[i]].edges[3].index
            for i in range(f_indyd.shape[0]):
                f_indyd[i] = self.tree.cells[c_indyd[i]].edges[0].index
            for i in range(f_indyu.shape[0]):
                f_indyu[i] = self.tree.cells[c_indyu[i]].edges[1].index
        if self._dim == 3:
            f_indzd = np.empty(c_indzd.shape[0], dtype=np.int64)
            f_indzu = np.empty(c_indzu.shape[0], dtype=np.int64)

            for i in range(f_indxd.shape[0]):
                f_indxd[i] = self.tree.cells[c_indxd[i]].faces[0].index
            for i in range(f_indxu.shape[0]):
                f_indxu[i] = self.tree.cells[c_indxu[i]].faces[1].index
            for i in range(f_indyd.shape[0]):
                f_indyd[i] = self.tree.cells[c_indyd[i]].faces[2].index
            for i in range(f_indyu.shape[0]):
                f_indyu[i] = self.tree.cells[c_indyu[i]].faces[3].index
            for i in range(f_indzd.shape[0]):
                f_indzd[i] = self.tree.cells[c_indzd[i]].faces[4].index
            for i in range(f_indzu.shape[0]):
                f_indzu[i] = self.tree.cells[c_indzu[i]].faces[5].index
        ixd = np.array(f_indxd)
        ixu = np.array(f_indxu)
        iyd = np.array(f_indyd)
        iyu = np.array(f_indyu)
        if self._dim == 3:
            izd = np.array(f_indzd)
            izu = np.array(f_indzu)
            return ixd, ixu, iyd, iyu, izd, izu
        else:
            return ixd, ixu, iyd, iyu

    def get_boundary_cells(self, active_ind=None, direction='zu'):
        """Returns the indices of boundary cells in a given direction given an active index array.

        Parameters
        ----------
        active_ind : array_like of bool, optional
            If not None, then this must show which cells are active
        direction: str, optional
            must be one of ('zu', 'zd', 'xu', 'xd', 'yu', 'yd')

        Returns
        -------
        numpy.array
            Array of indices for the boundary cells in the requested direction
        """

        direction = direction.lower()
        if direction[0] == 'z' and self._dim == 2:
            dir_str = 'y'+direction[1]
        else:
            dir_str = direction
        cdef int_t dir_ind = {'xd':0, 'xu':1, 'yd':2, 'yu':3, 'zd':4, 'zu':5}[dir_str]
        if active_ind is None:
            return self.cell_boundary_indices[dir_ind]

        active_ind = np.require(active_ind, dtype=np.int8, requirements='C')
        cdef np.int8_t[:] act = active_ind
        cdef np.int8_t[:] is_on_boundary = np.zeros(self.n_cells, dtype=np.int8)

        cdef c_Cell *cell
        cdef c_Cell *neighbor

        for cell in self.tree.cells:
            if not act[cell.index]:
                continue
            is_bound = 0
            neighbor = cell.neighbors[dir_ind]
            if neighbor is NULL:
                is_bound = 1
            elif neighbor.is_leaf():
                is_bound = not act[neighbor.index]
            else:
                if dir_ind == 1 or dir_ind == 3 or dir_ind == 5:
                    is_bound = is_bound or (not act[neighbor.children[0].index])
                if dir_ind == 0 or dir_ind == 3 or dir_ind == 5:
                    is_bound = is_bound or (not act[neighbor.children[1].index])
                if dir_ind == 1 or dir_ind == 2 or dir_ind == 5:
                    is_bound = is_bound or (not act[neighbor.children[2].index])
                if dir_ind == 0 or dir_ind == 2 or dir_ind == 5:
                    is_bound = is_bound or (not act[neighbor.children[3].index])

                if self._dim == 3:
                    if dir_ind == 1 or dir_ind == 3 or dir_ind == 4:
                        is_bound = is_bound or (not act[neighbor.children[4].index])
                    if dir_ind == 0 or dir_ind == 3 or dir_ind == 4:
                        is_bound = is_bound or (not act[neighbor.children[5].index])
                    if dir_ind == 1 or dir_ind == 2 or dir_ind == 4:
                        is_bound = is_bound or (not act[neighbor.children[6].index])
                    if dir_ind == 0 or dir_ind == 2 or dir_ind == 4:
                        is_bound = is_bound or (not act[neighbor.children[7].index])

            is_on_boundary[cell.index] = is_bound

        return np.where(is_on_boundary)

    @cython.cdivision(True)
    def get_cells_along_line(self, x0, x1):
        """Finds the cells along a line segment defined by two points

        Parameters
        ----------
        x0,x1 : array_like of length (dim)
            Begining and ending point of the line segment.

        Returns
        -------
        list of ints
            Indexes for cells that contain the a line defined by the two input
            points, ordered in the direction of the line.
        """
        cdef np.float64_t ax, ay, az, bx, by, bz

        cdef int dim = self.dim
        ax = x0[0]
        ay = x0[1]
        az = x0[2] if dim==3 else 0

        bx = x1[0]
        by = x1[1]
        bz = x1[2] if dim==3 else 0

        cdef vector[long long int] cell_indexes;

        #find initial cell
        cdef c_Cell *cur_cell = self.tree.containing_cell(ax, ay, az)
        cell_indexes.push_back(cur_cell.index)
        #find last cell
        cdef c_Cell *last_cell = self.tree.containing_cell(bx, by, bz)
        cdef c_Cell *next_cell
        cdef int ix, iy, iz
        cdef double tx, ty, tz, ipx, ipy, ipz

        if dim==3:
            last_point = 7
        else:
            last_point = 3

        cdef int iter = 0

        while cur_cell.index != last_cell.index:
            #find which direction to look:
            p0 = cur_cell.points[0].location
            pF = cur_cell.points[last_point].location

            if ax>bx:
                tx = (p0[0]-ax)/(bx-ax)
            elif ax<bx:
                tx = (pF[0]-ax)/(bx-ax)
            else:
                tx = INFINITY

            if ay>by:
                ty = (p0[1]-ay)/(by-ay)
            elif ay<by:
                ty = (pF[1]-ay)/(by-ay)
            else:
                ty = INFINITY

            if az>bz:
                tz = (p0[2]-az)/(bz-az)
            elif az<bz:
                tz = (pF[2]-az)/(bz-az)
            else:
                tz = INFINITY

            t = min(tx,ty,tz)

            #intersection point
            ipx = (bx-ax)*t+ax
            ipy = (by-ay)*t+ay
            ipz = (bz-az)*t+az

            next_cell = cur_cell
            if tx<=ty and tx<=tz:
                # step in x direction
                if ax>bx: # go -x
                    next_cell = next_cell.neighbors[0]
                else: # go +x
                    next_cell = next_cell.neighbors[1]
            if ty<=tx and ty<=tz:
                # step in y direction
                if ay>by: # go -y
                    next_cell = next_cell.neighbors[2]
                else: # go +y
                    next_cell = next_cell.neighbors[3]
            if dim==3 and tz<=tx and tz<=ty:
                # step in z direction
                if az>bz: # go -z
                    next_cell = next_cell.neighbors[4]
                else: # go +z
                    next_cell = next_cell.neighbors[5]

            # check if next_cell is not a leaf
            # (if so need to traverse down the children and find the closest leaf cell)
            while not next_cell.is_leaf():
                # should be able to use cp to check which cell to go to
                cp = next_cell.children[0].points[last_point].location
                # this basically finds the child cell closest to the intersection point
                ix = ipx>cp[0] or (ipx==cp[0] and ax<bx)
                iy = ipy>cp[1] or (ipy==cp[1] and ay<by)
                iz = dim==3 and (ipz>cp[2] or  (ipz==cp[2] and az<bz))
                next_cell = next_cell.children[ix + 2*iy + 4*iz]

            #this now should have stepped appropriately across diagonals and such

            cur_cell = next_cell
            cell_indexes.push_back(cur_cell.index)
            if cur_cell.index == -1:
                raise Exception('Path not found')
        return cell_indexes

    @property
    def face_divergence(self):
        """
        Construct divergence operator (face-stg to cell-centres).
        """
        if self._face_divergence is not None:
            return self._face_divergence
        if self._dim == 2:
            D = self._face_divergence_2D() # Because it uses edges instead of faces
        else:
            D = self._face_divergence_3D()
        R = self._deflate_faces()
        self._face_divergence = D*R
        return self._face_divergence

    @cython.cdivision(True)
    @cython.boundscheck(False)
    def _face_divergence_2D(self):
        cdef np.int64_t[:] I = np.empty(self.n_cells*4, dtype=np.int64)
        cdef np.int64_t[:] J = np.empty(self.n_cells*4, dtype=np.int64)
        cdef np.float64_t[:] V = np.empty(self.n_cells*4, dtype=np.float64)

        cdef np.int64_t i = 0
        cdef Edge *edges[4]
        cdef np.int64_t offset = self.tree.edges_y.size()
        cdef double volume

        for cell in self.tree.cells:
            edges = cell.edges
            i = cell.index
            I[i*4 : i*4 + 4] = i
            J[i*4    ] = edges[0].index + offset #x edge, y face (add offset)
            J[i*4 + 1] = edges[1].index + offset #x edge, y face (add offset)
            J[i*4 + 2] = edges[2].index #y edge, x face
            J[i*4 + 3] = edges[3].index #y edge, x face

            volume = cell.volume
            V[i*4    ] = -edges[0].length/volume
            V[i*4 + 1] =  edges[1].length/volume
            V[i*4 + 2] = -edges[2].length/volume
            V[i*4 + 3] =  edges[3].length/volume
        return sp.csr_matrix((V, (I, J)))

    @cython.cdivision(True)
    @cython.boundscheck(False)
    def _face_divergence_3D(self):
        cdef:
            np.int64_t[:] I = np.empty(self.n_cells*6, dtype=np.int64)
            np.int64_t[:] J = np.empty(self.n_cells*6, dtype=np.int64)
            np.float64_t[:] V = np.empty(self.n_cells*6, dtype=np.float64)

            np.int64_t i = 0
            Face *faces[6]
            np.int64_t offset1 = self.tree.faces_x.size()
            np.int64_t offset2 = offset1 + self.tree.faces_y.size()
            double volume, fx_area, fy_area, fz_area

        for cell in self.tree.cells:
            faces = cell.faces
            i = cell.index
            I[i*6 : i*6 + 6] = i
            J[i*6    ] = faces[0].index #x1 face
            J[i*6 + 1] = faces[1].index #x2 face
            J[i*6 + 2] = faces[2].index + offset1 #y face (add offset1)
            J[i*6 + 3] = faces[3].index + offset1 #y face (add offset1)
            J[i*6 + 4] = faces[4].index + offset2 #z face (add offset2)
            J[i*6 + 5] = faces[5].index + offset2 #z face (add offset2)

            volume = cell.volume
            fx_area = faces[0].area
            fy_area = faces[2].area
            fz_area = faces[4].area
            V[i*6    ] = -fx_area/volume
            V[i*6 + 1] =  fx_area/volume
            V[i*6 + 2] = -fy_area/volume
            V[i*6 + 3] =  fy_area/volume
            V[i*6 + 4] = -fz_area/volume
            V[i*6 + 5] =  fz_area/volume
        return sp.csr_matrix((V, (I, J)))

    @property
    @cython.cdivision(True)
    @cython.boundscheck(False)
    def edge_curl(self):
        """
        Construct the 3D curl operator.
        """
        if self._edge_curl is not None:
            return self._edge_curl
        cdef:
            int_t dim = self._dim
            np.int64_t[:] I = np.empty(4*self.n_faces, dtype=np.int64)
            np.int64_t[:] J = np.empty(4*self.n_faces, dtype=np.int64)
            np.float64_t[:] V = np.empty(4*self.n_faces, dtype=np.float64)
            Face *face
            int_t ii
            int_t face_offset_y = self.n_faces_x
            int_t face_offset_z = self.n_faces_x + self.n_faces_y
            int_t edge_offset_y = self.n_total_edges_x
            int_t edge_offset_z = self.n_total_edges_x + self.n_total_edges_y
            double area

        for it in self.tree.faces_x:
            face = it.second
            if face.hanging:
                continue
            ii = face.index
            I[4*ii : 4*ii + 4] = ii
            J[4*ii    ] = face.edges[0].index + edge_offset_z
            J[4*ii + 1] = face.edges[1].index + edge_offset_y
            J[4*ii + 2] = face.edges[2].index + edge_offset_z
            J[4*ii + 3] = face.edges[3].index + edge_offset_y

            area = face.area
            V[4*ii    ] = -face.edges[0].length/area
            V[4*ii + 1] = -face.edges[1].length/area
            V[4*ii + 2] =  face.edges[2].length/area
            V[4*ii + 3] =  face.edges[3].length/area

        for it in self.tree.faces_y:
            face = it.second
            if face.hanging:
                continue
            ii = face.index + face_offset_y
            I[4*ii : 4*ii + 4] = ii
            J[4*ii    ] = face.edges[0].index + edge_offset_z
            J[4*ii + 1] = face.edges[1].index
            J[4*ii + 2] = face.edges[2].index + edge_offset_z
            J[4*ii + 3] = face.edges[3].index

            area = face.area
            V[4*ii    ] =  face.edges[0].length/area
            V[4*ii + 1] =  face.edges[1].length/area
            V[4*ii + 2] = -face.edges[2].length/area
            V[4*ii + 3] = -face.edges[3].length/area

        for it in self.tree.faces_z:
            face = it.second
            if face.hanging:
                continue
            ii = face.index + face_offset_z
            I[4*ii : 4*ii + 4] = ii
            J[4*ii    ] = face.edges[0].index + edge_offset_y
            J[4*ii + 1] = face.edges[1].index
            J[4*ii + 2] = face.edges[2].index + edge_offset_y
            J[4*ii + 3] = face.edges[3].index

            area = face.area
            V[4*ii    ] = -face.edges[0].length/area
            V[4*ii + 1] = -face.edges[1].length/area
            V[4*ii + 2] =  face.edges[2].length/area
            V[4*ii + 3] =  face.edges[3].length/area

        C = sp.csr_matrix((V, (I, J)),shape=(self.n_faces, self.n_total_edges))
        R = self._deflate_edges()
        self._edge_curl = C*R
        return self._edge_curl

    @property
    @cython.cdivision(True)
    @cython.boundscheck(False)
    def nodal_gradient(self):
        """
        Construct gradient operator (nodes to edges).
        """
        if self._nodal_gradient is not None:
            return self._nodal_gradient
        cdef:
            int_t dim = self._dim
            np.int64_t[:] I = np.empty(2*self.n_edges, dtype=np.int64)
            np.int64_t[:] J = np.empty(2*self.n_edges, dtype=np.int64)
            np.float64_t[:] V = np.empty(2*self.n_edges, dtype=np.float64)
            Edge *edge
            double length
            int_t ii
            np.int64_t offset1 = self.n_edges_x
            np.int64_t offset2 = offset1 + self.n_edges_y

        for it in self.tree.edges_x:
            edge = it.second
            if edge.hanging: continue
            ii = edge.index
            I[ii*2 : ii*2 + 2] = ii
            J[ii*2    ] = edge.points[0].index
            J[ii*2 + 1] = edge.points[1].index

            length = edge.length
            V[ii*2    ] = -1.0/length
            V[ii*2 + 1] =  1.0/length

        for it in self.tree.edges_y:
            edge = it.second
            if edge.hanging: continue
            ii = edge.index + offset1
            I[ii*2 : ii*2 + 2] = ii
            J[ii*2    ] = edge.points[0].index
            J[ii*2 + 1] = edge.points[1].index

            length = edge.length
            V[ii*2    ] = -1.0/length
            V[ii*2 + 1] =  1.0/length

        if(dim>2):
            for it in self.tree.edges_z:
                edge = it.second
                if edge.hanging: continue
                ii = edge.index + offset2
                I[ii*2 : ii*2 + 2] = ii
                J[ii*2    ] = edge.points[0].index
                J[ii*2 + 1] = edge.points[1].index

                length = edge.length
                V[ii*2    ] = -1.0/length
                V[ii*2 + 1] =  1.0/length


        Rn = self._deflate_nodes()
        G = sp.csr_matrix((V, (I, J)), shape=(self.n_edges, self.n_total_nodes))
        self._nodal_gradient = G*Rn
        return self._nodal_gradient

    @property
    def nodal_laplacian(self):
        raise NotImplementedError('Nodal Laplacian has not been implemented for TreeMesh')

    @cython.boundscheck(False)
    def average_cell_to_total_face_x(self):
        """Average matrix for cell center to total (including hanging) x faces"""
        cdef np.int64_t[:] I = np.zeros(2*self.n_total_faces_x, dtype=np.int64)
        cdef np.int64_t[:] J = np.zeros(2*self.n_total_faces_x, dtype=np.int64)
        cdef np.float64_t[:] V = np.zeros(2*self.n_total_faces_x, dtype=np.float64)
        cdef int dim = self._dim
        cdef int_t ind

        for cell in self.tree.cells :
            next_cell = cell.neighbors[1]
            if next_cell == NULL:
                continue
            if dim == 2:
                if next_cell.is_leaf():
                    ind = cell.edges[3].index
                    I[2*ind    ] = ind
                    I[2*ind + 1] = ind
                    J[2*ind    ] = cell.index
                    J[2*ind + 1] = next_cell.index
                    V[2*ind    ] = 0.5
                    V[2*ind + 1] = 0.5
                else:
                    for i in range(2): # two neighbors in +x direction
                        ind = next_cell.children[2*i].edges[2].index
                        I[2*ind    ] = ind
                        I[2*ind + 1] = ind
                        J[2*ind    ] = cell.index
                        J[2*ind + 1] = next_cell.children[2*i].index
                        V[2*ind    ] = 0.5
                        V[2*ind + 1] = 0.5
            else:
                if cell.neighbors[1].is_leaf():
                    ind = cell.faces[1].index
                    I[2*ind    ] = ind
                    I[2*ind + 1] = ind
                    J[2*ind    ] = cell.index
                    J[2*ind + 1] = next_cell.index
                    V[2*ind    ] = 0.5
                    V[2*ind + 1] = 0.5
                else:
                    for i in range(4): # four neighbors in +x direction
                        ind = next_cell.children[2*i].faces[0].index
                        I[2*ind    ] = ind
                        I[2*ind + 1] = ind
                        J[2*ind    ] = cell.index
                        J[2*ind + 1] = next_cell.children[2*i].index
                        V[2*ind    ] = 0.5
                        V[2*ind + 1] = 0.5

        return sp.csr_matrix((V, (I,J)), shape=(self.n_total_faces_x, self.n_cells))

    @cython.boundscheck(False)
    def average_cell_to_total_face_y(self):
        """Average matrix for cell center to total (including hanging) y faces"""
        cdef np.int64_t[:] I = np.zeros(2*self.n_total_faces_y, dtype=np.int64)
        cdef np.int64_t[:] J = np.zeros(2*self.n_total_faces_y, dtype=np.int64)
        cdef np.float64_t[:] V = np.zeros(2*self.n_total_faces_y, dtype=np.float64)
        cdef int dim = self._dim
        cdef int_t ind

        for cell in self.tree.cells :
            next_cell = cell.neighbors[3]
            if next_cell==NULL:
                continue
            if dim==2:
                if next_cell.is_leaf():
                    ind = cell.edges[1].index
                    I[2*ind    ] = ind
                    I[2*ind + 1] = ind
                    J[2*ind    ] = cell.index
                    J[2*ind + 1] = next_cell.index
                    V[2*ind    ] = 0.5
                    V[2*ind + 1] = 0.5
                else:
                    for i in range(2): # two neighbors in +y direction
                        ind = next_cell.children[i].edges[0].index
                        I[2*ind    ] = ind
                        I[2*ind + 1] = ind
                        J[2*ind    ] = cell.index
                        J[2*ind + 1] = next_cell.children[i].index
                        V[2*ind    ] = 0.5
                        V[2*ind + 1] = 0.5
            else:
                if next_cell.is_leaf():
                    ind = cell.faces[3].index
                    I[2*ind    ] = ind
                    I[2*ind + 1] = ind
                    J[2*ind    ] = cell.index
                    J[2*ind + 1] = next_cell.index
                    V[2*ind    ] = 0.5
                    V[2*ind + 1] = 0.5
                else:
                    for i in range(4): # four neighbors in +y direction
                        ind = next_cell.children[(i>>1)*4 + i%2].faces[2].index
                        I[2*ind    ] = ind
                        I[2*ind + 1] = ind
                        J[2*ind    ] = cell.index
                        J[2*ind + 1] = next_cell.children[(i>>1)*4 + i%2].index
                        V[2*ind    ] = 0.5
                        V[2*ind + 1] = 0.5
        return sp.csr_matrix((V, (I,J)), shape=(self.n_total_faces_y, self.n_cells))

    @cython.boundscheck(False)
    def average_cell_to_total_face_z(self):
        """Average matrix for cell center to total (including hanging) z faces"""
        cdef np.int64_t[:] I = np.zeros(2*self.n_total_faces_z, dtype=np.int64)
        cdef np.int64_t[:] J = np.zeros(2*self.n_total_faces_z, dtype=np.int64)
        cdef np.float64_t[:] V = np.zeros(2*self.n_total_faces_z, dtype=np.float64)
        cdef int_t ind

        for cell in self.tree.cells :
            next_cell = cell.neighbors[5]
            if next_cell==NULL:
                continue
            if next_cell.is_leaf():
                ind = cell.faces[5].index
                I[2*ind    ] = ind
                I[2*ind + 1] = ind
                J[2*ind    ] = cell.index
                J[2*ind + 1] = next_cell.index
                V[2*ind    ] = 0.5
                V[2*ind + 1] = 0.5
            else:
                for i in range(4): # four neighbors in +z direction
                    ind = next_cell.children[i].faces[4].index
                    I[2*ind    ] = ind
                    I[2*ind + 1] = ind
                    J[2*ind    ] = cell.index
                    J[2*ind + 1] = next_cell.children[i].index
                    V[2*ind    ] = 0.5
                    V[2*ind + 1] = 0.5

        return sp.csr_matrix((V, (I,J)), shape=(self.n_total_faces_z, self.n_cells))

    @property
    @cython.boundscheck(False)
    def stencil_cell_gradient_x(self):
        """Cell gradient stencil matrix to total (including hanging) x faces"""
        if getattr(self, '_stencil_cell_gradient_x', None) is not None:
            return self._stencil_cell_gradient_x
        cdef np.int64_t[:] I = np.zeros(2*self.n_total_faces_x, dtype=np.int64)
        cdef np.int64_t[:] J = np.zeros(2*self.n_total_faces_x, dtype=np.int64)
        cdef np.float64_t[:] V = np.zeros(2*self.n_total_faces_x, dtype=np.float64)
        cdef int dim = self._dim
        cdef int_t ind

        for cell in self.tree.cells :
            next_cell = cell.neighbors[1]
            if next_cell == NULL:
                continue
            if dim == 2:
                if next_cell.is_leaf():
                    ind = cell.edges[3].index
                    I[2*ind    ] = ind
                    I[2*ind + 1] = ind
                    J[2*ind    ] = cell.index
                    J[2*ind + 1] = next_cell.index
                    V[2*ind    ] = -1.0
                    V[2*ind + 1] =  1.0
                else:
                    for i in range(2): # two neighbors in +x direction
                        ind = next_cell.children[2*i].edges[2].index
                        I[2*ind    ] = ind
                        I[2*ind + 1] = ind
                        J[2*ind    ] = cell.index
                        J[2*ind + 1] = next_cell.children[2*i].index
                        V[2*ind    ] = -1.0
                        V[2*ind + 1] =  1.0
            else:
                if cell.neighbors[1].is_leaf():
                    ind = cell.faces[1].index
                    I[2*ind    ] = ind
                    I[2*ind + 1] = ind
                    J[2*ind    ] = cell.index
                    J[2*ind + 1] = next_cell.index
                    V[2*ind    ] = -1.0
                    V[2*ind + 1] =  1.0
                else:
                    for i in range(4): # four neighbors in +x direction
                        ind = next_cell.children[2*i].faces[0].index #0 2 4 6
                        I[2*ind    ] = ind
                        I[2*ind + 1] = ind
                        J[2*ind    ] = cell.index
                        J[2*ind + 1] = next_cell.children[2*i].index
                        V[2*ind    ] = -1.0
                        V[2*ind + 1] =  1.0

        self._stencil_cell_gradient_x = (
            sp.csr_matrix((V, (I,J)), shape=(self.n_total_faces_x, self.n_cells))
        )
        return self._stencil_cell_gradient_x

    @property
    @cython.boundscheck(False)
    def stencil_cell_gradient_y(self):
        """Cell gradient stencil matrix to total (including hanging) y faces"""
        if getattr(self, '_stencil_cell_gradient_y', None) is not None:
            return self._stencil_cell_gradient_y

        cdef np.int64_t[:] I = np.zeros(2*self.n_total_faces_y, dtype=np.int64)
        cdef np.int64_t[:] J = np.zeros(2*self.n_total_faces_y, dtype=np.int64)
        cdef np.float64_t[:] V = np.zeros(2*self.n_total_faces_y, dtype=np.float64)
        cdef int dim = self._dim
        cdef int_t ind

        for cell in self.tree.cells :
            next_cell = cell.neighbors[3]
            if next_cell == NULL:
                continue
            if dim==2:
                if next_cell.is_leaf():
                    ind = cell.edges[1].index
                    I[2*ind    ] = ind
                    I[2*ind + 1] = ind
                    J[2*ind    ] = cell.index
                    J[2*ind + 1] = next_cell.index
                    V[2*ind    ] = -1.0
                    V[2*ind + 1] =  1.0
                else:
                    for i in range(2): # two neighbors in +y direction
                        ind = next_cell.children[i].edges[0].index
                        I[2*ind    ] = ind
                        I[2*ind + 1] = ind
                        J[2*ind    ] = cell.index
                        J[2*ind + 1] = next_cell.children[i].index
                        V[2*ind    ] = -1.0
                        V[2*ind + 1] =  1.0
            else:
                if next_cell.is_leaf():
                    ind = cell.faces[3].index
                    I[2*ind    ] = ind
                    I[2*ind + 1] = ind
                    J[2*ind    ] = cell.index
                    J[2*ind + 1] = next_cell.index
                    V[2*ind    ] = -1.0
                    V[2*ind + 1] =  1.0
                else:
                    for i in range(4): # four neighbors in +y direction
                        ind = next_cell.children[(i>>1)*4 + i%2].faces[2].index #0, 1, 4, 5
                        I[2*ind    ] = ind
                        I[2*ind + 1] = ind
                        J[2*ind    ] = cell.index
                        J[2*ind + 1] = next_cell.children[(i>>1)*4 + i%2].index
                        V[2*ind    ] = -1.0
                        V[2*ind + 1] = 1.0

        self._stencil_cell_gradient_y = (
            sp.csr_matrix((V, (I,J)), shape=(self.n_total_faces_y, self.n_cells))
        )
        return self._stencil_cell_gradient_y

    @property
    @cython.boundscheck(False)
    def stencil_cell_gradient_z(self):
        """Cell gradient stencil matrix to total (including hanging) z faces"""
        if getattr(self, '_stencil_cell_gradient_z', None) is not None:
            return self._stencil_cell_gradient_z

        cdef np.int64_t[:] I = np.zeros(2*self.n_total_faces_z, dtype=np.int64)
        cdef np.int64_t[:] J = np.zeros(2*self.n_total_faces_z, dtype=np.int64)
        cdef np.float64_t[:] V = np.zeros(2*self.n_total_faces_z, dtype=np.float64)
        cdef int_t ind

        for cell in self.tree.cells :
            next_cell = cell.neighbors[5]
            if next_cell==NULL:
                continue
            if next_cell.is_leaf():
                ind = cell.faces[5].index
                I[2*ind    ] = ind
                I[2*ind + 1] = ind
                J[2*ind    ] = cell.index
                J[2*ind + 1] = next_cell.index
                V[2*ind    ] = -1.0
                V[2*ind + 1] =  1.0
            else:
                for i in range(4): # four neighbors in +z direction
                    ind = next_cell.children[i].faces[4].index #0, 1, 2, 3
                    I[2*ind    ] = ind
                    I[2*ind + 1] = ind
                    J[2*ind    ] = cell.index
                    J[2*ind + 1] = next_cell.children[i].index
                    V[2*ind    ] = -1.0
                    V[2*ind + 1] =  1.0

        self._stencil_cell_gradient_z = (
            sp.csr_matrix((V, (I,J)), shape=(self.n_total_faces_z, self.n_cells))
        )
        return self._stencil_cell_gradient_z

    @cython.boundscheck(False)
    def _deflate_edges_x(self):
        #I is output index (with hanging)
        #J is input index (without hanging)
        cdef np.int64_t[:] I = np.empty(2*self.n_total_edges_x, dtype=np.int64)
        cdef np.int64_t[:] J = np.empty(2*self.n_total_edges_x, dtype=np.int64)
        cdef np.float64_t[:] V = np.empty(2*self.n_total_edges_x, dtype=np.float64)
        cdef Edge *edge
        cdef np.int64_t ii
        #x edges:
        for it in self.tree.edges_x:
            edge = it.second
            ii = edge.index
            I[2*ii    ] = ii
            I[2*ii + 1] = ii
            if edge.hanging:
                J[2*ii    ] = edge.parents[0].index
                J[2*ii + 1] = edge.parents[1].index
            else:
                J[2*ii    ] = ii
                J[2*ii + 1] = ii
            V[2*ii    ] = 0.5
            V[2*ii + 1] = 0.5
        Rh = sp.csr_matrix((V, (I, J)), shape=(self.n_total_edges_x, self.n_total_edges_x))
        # Test if it needs to be deflated again, (if any parents were also hanging)
        last_ind = max(np.nonzero(Rh.getnnz(0)>0)[0][-1], self.n_edges_x)
        while(last_ind > self.n_edges_x):
            Rh = Rh*Rh
            last_ind = max(np.nonzero(Rh.getnnz(0)>0)[0][-1], self.n_edges_x)
        Rh = Rh[:, : last_ind]
        return Rh

    @cython.boundscheck(False)
    def _deflate_edges_y(self):
        #I is output index (with hanging)
        #J is input index (without hanging)
        cdef int_t dim = self._dim
        cdef np.int64_t[:] I = np.empty(2*self.n_total_edges_y, dtype=np.int64)
        cdef np.int64_t[:] J = np.empty(2*self.n_total_edges_y, dtype=np.int64)
        cdef np.float64_t[:] V = np.empty(2*self.n_total_edges_y, dtype=np.float64)
        cdef Edge *edge
        cdef np.int64_t ii
        #x edges:
        for it in self.tree.edges_y:
            edge = it.second
            ii = edge.index
            I[2*ii    ] = ii
            I[2*ii + 1] = ii
            if edge.hanging:
                J[2*ii    ] = edge.parents[0].index
                J[2*ii + 1] = edge.parents[1].index
            else:
                J[2*ii    ] = ii
                J[2*ii + 1] = ii
            V[2*ii    ] = 0.5
            V[2*ii + 1] = 0.5
        Rh = sp.csr_matrix((V, (I, J)), shape=(self.n_total_edges_y, self.n_total_edges_y))
        # Test if it needs to be deflated again, (if any parents were also hanging)
        last_ind = max(np.nonzero(Rh.getnnz(0)>0)[0][-1], self.n_edges_y)
        while(last_ind > self.n_edges_y):
            Rh = Rh*Rh
            last_ind = max(np.nonzero(Rh.getnnz(0)>0)[0][-1], self.n_edges_y)
        Rh = Rh[:, : last_ind]
        return Rh

    @cython.boundscheck(False)
    def _deflate_edges_z(self):
        #I is output index (with hanging)
        #J is input index (without hanging)
        cdef int_t dim = self._dim
        cdef np.int64_t[:] I = np.empty(2*self.n_total_edges_z, dtype=np.int64)
        cdef np.int64_t[:] J = np.empty(2*self.n_total_edges_z, dtype=np.int64)
        cdef np.float64_t[:] V = np.empty(2*self.n_total_edges_z, dtype=np.float64)
        cdef Edge *edge
        cdef np.int64_t ii
        #x edges:
        for it in self.tree.edges_z:
            edge = it.second
            ii = edge.index
            I[2*ii    ] = ii
            I[2*ii + 1] = ii
            if edge.hanging:
                J[2*ii    ] = edge.parents[0].index
                J[2*ii + 1] = edge.parents[1].index
            else:
                J[2*ii    ] = ii
                J[2*ii + 1] = ii
            V[2*ii    ] = 0.5
            V[2*ii + 1] = 0.5
        Rh = sp.csr_matrix((V, (I, J)), shape=(self.n_total_edges_z, self.n_total_edges_z))
        # Test if it needs to be deflated again, (if any parents were also hanging)
        last_ind = max(np.nonzero(Rh.getnnz(0)>0)[0][-1], self.n_edges_z)
        while(last_ind > self.n_edges_z):
            Rh = Rh*Rh
            last_ind = max(np.nonzero(Rh.getnnz(0)>0)[0][-1], self.n_edges_z)
        Rh = Rh[:, : last_ind]
        return Rh

    def _deflate_edges(self):
        """Returns a matrix to remove hanging edges.
        A hanging edge can either have 1 or 2 parents.
        If a hanging edge has a single parent, it's value is the same as the parent
        If a hanging edge has 2 parents, it's an average of the two parents
        """
        if self._dim == 2:
            Rx = self._deflate_edges_x()
            Ry = self._deflate_edges_y()
            return sp.block_diag((Rx, Ry))
        else:
            Rx = self._deflate_edges_x()
            Ry = self._deflate_edges_y()
            Rz = self._deflate_edges_z()
            return sp.block_diag((Rx, Ry, Rz))

    def _deflate_faces(self):
        """ Returns a matrix that removes hanging faces
        The operation assigns the hanging face the value of its parent.
        A hanging face will only ever have 1 parent.
        """
        if(self._dim == 2):
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
        cdef np.int64_t[:] I = np.empty(self.n_total_faces_x, dtype=np.int64)
        cdef np.int64_t[:] J = np.empty(self.n_total_faces_x, dtype=np.int64)
        cdef np.float64_t[:] V = np.empty(self.n_total_faces_x, dtype=np.float64)
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
        cdef np.int64_t[:] I = np.empty(self.n_total_faces_y, dtype=np.int64)
        cdef np.int64_t[:] J = np.empty(self.n_total_faces_y, dtype=np.int64)
        cdef np.float64_t[:] V = np.empty(self.n_total_faces_y, dtype=np.float64)
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
        cdef np.int64_t[:] I = np.empty(self.n_total_faces_z, dtype=np.int64)
        cdef np.int64_t[:] J = np.empty(self.n_total_faces_z, dtype=np.int64)
        cdef np.float64_t[:] V = np.empty(self.n_total_faces_z, dtype=np.float64)
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
        """ Returns a matrix that removes hanging faces
        A hanging node will have 2 parents in 2D or 2 or 4 parents in 3D.
        This matrix assigns the hanging node the average value of its parents.
        """
        cdef np.int64_t[:] I = np.empty(4*self.n_total_nodes, dtype=np.int64)
        cdef np.int64_t[:] J = np.empty(4*self.n_total_nodes, dtype=np.int64)
        cdef np.float64_t[:] V = np.empty(4*self.n_total_nodes, dtype=np.float64)

        # I is output index
        # J is input index
        cdef Node *node
        cdef np.int64_t ii, i, offset
        offset = self.n_nodes
        cdef double[4] weights

        for it in self.tree.nodes:
            node = it.second
            ii = node.index
            I[4*ii:4*ii + 4] = ii
            if node.hanging:
                J[4*ii    ] = node.parents[0].index
                J[4*ii + 1] = node.parents[1].index
                J[4*ii + 2] = node.parents[2].index
                J[4*ii + 3] = node.parents[3].index
            else:
                J[4*ii : 4*ii + 4] = ii
            V[4*ii : 4*ii + 4] = 0.25;

        Rh = sp.csr_matrix((V, (I, J)), shape=(self.n_total_nodes, self.n_total_nodes))
        # Test if it needs to be deflated again, (if any parents were also hanging)
        last_ind = max(np.nonzero(Rh.getnnz(0)>0)[0][-1], self.n_nodes)
        while(last_ind > self.n_nodes):
            Rh = Rh*Rh
            last_ind = max(np.nonzero(Rh.getnnz(0)>0)[0][-1], self.n_nodes)
        Rh = Rh[:, : last_ind]
        return Rh

    @property
    @cython.boundscheck(False)
    def average_edge_x_to_cell(self):
        """
        Construct the averaging operator on cell edges in the x direction to
        cell centers.
        """
        if self._average_edge_x_to_cell is not None:
            return self._average_edge_x_to_cell
        cdef np.int64_t[:] I,J
        cdef np.float64_t[:] V
        cdef np.int64_t ind, ii, n_epc
        cdef double scale

        n_epc = 2*(self._dim-1)
        I = np.empty(self.n_cells*n_epc, dtype=np.int64)
        J = np.empty(self.n_cells*n_epc, dtype=np.int64)
        V = np.empty(self.n_cells*n_epc, dtype=np.float64)
        scale = 1.0/n_epc
        for cell in self.tree.cells:
            ind = cell.index
            for ii in range(n_epc):
                I[ind*n_epc + ii] = ind
                J[ind*n_epc + ii] = cell.edges[ii].index
                V[ind*n_epc + ii] = scale

        Rex = self._deflate_edges_x()
        self._average_edge_x_to_cell = sp.csr_matrix((V, (I, J)))*Rex
        return self._average_edge_x_to_cell

    @property
    @cython.boundscheck(False)
    def average_edge_y_to_cell(self):
        """
        Construct the averaging operator on cell edges in the y direction to
        cell centers.
        """
        if self._average_edge_y_to_cell is not None:
            return self._average_edge_y_to_cell
        cdef np.int64_t[:] I,J
        cdef np.float64_t[:] V
        cdef np.int64_t ind, ii, n_epc
        cdef double scale

        n_epc = 2*(self._dim-1)
        I = np.empty(self.n_cells*n_epc, dtype=np.int64)
        J = np.empty(self.n_cells*n_epc, dtype=np.int64)
        V = np.empty(self.n_cells*n_epc, dtype=np.float64)
        scale = 1.0/n_epc
        for cell in self.tree.cells:
            ind = cell.index
            for ii in range(n_epc):
                I[ind*n_epc + ii] = ind
                J[ind*n_epc + ii] = cell.edges[n_epc + ii].index #y edges
                V[ind*n_epc + ii] = scale

        Rey = self._deflate_edges_y()
        self._average_edge_y_to_cell = sp.csr_matrix((V, (I, J)))*Rey
        return self._average_edge_y_to_cell

    @property
    @cython.boundscheck(False)
    def average_edge_z_to_cell(self):
        """
        Construct the averaging operator on cell edges in the z direction to
        cell centers.
        """
        if self._average_edge_z_to_cell is not None:
            return self._average_edge_z_to_cell
        if self._dim == 2:
            raise Exception('There are no z-edges in 2D')
        cdef np.int64_t[:] I,J
        cdef np.float64_t[:] V
        cdef np.int64_t ind, ii, n_epc
        cdef double scale

        n_epc = 2*(self._dim-1)
        I = np.empty(self.n_cells*n_epc, dtype=np.int64)
        J = np.empty(self.n_cells*n_epc, dtype=np.int64)
        V = np.empty(self.n_cells*n_epc, dtype=np.float64)
        scale = 1.0/n_epc
        for cell in self.tree.cells:
            ind = cell.index
            for ii in range(n_epc):
                I[ind*n_epc + ii] = ind
                J[ind*n_epc + ii] = cell.edges[ii + 2*n_epc].index
                V[ind*n_epc + ii] = scale

        Rez = self._deflate_edges_z()
        self._average_edge_z_to_cell = sp.csr_matrix((V, (I, J)))*Rez
        return self._average_edge_z_to_cell

    @property
    def average_edge_to_cell(self):
        "Construct the averaging operator on cell edges to cell centers."
        if self._average_edge_to_cell is None:
            stacks = [self.average_edge_x_to_cell, self.average_edge_y_to_cell]
            if self._dim == 3:
                stacks += [self.average_edge_z_to_cell]
            self._average_edge_to_cell = 1.0/self._dim * sp.hstack(stacks).tocsr()
        return self._average_edge_to_cell

    @property
    def average_edge_to_cell_vector(self):
        "Construct the averaging operator on cell edges to cell centers."
        if self._average_edge_to_cell_vector is None:
            stacks = [self.average_edge_x_to_cell, self.average_edge_y_to_cell]
            if self._dim == 3:
                stacks += [self.average_edge_z_to_cell]
            self._average_edge_to_cell_vector = sp.block_diag(stacks).tocsr()
        return self._average_edge_to_cell_vector

    @property
    @cython.boundscheck(False)
    def average_face_x_to_cell(self):
        """
        Construct the averaging operator on cell faces in the x direction to
        cell centers.
        """
        if self._average_face_x_to_cell is not None:
            return self._average_face_x_to_cell
        if self._dim == 2:
            return self.average_edge_y_to_cell

        cdef np.int64_t[:] I,J
        cdef np.float64_t[:] V
        cdef Face *face1
        cdef Face *face2
        cdef np.int64_t ii
        I = np.empty(self.n_cells*2, dtype=np.int64)
        J = np.empty(self.n_cells*2, dtype=np.int64)
        V = np.empty(self.n_cells*2, dtype=np.float64)

        for cell in self.tree.cells:
            face1 = cell.faces[0] # x face
            face2 = cell.faces[1] # x face
            ii = cell.index
            I[ii*2 : ii*2 + 2] = ii
            J[ii*2    ] = face1.index
            J[ii*2 + 1] = face2.index
            V[ii*2 : ii*2 + 2] = 0.5

        Rfx = self._deflate_faces_x()
        self._average_face_x_to_cell = sp.csr_matrix((V, (I, J)))*Rfx
        return self._average_face_x_to_cell

    @property
    @cython.boundscheck(False)
    def average_face_y_to_cell(self):
        """
        Construct the averaging operator on cell faces in the y direction to
        cell centers.
        """
        if self._average_face_y_to_cell is not None:
            return self._average_face_y_to_cell
        if self._dim == 2:
            return self.average_edge_x_to_cell

        cdef np.int64_t[:] I,J
        cdef np.float64_t[:] V
        cdef Face *face1
        cdef Face *face2
        cdef np.int64_t ii
        I = np.empty(self.n_cells*2, dtype=np.int64)
        J = np.empty(self.n_cells*2, dtype=np.int64)
        V = np.empty(self.n_cells*2, dtype=np.float64)

        for cell in self.tree.cells:
            face1 = cell.faces[2] # y face
            face2 = cell.faces[3] # y face
            ii = cell.index
            I[ii*2 : ii*2 + 2] = ii
            J[ii*2    ] = face1.index
            J[ii*2 + 1] = face2.index
            V[ii*2 : ii*2 + 2] = 0.5

        Rfy = self._deflate_faces_y()
        self._average_face_y_to_cell = sp.csr_matrix((V, (I, J)))*Rfy
        return self._average_face_y_to_cell

    @property
    @cython.boundscheck(False)
    def average_face_z_to_cell(self):
        """
        Construct the averaging operator on cell faces in the z direction to
        cell centers.
        """
        if self._average_face_z_to_cell is not None:
            return self._average_face_z_to_cell
        if self._dim == 2:
            raise Exception('There are no z-faces in 2D')
        cdef np.int64_t[:] I,J
        cdef np.float64_t[:] V
        cdef Face *face1
        cdef Face *face2
        cdef np.int64_t ii
        I = np.empty(self.n_cells*2, dtype=np.int64)
        J = np.empty(self.n_cells*2, dtype=np.int64)
        V = np.empty(self.n_cells*2, dtype=np.float64)

        for cell in self.tree.cells:
            face1 = cell.faces[4]
            face2 = cell.faces[5]
            ii = cell.index
            I[ii*2 : ii*2 + 2] = ii
            J[ii*2    ] = face1.index
            J[ii*2 + 1] = face2.index
            V[ii*2 : ii*2 + 2] = 0.5

        Rfy = self._deflate_faces_z()
        self._average_face_z_to_cell = sp.csr_matrix((V, (I, J)))*Rfy
        return self._average_face_z_to_cell

    @property
    def average_face_to_cell(self):
        "Construct the averaging operator on cell faces to cell centers."
        if self._average_face_to_cell is None:
            stacks = [self.average_face_x_to_cell, self.aveFy2CC]
            if self._dim == 3:
                stacks += [self.average_face_z_to_cell]
            self._average_face_to_cell = 1./self._dim*sp.hstack(stacks).tocsr()
        return self._average_face_to_cell

    @property
    def average_face_to_cell_vector(self):
        "Construct the averaging operator on cell faces to cell centers."
        if self._average_face_to_cell_vector is None:
            stacks = [self.average_face_x_to_cell, self.aveFy2CC]
            if self._dim == 3:
                stacks += [self.average_face_z_to_cell]
            self._average_face_to_cell_vector = sp.block_diag(stacks).tocsr()
        return self._average_face_to_cell_vector

    @property
    @cython.boundscheck(False)
    def average_node_to_cell(self):
        "Construct the averaging operator on cell nodes to cell centers."
        cdef np.int64_t[:] I, J
        cdef np.float64_t[:] V
        cdef np.int64_t ii, id, n_ppc
        cdef double scale
        if self._average_node_to_cell is None:
            n_ppc = 1<<self._dim
            scale = 1.0/n_ppc
            I = np.empty(self.n_cells*n_ppc, dtype=np.int64)
            J = np.empty(self.n_cells*n_ppc, dtype=np.int64)
            V = np.empty(self.n_cells*n_ppc, dtype=np.float64)

            for cell in self.tree.cells:
                ii = cell.index
                for id in range(n_ppc):
                    I[ii*n_ppc + id] = ii
                    J[ii*n_ppc + id] = cell.points[id].index
                    V[ii*n_ppc + id] = scale

            Rn = self._deflate_nodes()
            self._average_node_to_cell = sp.csr_matrix((V, (I, J)), shape=(self.n_cells, self.n_total_nodes))*Rn
        return self._average_node_to_cell

    @property
    def average_node_to_edge_x(self):
        """
        Averaging operator on cell nodes to x-edges
        """
        if self._average_node_to_edge_x is not None:
            return self._average_node_to_edge_x
        cdef np.int64_t[:] I, J
        cdef np.float64_t[:] V
        cdef np.int64_t ii, id
        I = np.empty(self.n_edges_x*2, dtype=np.int64)
        J = np.empty(self.n_edges_x*2, dtype=np.int64)
        V = np.empty(self.n_edges_x*2, dtype=np.float64)

        for it in self.tree.edges_x:
            edge = it.second
            if edge.hanging:
                continue
            ii = edge.index
            for id in range(2):
                I[ii*2 + id] = ii
                J[ii*2 + id] = edge.points[id].index
                V[ii*2 + id] = 0.5

        Rn = self._deflate_nodes()
        self._average_node_to_edge_x = sp.csr_matrix((V, (I, J)), shape=(self.n_edges_x, self.n_total_nodes))*Rn
        return self._average_node_to_edge_x

    @property
    def average_node_to_edge_y(self):
        """
        Averaging operator on cell nodes to y-edges
        """
        if self._average_node_to_edge_y is not None:
            return self._average_node_to_edge_y
        cdef np.int64_t[:] I, J
        cdef np.float64_t[:] V
        cdef np.int64_t ii, id
        I = np.empty(self.n_edges_y*2, dtype=np.int64)
        J = np.empty(self.n_edges_y*2, dtype=np.int64)
        V = np.empty(self.n_edges_y*2, dtype=np.float64)

        for it in self.tree.edges_y:
            edge = it.second
            if edge.hanging:
                continue
            ii = edge.index
            for id in range(2):
                I[ii*2 + id] = ii
                J[ii*2 + id] = edge.points[id].index
                V[ii*2 + id] = 0.5

        Rn = self._deflate_nodes()
        self._average_node_to_edge_y = sp.csr_matrix((V, (I, J)), shape=(self.n_edges_y, self.n_total_nodes))*Rn
        return self._average_node_to_edge_y

    @property
    def average_node_to_edge_z(self):
        """
        Averaging operator on cell nodes to z-edges
        """
        if self._dim == 2:
            raise Exception('TreeMesh has no z-edges in 2D')
        if self._average_node_to_edge_z is not None:
            return self._average_node_to_edge_z
        cdef np.int64_t[:] I, J
        cdef np.float64_t[:] V
        cdef np.int64_t ii, id
        I = np.empty(self.n_edges_z*2, dtype=np.int64)
        J = np.empty(self.n_edges_z*2, dtype=np.int64)
        V = np.empty(self.n_edges_z*2, dtype=np.float64)

        for it in self.tree.edges_z:
            edge = it.second
            if edge.hanging:
                continue
            ii = edge.index
            for id in range(2):
                I[ii*2 + id] = ii
                J[ii*2 + id] = edge.points[id].index
                V[ii*2 + id] = 0.5

        Rn = self._deflate_nodes()
        self._average_node_to_edge_z = sp.csr_matrix((V, (I, J)), shape=(self.n_edges_z, self.n_total_nodes))*Rn
        return self._average_node_to_edge_z

    @property
    def average_node_to_edge(self):
        """
        Construct the averaging operator on cell nodes to cell edges, keeping
        each dimension separate.
        """
        if self._average_node_to_edge is not None:
            return self._average_node_to_edge

        stacks = [self.average_node_to_edge_x, self.average_node_to_edge_y]
        if self._dim == 3:
            stacks += [self.average_node_to_edge_z]
        self._average_node_to_edge = sp.vstack(stacks).tocsr()
        return self._average_node_to_edge

    @property
    def average_node_to_face_x(self):
        """
        Averaging operator on cell nodes to x-faces
        """
        if self._dim == 2:
            return self.average_node_to_edge_y
        if self._average_node_to_face_x is not None:
            return self._average_node_to_face_x
        cdef np.int64_t[:] I, J
        cdef np.float64_t[:] V
        cdef np.int64_t ii, id
        I = np.empty(self.n_faces_x*4, dtype=np.int64)
        J = np.empty(self.n_faces_x*4, dtype=np.int64)
        V = np.empty(self.n_faces_x*4, dtype=np.float64)

        for it in self.tree.faces_x:
            face = it.second
            if face.hanging:
                continue
            ii = face.index
            for id in range(4):
                I[ii*4 + id] = ii
                J[ii*4 + id] = face.points[id].index
                V[ii*4 + id] = 0.25

        Rn = self._deflate_nodes()
        self._average_node_to_face_x = sp.csr_matrix((V, (I, J)), shape=(self.n_faces_x, self.n_total_nodes))*Rn
        return self._average_node_to_face_x

    @property
    def average_node_to_face_y(self):
        """
        Averaging operator on cell nodes to y-faces
        """
        if self._dim == 2:
            return self.average_node_to_edge_x
        if self._average_node_to_face_y is not None:
            return self._average_node_to_face_y
        cdef np.int64_t[:] I, J
        cdef np.float64_t[:] V
        cdef np.int64_t ii, id

        I = np.empty(self.n_faces_y*4, dtype=np.int64)
        J = np.empty(self.n_faces_y*4, dtype=np.int64)
        V = np.empty(self.n_faces_y*4, dtype=np.float64)

        for it in self.tree.faces_y:
            face = it.second
            if face.hanging:
                continue
            ii = face.index
            for id in range(4):
                I[ii*4 + id] = ii
                J[ii*4 + id] = face.points[id].index
                V[ii*4 + id] = 0.25

        Rn = self._deflate_nodes()
        self._average_node_to_face_y = sp.csr_matrix((V, (I, J)), shape=(self.n_faces_y, self.n_total_nodes))*Rn
        return self._average_node_to_face_y

    @property
    def average_node_to_face_z(self):
        """
        Averaging operator on cell nodes to z-faces
        """
        if self._dim == 2:
            raise Exception('TreeMesh has no z faces in 2D')
        cdef np.int64_t[:] I, J
        cdef np.float64_t[:] V
        cdef np.int64_t ii, id,
        if self._average_node_to_face_z is not None:
            return self._average_node_to_face_z

        I = np.empty(self.n_faces_z*4, dtype=np.int64)
        J = np.empty(self.n_faces_z*4, dtype=np.int64)
        V = np.empty(self.n_faces_z*4, dtype=np.float64)

        for it in self.tree.faces_z:
            face = it.second
            if face.hanging:
                continue
            ii = face.index
            for id in range(4):
                I[ii*4 + id] = ii
                J[ii*4 + id] = face.points[id].index
                V[ii*4 + id] = 0.25

        Rn = self._deflate_nodes()
        self._average_node_to_face_z = sp.csr_matrix((V, (I, J)), shape=(self.n_faces_z, self.n_total_nodes))*Rn
        return self._average_node_to_face_z

    @property
    def average_node_to_face(self):
        """
        Construct the averaging operator on cell nodes to cell edges, keeping
        each dimension separate.
        """
        if self._average_node_to_face is not None:
            return self._average_node_to_face

        stacks = [self.average_node_to_face_x, self.average_node_to_face_y]
        if self._dim == 3:
            stacks += [self.average_node_to_face_z]
        self._average_node_to_face = sp.vstack(stacks).tocsr()
        return self._average_node_to_face

    @property
    def average_cell_to_face(self):
        "Construct the averaging operator on cell centers to cell faces."
        if self._average_cell_to_face is not None:
            return self._average_cell_to_face
        stacks = [self.average_cell_to_face_x, self.average_cell_to_face_y]
        if self._dim == 3:
            stacks.append(self.average_cell_to_face_z)

        self._average_cell_to_face = sp.vstack(stacks).tocsr()
        return self._average_cell_to_face

    @property
    def average_cell_vector_to_face(self):
        "Construct the averaging operator on cell centers to cell faces."
        if self._average_cell_vector_to_face is not None:
            return self._average_cell_vector_to_face
        stacks = [self.average_cell_to_face_x, self.average_cell_to_face_y]
        if self._dim == 3:
            stacks.append(self.average_cell_to_face_z)

        self._average_cell_vector_to_face = sp.block_diag(stacks).tocsr()
        return self._average_cell_vector_to_face

    @property
    def average_cell_to_face_x(self):
        "Construct the averaging operator on cell centers to cell x-faces."
        if self._average_cell_to_face_x is not None:
            return self._average_cell_to_face_x
        cdef np.int64_t[:] I = np.zeros(2*self.n_total_faces_x, dtype=np.int64)
        cdef np.int64_t[:] J = np.zeros(2*self.n_total_faces_x, dtype=np.int64)
        cdef np.float64_t[:] V = np.zeros(2*self.n_total_faces_x, dtype=np.float64)
        cdef int dim = self._dim
        cdef int_t ind, ind_parent
        cdef int_t children_per_parent = (dim-1)*2
        cdef c_Cell* child
        cdef c_Cell* next_cell
        cdef c_Cell* prev_cell
        cdef double w

        for cell in self.tree.cells :
            next_cell = cell.neighbors[1]
            prev_cell = cell.neighbors[0]
            # handle extrapolation to boundary faces
            if next_cell == NULL:
                if dim == 2:
                    ind = cell.edges[3].index # +x face
                else:
                    ind = cell.faces[1].index # +x face
                I[2*ind  ] = ind
                J[2*ind  ] = cell.index
                V[2*ind  ] = 1.0
                continue
            if prev_cell == NULL:
                if dim == 2:
                    ind = cell.edges[2].index # -x face
                else:
                    ind = cell.faces[0].index # -x face
                I[2*ind  ] = ind
                J[2*ind  ] = cell.index
                V[2*ind  ] = 1.0

            if next_cell.is_leaf():
                if next_cell.level == cell.level:
                    #I am on the same level and easy to interpolate
                    if dim == 2:
                        ind = cell.edges[3].index
                        w = (next_cell.location[0] - cell.edges[3].location[0])/(
                             next_cell.location[0] - cell.location[0])
                    else:
                        ind = cell.faces[1].index
                        w = (next_cell.location[0] - cell.faces[1].location[0])/(
                             next_cell.location[0] - cell.location[0])
                    I[2*ind  ] = ind
                    I[2*ind+1] = ind
                    J[2*ind  ] = cell.index
                    J[2*ind+1] = next_cell.index
                    V[2*ind  ] = w
                    V[2*ind+1] = (1.0-w)
                else:
                    # if next cell is a level larger than i am, then i need to accumulate into w
                    if dim == 2:
                        ind = cell.edges[3].index
                        ind_parent = cell.edges[3].parents[0].index
                        w = (next_cell.location[0] - cell.edges[3].location[0])/(
                             next_cell.location[0] - cell.location[0])
                    else:
                        ind = cell.faces[1].index
                        ind_parent = cell.faces[1].parent.index
                        w = (next_cell.location[0] - cell.faces[1].location[0])/(
                             next_cell.location[0] - cell.location[0])
                    I[2*ind  ] = ind_parent
                    I[2*ind+1] = ind_parent
                    J[2*ind  ] = cell.index
                    J[2*ind+1] = next_cell.index
                    V[2*ind  ] = w/children_per_parent
                    V[2*ind+1] = (1.0-w)/children_per_parent
            else:
                #should mean next cell is not a leaf so need to loop over children
                if dim == 2:
                    ind_parent = cell.edges[3].index
                    w = (next_cell.children[0].location[0] - cell.edges[3].location[0])/(
                         next_cell.children[0].location[0] - cell.location[0])
                    for i in range(2):
                        child = next_cell.children[2*i]
                        ind = child.edges[2].index
                        I[2*ind    ] = ind_parent
                        I[2*ind + 1] = ind_parent
                        J[2*ind    ] = cell.index
                        J[2*ind + 1] = child.index
                        V[2*ind    ] = w/children_per_parent
                        V[2*ind + 1] = (1.0-w)/children_per_parent
                else:
                    ind_parent = cell.faces[1].index
                    w = (next_cell.children[0].location[0] - cell.faces[1].location[0])/(
                         next_cell.children[0].location[0] - cell.location[0])
                    for i in range(4): # four neighbors in +x direction
                        child = next_cell.children[2*i] #0 2 4 6
                        ind = child.faces[0].index
                        I[2*ind    ] = ind_parent
                        I[2*ind + 1] = ind_parent
                        J[2*ind    ] = cell.index
                        J[2*ind + 1] = child.index
                        V[2*ind    ] = w/children_per_parent
                        V[2*ind + 1] = (1.0-w)/children_per_parent

        self._average_cell_to_face_x = sp.csr_matrix((V, (I, J)), shape=(self.n_faces_x, self.n_cells))
        return self._average_cell_to_face_x

    @property
    def average_cell_to_face_y(self):
        "Construct the averaging operator on cell centers to cell y-faces."
        if self._average_cell_to_face_y is not None:
            return self._average_cell_to_face_y
        cdef np.int64_t[:] I = np.zeros(2*self.n_total_faces_y, dtype=np.int64)
        cdef np.int64_t[:] J = np.zeros(2*self.n_total_faces_y, dtype=np.int64)
        cdef np.float64_t[:] V = np.zeros(2*self.n_total_faces_y, dtype=np.float64)
        cdef int dim = self._dim
        cdef int_t ind, ind_parent
        cdef int_t children_per_parent = (dim-1)*2
        cdef c_Cell* child
        cdef c_Cell* next_cell
        cdef c_Cell* prev_cell
        cdef double w

        for cell in self.tree.cells :
            next_cell = cell.neighbors[3]
            prev_cell = cell.neighbors[2]
            # handle extrapolation to boundary faces
            if next_cell == NULL:
                if dim == 2:
                    ind = cell.edges[1].index
                else:
                    ind = cell.faces[3].index
                I[2*ind  ] = ind
                J[2*ind  ] = cell.index
                V[2*ind  ] = 1.0
                continue
            if prev_cell == NULL:
                if dim == 2:
                    ind = cell.edges[0].index
                else:
                    ind = cell.faces[2].index
                I[2*ind  ] = ind
                J[2*ind  ] = cell.index
                V[2*ind  ] = 1.0

            if next_cell.is_leaf():
                if next_cell.level == cell.level:
                    #I am on the same level and easy to interpolate
                    if dim == 2:
                        ind = cell.edges[1].index
                        w = (next_cell.location[1] - cell.edges[1].location[1])/(
                             next_cell.location[1] - cell.location[1])
                    else:
                        ind = cell.faces[3].index
                        w = (next_cell.location[1] - cell.faces[3].location[1])/(
                             next_cell.location[1] - cell.location[1])
                    I[2*ind  ] = ind
                    I[2*ind+1] = ind
                    J[2*ind  ] = cell.index
                    J[2*ind+1] = next_cell.index
                    V[2*ind  ] = w
                    V[2*ind+1] = (1.0-w)
                else:
                    # if next cell is a level larger than i am
                    if dim == 2:
                        ind = cell.edges[1].index
                        ind_parent = cell.edges[1].parents[0].index
                        w = (next_cell.location[1] - cell.edges[1].location[1])/(
                             next_cell.location[1] - cell.location[1])
                    else:
                        ind = cell.faces[3].index
                        ind_parent = cell.faces[3].parent.index
                        w = (next_cell.location[1] - cell.faces[3].location[1])/(
                             next_cell.location[1] - cell.location[1])
                    I[2*ind  ] = ind_parent
                    I[2*ind+1] = ind_parent
                    J[2*ind  ] = cell.index
                    J[2*ind+1] = next_cell.index
                    V[2*ind  ] = w/children_per_parent
                    V[2*ind+1] = (1.0-w)/children_per_parent
            else:
                #should mean next cell is not a leaf so need to loop over children
                if dim == 2:
                    ind_parent = cell.edges[1].index
                    w = (next_cell.children[0].location[1] - cell.edges[1].location[1])/(
                         next_cell.children[0].location[1] - cell.location[1])
                    for i in range(2):
                        child = next_cell.children[i]
                        ind = child.edges[0].index
                        I[2*ind    ] = ind_parent
                        I[2*ind + 1] = ind_parent
                        J[2*ind    ] = cell.index
                        J[2*ind + 1] = child.index
                        V[2*ind    ] = w/children_per_parent
                        V[2*ind + 1] = (1.0-w)/children_per_parent
                else:
                    ind_parent = cell.faces[3].index
                    w = (next_cell.children[0].location[1] - cell.faces[3].location[1])/(
                         next_cell.children[0].location[1] - cell.location[1])
                    for i in range(4): # four neighbors in +y direction
                        child = next_cell.children[(i>>1)*4 + i%2] #0 1 4 5
                        ind = child.faces[2].index
                        I[2*ind    ] = ind_parent
                        I[2*ind + 1] = ind_parent
                        J[2*ind    ] = cell.index
                        J[2*ind + 1] = child.index
                        V[2*ind    ] = w/children_per_parent
                        V[2*ind + 1] = (1.0-w)/children_per_parent

        self._average_cell_to_face_y = sp.csr_matrix((V, (I,J)), shape=(self.n_faces_y, self.n_cells))
        return self._average_cell_to_face_y

    @property
    def average_cell_to_face_z(self):
        "Construct the averaging operator on cell centers to cell z-faces."
        if self.dim == 2:
            raise Exception('TreeMesh has no z-faces in 2D')
        if self._average_cell_to_face_z is not None:
            return self._average_cell_to_face_z
        cdef np.int64_t[:] I = np.zeros(2*self.n_total_faces_z, dtype=np.int64)
        cdef np.int64_t[:] J = np.zeros(2*self.n_total_faces_z, dtype=np.int64)
        cdef np.float64_t[:] V = np.zeros(2*self.n_total_faces_z, dtype=np.float64)
        cdef int dim = self._dim
        cdef int_t ind, ind_parent
        cdef int_t children_per_parent = (dim-1)*2
        cdef c_Cell* child
        cdef c_Cell* next_cell
        cdef c_Cell* prev_cell
        cdef double w

        for cell in self.tree.cells :
            next_cell = cell.neighbors[5]
            prev_cell = cell.neighbors[4]
            # handle extrapolation to boundary faces
            if next_cell == NULL:
                ind = cell.faces[5].index # +z face
                I[2*ind  ] = ind
                J[2*ind  ] = cell.index
                V[2*ind  ] = 1.0
                continue
            if prev_cell == NULL:
                ind = cell.faces[4].index # -z face
                I[2*ind  ] = ind
                J[2*ind  ] = cell.index
                V[2*ind  ] = 1.0

            if next_cell.is_leaf():
                if next_cell.level == cell.level:
                    #I am on the same level and easy to interpolate
                    ind = cell.faces[5].index
                    w = (next_cell.location[2] - cell.faces[5].location[2])/(
                         next_cell.location[2] - cell.location[2])
                    I[2*ind  ] = ind
                    I[2*ind+1] = ind
                    J[2*ind  ] = cell.index
                    J[2*ind+1] = next_cell.index
                    V[2*ind  ] = w
                    V[2*ind+1] = (1.0-w)
                else:
                    # if next cell is a level larger than i am
                    ind = cell.faces[5].index
                    ind_parent = cell.faces[5].parent.index
                    w = (next_cell.location[2] - cell.faces[5].location[2])/(
                         next_cell.location[2] - cell.location[2])
                    I[2*ind  ] = ind_parent
                    I[2*ind+1] = ind_parent
                    J[2*ind  ] = cell.index
                    J[2*ind+1] = next_cell.index
                    V[2*ind  ] = w/children_per_parent
                    V[2*ind+1] = (1.0-w)/children_per_parent
            else:
                #should mean next cell is not a leaf so need to loop over children
                ind_parent = cell.faces[5].index
                w = (next_cell.children[0].location[2] - cell.faces[5].location[2])/(
                     next_cell.children[0].location[2] - cell.location[2])
                for i in range(4): # four neighbors in +x direction
                    child = next_cell.children[i]
                    ind = child.faces[4].index #0 1 2 3
                    I[2*ind    ] = ind_parent
                    I[2*ind + 1] = ind_parent
                    J[2*ind    ] = cell.index
                    J[2*ind + 1] = child.index
                    V[2*ind    ] = w/children_per_parent
                    V[2*ind + 1] = (1.0-w)/children_per_parent

        self._average_cell_to_face_z = sp.csr_matrix((V, (I,J)), shape=(self.n_faces_z, self.n_cells))
        return self._average_cell_to_face_z

    def _get_containing_cell_index(self, loc):
        cdef double x, y, z
        x = loc[0]
        y = loc[1]
        if self._dim == 3:
            z = loc[2]
        else:
            z = 0
        return self.tree.containing_cell(x, y, z).index

    def _get_containing_cell_indexes(self, locs):
        locs = np.require(np.atleast_2d(locs), dtype=np.float64, requirements='C')
        cdef double[:,:] d_locs = locs
        cdef int_t n_locs = d_locs.shape[0]
        cdef np.int64_t[:] indexes = np.empty(n_locs, dtype=np.int64)
        cdef double x, y, z
        for i in range(n_locs):
            x = d_locs[i, 0]
            y = d_locs[i, 1]
            if self._dim == 3:
                z = d_locs[i, 2]
            else:
                z = 0
            indexes[i] = self.tree.containing_cell(x, y, z).index
        if n_locs==1:
            return indexes[0]
        return np.array(indexes)

    def _count_cells_per_index(self):
        cdef np.int64_t[:] counts = np.zeros(self.max_level+1, dtype=np.int64)
        for cell in self.tree.cells:
            counts[cell.level] += 1
        return np.array(counts)

    def _cell_levels_by_indexes(self, index):
        index = np.require(np.atleast_1d(index), dtype=np.int64, requirements='C')
        cdef np.int64_t[:] inds = index
        cdef int_t n_cells = inds.shape[0]
        cdef np.int64_t[:] levels = np.empty(n_cells, dtype=np.int64)
        for i in range(n_cells):
            levels[i] = self.tree.cells[inds[i]].level
        if n_cells == 1:
            return levels[0]
        else:
            return np.array(levels)

    def _getFaceP(self, xFace, yFace, zFace):
        cdef int dim = self._dim
        cdef int_t ind, id

        cdef np.int64_t[:] I, J, J1, J2, J3
        cdef np.float64_t[:] V

        J1 = np.empty(self.n_cells, dtype=np.int64)
        J2 = np.empty(self.n_cells, dtype=np.int64)
        if dim==3:
            J3 = np.empty(self.n_cells, dtype=np.int64)

        cdef int[3] faces
        cdef np.int64_t[:] offsets = np.empty(self._dim, dtype=np.int64)
        faces[0] = (xFace == 'fXp')
        faces[1] = (yFace == 'fYp')
        if dim == 3:
            faces[2] = (zFace == 'fZp')

        if dim == 2:
            offsets[0] = 0
            offsets[1] = self.n_total_faces_x
        else:
            offsets[0] = 0
            offsets[1] = self.n_total_faces_x
            offsets[2] = self.n_total_faces_x + self.n_total_faces_y

        for cell in self.tree.cells:
            ind = cell.index
            if dim==2:
                J1[ind] = cell.edges[2 + faces[0]].index
                J2[ind] = cell.edges[    faces[1]].index + offsets[1]
            else:
                J1[ind] = cell.faces[    faces[0]].index
                J2[ind] = cell.faces[2 + faces[1]].index + offsets[1]
                J3[ind] = cell.faces[4 + faces[2]].index + offsets[2]

        I = np.arange(dim*self.n_cells, dtype=np.int64)
        if dim==2:
            J = np.r_[J1, J2]
        else:
            J = np.r_[J1, J2, J3]
        V = np.ones(self.n_cells*dim, dtype=np.float64)

        P = sp.csr_matrix((V, (I, J)), shape=(self._dim*self.n_cells, self.n_total_faces))
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
        cdef int dim = self._dim
        cdef int_t ind, id
        cdef int epc = 1<<(dim-1) #edges per cell 2/4

        cdef np.int64_t[:] I, J, J1, J2, J3
        cdef np.float64_t[:] V

        J1 = np.empty(self.n_cells, dtype=np.int64)
        J2 = np.empty(self.n_cells, dtype=np.int64)
        if dim == 3:
            J3 = np.empty(self.n_cells, dtype=np.int64)

        cdef int[3] edges
        cdef np.int64_t[:] offsets = np.empty(self._dim, dtype=np.int64)
        try:
            edges[0] = int(xEdge[-1]) #0, 1, 2, 3
            edges[1] = int(yEdge[-1]) #0, 1, 2, 3
            if dim == 3:
                edges[2] = int(zEdge[-1]) #0, 1, 2, 3
        except ValueError:
            raise Exception('Last character of edge string must be 0, 1, 2, or 3')

        offsets[0] = 0
        offsets[1] = self.n_total_edges_x
        if dim==3:
            offsets[2] = self.n_total_edges_x + self.n_total_edges_y

        for cell in self.tree.cells:
            ind = cell.index
            J1[ind] = cell.edges[0*epc + edges[0]].index + offsets[0]
            J2[ind] = cell.edges[1*epc + edges[1]].index + offsets[1]
            if dim==3:
                J3[ind] = cell.edges[2*epc + edges[2]].index + offsets[2]

        I = np.arange(dim*self.n_cells, dtype=np.int64)
        if dim==2:
            J = np.r_[J1, J2]
        else:
            J = np.r_[J1, J2, J3]
        V = np.ones(self.n_cells*dim, dtype=np.float64)

        P = sp.csr_matrix((V, (I, J)), shape=(self._dim*self.n_cells, self.n_total_edges))
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

    def _getEdgeIntMat(self, locs, zerosOutside, direction):
        cdef:
            double[:, :] locations = locs
            int_t dir, dir1, dir2
            int_t dim = self._dim
            int_t n_loc = locs.shape[0]
            int_t n_edges = 2 if self._dim == 2 else 4
            np.int64_t[:] I = np.empty(n_loc*n_edges, dtype=np.int64)
            np.int64_t[:] J = np.empty(n_loc*n_edges, dtype=np.int64)
            np.float64_t[:] V = np.empty(n_loc*n_edges, dtype=np.float64)

            int_t ii, i, j, offset
            c_Cell *cell
            double x, y, z
            double w1, w2
            double eps = 100*np.finfo(float).eps
            int zeros_out = zerosOutside

        if direction == 'x':
            dir, dir1, dir2 = 0, 1, 2
            offset = 0
        elif direction == 'y':
            dir, dir1, dir2 = 1, 0, 2
            offset = self.n_total_edges_x
        elif direction == 'z':
            dir, dir1, dir2 = 2, 0, 1
            offset = self.n_total_edges_x + self.n_total_edges_y
        else:
            raise ValueError('Invalid direction, must be x, y, or z')

        for i in range(n_loc):
            x = locations[i, 0]
            y = locations[i, 1]
            z = locations[i, 2] if dim==3 else 0.0
            #get containing (or closest) cell
            cell = self.tree.containing_cell(x, y, z)
            for j in range(n_edges):
                I[n_edges*i+j] = i
                J[n_edges*i+j] = cell.edges[n_edges*dir+j].index + offset

            w1 = ((cell.edges[n_edges*dir+1].location[dir1] - locations[i, dir1])/
                  (cell.edges[n_edges*dir+1].location[dir1] - cell.edges[n_edges*dir].location[dir1]))
            if dim == 3:
                w2 = ((cell.edges[n_edges*dir+3].location[dir2] - locations[i, dir2])/
                      (cell.edges[n_edges*dir+3].location[dir2] - cell.edges[n_edges*dir].location[dir2]))
            else:
                w2 = 1.0
            if zeros_out:
                if (w1 < -eps or w1 > 1 + eps or w2 < -eps or w2 > eps):
                    for j in range(n_edges):
                        V[n_edges*i + j] = 0.0
                    continue
            w1 = _clip01(w1)
            w2 = _clip01(w2)

            V[n_edges*i  ] = w1*w2
            V[n_edges*i+1] = (1.0-w1)*w2
            if dim == 3:
                V[n_edges*i+2] = w1*(1.0-w2)
                V[n_edges*i+3] = (1.0-w1)*(1.0-w2)

        Re = self._deflate_edges()
        A = sp.csr_matrix((V, (I, J)), shape=(locs.shape[0], self.n_total_edges))
        return A*Re

    def _getFaceIntMat(self, locs, zerosOutside, direction):
        cdef:
            double[:, :] locations = locs
            int_t dir, dir2d
            int_t dim = self._dim
            int_t n_loc = locs.shape[0]
            int_t n_faces = 2
            np.int64_t[:] I = np.empty(n_loc*n_faces, dtype=np.int64)
            np.int64_t[:] J = np.empty(n_loc*n_faces, dtype=np.int64)
            np.float64_t[:] V = np.empty(n_loc*n_faces, dtype=np.float64)

            int_t ii, i, offset
            c_Cell *cell
            double x, y, z
            double w
            double eps = 100*np.finfo(float).eps
            int zeros_out = zerosOutside

        if direction == 'x':
            dir = 0
            dir2d = 1
            offset = 0
        elif direction == 'y':
            dir = 1
            dir2d = 0
            offset = self.n_total_faces_x
        elif direction == 'z':
            dir = 2
            offset = self.n_total_faces_x + self.n_total_faces_y
        else:
            raise ValueError('Invalid direction, must be x, y, or z')

        for i in range(n_loc):
            x = locations[i, 0]
            y = locations[i, 1]
            z = locations[i, 2] if dim==3 else 0.0
            #get containing (or closest) cell
            cell = self.tree.containing_cell(x, y, z)
            I[n_faces*i  ] = i
            I[n_faces*i+1] = i
            if self._dim == 3:
                J[n_faces*i  ] = cell.faces[dir*2  ].index + offset
                J[n_faces*i+1] = cell.faces[dir*2+1].index + offset
                w = ((cell.faces[dir*2+1].location[dir] - locations[i, dir])/
                      (cell.faces[dir*2+1].location[dir] - cell.faces[dir*2].location[dir]))
            else:
                J[n_faces*i  ] = cell.edges[dir2d*2  ].index + offset
                J[n_faces*i+1] = cell.edges[dir2d*2+1].index + offset
                w = ((cell.edges[dir2d*2+1].location[dir] - locations[i, dir])/
                      (cell.edges[dir2d*2+1].location[dir] - cell.edges[dir2d*2].location[dir]))
            if zeros_out:
                if (w < -eps or w > 1 + eps):
                    V[n_faces*i  ] = 0.0
                    V[n_faces*i+1] = 0.0
                    continue
            w = _clip01(w)
            V[n_faces*i  ] = w
            V[n_faces*i+1] = 1.0-w

        Rf = self._deflate_faces()
        return sp.csr_matrix((V, (I, J)), shape=(locs.shape[0], self.n_total_faces))*Rf

    def _getNodeIntMat(self, locs, zerosOutside):
        cdef:
            double[:, :] locations = locs
            int_t dim = self._dim
            int_t n_loc = locs.shape[0]
            int_t n_nodes = 1<<dim
            np.int64_t[:] I = np.empty(n_loc*n_nodes, dtype=np.int64)
            np.int64_t[:] J = np.empty(n_loc*n_nodes, dtype=np.int64)
            np.float64_t[:] V = np.empty(n_loc*n_nodes, dtype=np.float64)

            int_t ii, i
            c_Cell *cell
            double x, y, z
            double wx, wy, wz
            double eps = 100*np.finfo(float).eps
            int zeros_out = zerosOutside

        for i in range(n_loc):
            x = locations[i, 0]
            y = locations[i, 1]
            z = locations[i, 2] if dim==3 else 0.0
            #get containing (or closest) cell
            cell = self.tree.containing_cell(x, y, z)
            #calculate weights
            wx = ((cell.points[3].location[0] - x)/
                  (cell.points[3].location[0] - cell.points[0].location[0]))
            wy = ((cell.points[3].location[1] - y)/
                  (cell.points[3].location[1] - cell.points[0].location[1]))
            if dim == 3:
                wz = ((cell.points[7].location[2] - z)/
                      (cell.points[7].location[2] - cell.points[0].location[2]))
            else:
                wz = 1.0


            I[n_nodes*i:n_nodes*i + n_nodes] = i


            if zeros_out:
                if (wx < -eps or wy < -eps or wz < -eps or
                    wx>1 + eps or wy > 1 + eps or wz > 1 + eps):
                    for ii in range(n_nodes):
                        J[n_nodes*i + ii] = 0
                        V[n_nodes*i + ii] = 0.0
                    continue

            wx = _clip01(wx)
            wy = _clip01(wy)
            wz = _clip01(wz)
            for ii in range(n_nodes):
                J[n_nodes*i + ii] = cell.points[ii].index

            V[n_nodes*i    ] = wx*wy*wz
            V[n_nodes*i + 1] = (1 - wx)*wy*wz
            V[n_nodes*i + 2] = wx*(1 - wy)*wz
            V[n_nodes*i + 3] = (1 - wx)*(1 - wy)*wz
            if dim==3:
                V[n_nodes*i + 4] = wx*wy*(1 - wz)
                V[n_nodes*i + 5] = (1 - wx)*wy*(1 - wz)
                V[n_nodes*i + 6] = wx*(1 - wy)*(1 - wz)
                V[n_nodes*i + 7] = (1 - wx)*(1 - wy)*(1 - wz)

        Rn = self._deflate_nodes()
        return sp.csr_matrix((V, (I, J)), shape=(locs.shape[0],self.n_total_nodes))*Rn

    def _getCellIntMat(self, locs, zerosOutside):
        cdef:
            double[:, :] locations = locs
            int_t dim = self._dim
            int_t n_loc = locations.shape[0]
            np.int64_t[:] I = np.arange(n_loc, dtype=np.int64)
            np.int64_t[:] J = np.empty(n_loc, dtype=np.int64)
            np.float64_t[:] V = np.ones(n_loc, dtype=np.float64)

            int_t ii, i
            c_Cell *cell
            double x, y, z
            double eps = 100*np.finfo(float).eps
            int zeros_out = zerosOutside

        for i in range(n_loc):
            x = locations[i, 0]
            y = locations[i, 1]
            z = locations[i, 2] if dim==3 else 0.0
            # get containing (or closest) cell
            cell = self.tree.containing_cell(x, y, z)
            J[i] = cell.index
            if zeros_out:
                if x < cell.points[0].location[0]-eps:
                    V[i] = 0.0
                elif x > cell.points[3].location[0]+eps:
                    V[i] = 0.0
                elif y < cell.points[0].location[1]-eps:
                    V[i] = 0.0
                elif y > cell.points[3].location[1]+eps:
                    V[i] = 0.0
                elif dim == 3 and z < cell.points[0].location[2]-eps:
                    V[i] = 0.0
                elif dim == 3 and z > cell.points[7].location[2]+eps:
                    V[i] = 0.0

        return sp.csr_matrix((V, (I, J)), shape=(locs.shape[0],self.n_cells))

    @property
    def cell_nodes(self):
        """The index of nodes for each cell.

        Returns
        -------
        numpy.ndarray of ints
            Index array of shape (n_cells, 4) if 2D, or (n_cells, 6) if 3D

        Notes
        -----
        These indices will also point to hanging nodes.
        """
        cdef int_t npc = 4 if self.dim == 2 else 6
        inds = np.empty((self.n_cells, npc), dtype=np.int64)
        cdef np.int64_t[:, :] node_index = inds
        cdef int_t i

        for cell in self.tree.cells:
            for i in range(npc):
                node_index[cell.index, i] = cell.points[i].index

        return inds

    @property
    def edge_nodes(self):
        """The index of nodes for every edge.

        The index of the nodes at each end of every (including hanging) edge.

        Returns
        -------
        tuple of numpy.ndarray of ints
            One numpy array for each edge type (x, y, (z)) for this mesh.

        Notes
        -----
        These arrays will also index into the hanging nodes.
        """
        inds_x = np.empty((self.n_total_edges_x, 2), dtype=np.int64)
        inds_y = np.empty((self.n_total_edges_y, 2), dtype=np.int64)
        cdef np.int64_t[:, :] edge_inds

        edge_inds = inds_x
        for it in self.tree.edges_x:
            edge = it.second
            edge_inds[edge.index, 0] = edge.points[0].index
            edge_inds[edge.index, 1] = edge.points[1].index

        edge_inds = inds_y
        for it in self.tree.edges_y:
            edge = it.second
            edge_inds[edge.index, 0] = edge.points[0].index
            edge_inds[edge.index, 1] = edge.points[1].index

        if self.dim == 2:
            return inds_x, inds_y

        inds_z = np.empty((self.n_total_edges_z, 2), dtype=np.int64)
        edge_inds = inds_z
        for it in self.tree.edges_z:
            edge = it.second
            edge_inds[edge.index, 0] = edge.points[0].index
            edge_inds[edge.index, 1] = edge.points[1].index

        return inds_x, inds_y, inds_z

    def __getstate__(self):
        cdef int id, dim = self._dim
        indArr = np.empty((self.n_cells, dim), dtype=np.int)
        levels = np.empty((self.n_cells), dtype=np.int)
        cdef np.int_t[:, :] _indArr = indArr
        cdef np.int_t[:] _levels = levels
        for cell in self.tree.cells:
            for id in range(dim):
                _indArr[cell.index, id] = cell.location_ind[id]
            _levels[cell.index] = cell.level
        return indArr, levels

    def __setstate__(self, state):
        indArr, levels = state
        indArr = np.asarray(indArr)
        levels = np.asarray(levels)
        xs = np.array(self._xs)
        ys = np.array(self._ys)
        if self._dim == 3:
            zs = np.array(self._zs)
            points = np.column_stack((xs[indArr[:, 0]],
                                      ys[indArr[:, 1]],
                                      zs[indArr[:, 2]]))
        else:
            points = np.column_stack((xs[indArr[:, 0]], ys[indArr[:, 1]]))
        self.insert_cells(points, levels)

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Get the start, stop, and step from the slice
            return [self[ii] for ii in range(*key.indices(len(self)))]
        elif isinstance(key, (int, np.integer)):
            if key < 0:  # Handle negative indices
                key += len(self)
            if key >= len(self):
                raise IndexError(
                    "The index ({0:d}) is out of range.".format(key)
                )
            pycell = TreeCell()
            pycell._set(self.tree.cells[key])
            return pycell
        else:
            raise TypeError("Invalid argument type.")

    @property
    def _ubc_indArr(self):
        if self.__ubc_indArr is not None:
            return self.__ubc_indArr
        indArr, levels = self.__getstate__()

        max_level = self.tree.max_level

        levels = 1<<(max_level - levels)

        if self.dim == 2:
            indArr[:, -1] = (self._ys.shape[0]-1) - indArr[:, -1]
        else:
            indArr[:, -1] = (self._zs.shape[0]-1) - indArr[:, -1]

        indArr = (indArr - levels[:, None])//2
        indArr += 1

        self.__ubc_indArr = (indArr, levels)
        return self.__ubc_indArr

    @property
    def _ubc_order(self):
        if self.__ubc_order is not None:
            return self.__ubc_order
        indArr, _ = self._ubc_indArr
        if self.dim == 2:
            self.__ubc_order = np.lexsort((indArr[:, 0], indArr[:, 1]))
        else:
            self.__ubc_order = np.lexsort((indArr[:, 0], indArr[:, 1], indArr[:, 2]))
        return self.__ubc_order

    def __dealloc__(self):
        del self.tree
        del self.wrapper

    @cython.boundscheck(False)
    @cython.cdivision(True)
    def _vol_avg_from_tree(self, _TreeMesh meshin, values=None, output=None):
        # first check if they have the same tensor base, as it makes it a lot easier...
        cdef int_t same_base
        try:
            same_base = (
                np.allclose(self.nodes_x, meshin.nodes_x)
                and np.allclose(self.nodes_y, meshin.nodes_y)
                and (self.dim == 2 or np.allclose(self.nodes_z, meshin.nodes_z))
            )
        except ValueError:
            same_base = False
        cdef c_Cell * out_cell
        cdef c_Cell * in_cell

        cdef np.float64_t[:] vals = np.array([])
        cdef np.float64_t[:] outs = np.array([])
        cdef int_t build_mat = 1

        if values is not None:
            vals = values
            if output is None:
                output = np.empty(self.n_cells)
            output[:] = 0
            outs = output

            build_mat = 0

        cdef vector[int_t] row_inds, col_inds
        cdef vector[int_t] indptr
        cdef vector[double] all_weights

        cdef vector[int_t] *overlapping_cells
        cdef double *weights
        cdef double over_lap_vol
        cdef double x1m, x1p, y1m, y1p, z1m, z1p
        cdef double x2m, x2p, y2m, y2p, z2m, z2p
        cdef double[:] origin = meshin._origin
        cdef double[:] xF
        if self.dim == 2:
            xF = np.array([meshin._xs[-1], meshin._ys[-1]])
        else:
            xF = np.array([meshin._xs[-1], meshin._ys[-1], meshin._zs[-1]])

        cdef int_t nnz_counter = 0
        cdef int_t nnz_row = 0
        if build_mat:
            indptr.push_back(0)
        cdef int_t i, in_cell_ind
        cdef int_t n_overlap
        cdef double weight_sum
        cdef double weight
        cdef vector[int_t] out_visited
        cdef int_t n_unvisited

        # easier path if they share the same base:
        if same_base:
            if build_mat:
                all_weights.resize(meshin.n_cells, 0.0)
                row_inds.resize(meshin.n_cells, 0)
            out_visited.resize(self.n_cells, 0)
            for in_cell in meshin.tree.cells:
                # for each input cell find containing output cell
                out_cell = self.tree.containing_cell(
                    in_cell.location[0],
                    in_cell.location[1],
                    in_cell.location[2]
                )
                # if containing output cell is lower level (larger) than input cell:
                # contribution is related to difference of levels (aka ratio of volumes)
                # else:
                # contribution is 1.0
                if out_cell.level < in_cell.level:
                    out_visited[out_cell.index] = 1
                    weight = in_cell.volume/out_cell.volume
                    if not build_mat:
                        outs[out_cell.index] += weight*vals[in_cell.index]
                    else:
                        all_weights[in_cell.index] = weight
                        row_inds[in_cell.index] = out_cell.index

            if build_mat:
                P = sp.csr_matrix((all_weights, (row_inds, np.arange(meshin.n_cells))),
                                  shape=(self.n_cells, meshin.n_cells))

                n_unvisited = self.n_cells - np.sum(out_visited)
                row_inds.resize(n_unvisited, 0)
                col_inds.resize(n_unvisited, 0)
            i = 0
            # assign weights of 1 to unvisited output cells and find their containing cell
            for out_cell in self.tree.cells:
                if not out_visited[out_cell.index]:
                    in_cell = meshin.tree.containing_cell(
                        out_cell.location[0],
                        out_cell.location[1],
                        out_cell.location[2]
                    )

                    if not build_mat:
                        outs[out_cell.index] = vals[in_cell.index]
                    else:
                        row_inds[i] = out_cell.index
                        col_inds[i] = in_cell.index
                        i += 1
            if build_mat and n_unvisited > 0:
                P += sp.csr_matrix(
                    (np.ones(n_unvisited), (row_inds, col_inds)),
                    shape=(self.n_cells, meshin.n_cells)
                )
            if not build_mat:
                return output
            return P

        for cell in self.tree.cells:
            x1m = min(cell.points[0].location[0], xF[0])
            y1m = min(cell.points[0].location[1], xF[1])

            x1p = max(cell.points[3].location[0], origin[0])
            y1p = max(cell.points[3].location[1], origin[1])
            if self._dim==3:
                z1m = min(cell.points[0].location[2], xF[2])
                z1p = max(cell.points[7].location[2], origin[2])
            overlapping_cell_inds = meshin.tree.find_overlapping_cells(x1m, x1p, y1m, y1p, z1m, z1p)
            n_overlap = overlapping_cell_inds.size()
            weights = <double *> malloc(n_overlap*sizeof(double))
            i = 0
            weight_sum = 0.0
            nnz_row = 0
            for in_cell_ind in overlapping_cell_inds:
                in_cell = meshin.tree.cells[in_cell_ind]
                x2m = in_cell.points[0].location[0]
                y2m = in_cell.points[0].location[1]
                z2m = in_cell.points[0].location[2]
                x2p = in_cell.points[3].location[0]
                y2p = in_cell.points[3].location[1]
                z2p = in_cell.points[7].location[2] if self._dim==3 else 0.0

                if x1m == xF[0] or x1p == origin[0]:
                    over_lap_vol = 1.0
                else:
                    over_lap_vol = min(x1p, x2p) - max(x1m, x2m)
                if y1m == xF[1] or y1p == origin[1]:
                    over_lap_vol *= 1.0
                else:
                    over_lap_vol *= min(y1p, y2p) - max(y1m, y2m)
                if self._dim==3:
                    if z1m == xF[2] or z1p == origin[2]:
                        over_lap_vol *= 1.0
                    else:
                        over_lap_vol *= min(z1p, z2p) - max(z1m, z2m)

                weights[i] = over_lap_vol
                if build_mat and weights[i] != 0.0:
                    nnz_row += 1
                    row_inds.push_back(in_cell_ind)

                weight_sum += weights[i]
                i += 1
            for i in range(n_overlap):
                weights[i] /= weight_sum
                if build_mat and weights[i] != 0.0:
                    all_weights.push_back(weights[i])

            if not build_mat:
                for i in range(n_overlap):
                    outs[cell.index] += vals[overlapping_cell_inds[i]]*weights[i]
            else:
                nnz_counter += nnz_row
                indptr.push_back(nnz_counter)

            free(weights)
            overlapping_cell_inds.clear()

        if not build_mat:
            return output
        return sp.csr_matrix((all_weights, row_inds, indptr), shape=(self.n_cells, meshin.n_cells))

    @cython.boundscheck(False)
    @cython.cdivision(True)
    def _vol_avg_to_tens(self, out_tens_mesh, values=None, output=None):
        cdef vector[int_t] *overlapping_cells
        cdef double *weights
        cdef double over_lap_vol
        cdef double x1m, x1p, y1m, y1p, z1m, z1p
        cdef double x2m, x2p, y2m, y2p, z2m, z2p
        cdef double[:] origin
        cdef double[:] xF

        # first check if they have the same tensor base, as it makes it a lot easier...
        cdef int_t same_base
        try:
            same_base = (
                np.allclose(self.nodes_x, out_tens_mesh.nodes_x)
                and np.allclose(self.nodes_y, out_tens_mesh.nodes_y)
                and (self.dim == 2 or np.allclose(self.nodes_z, out_tens_mesh.nodes_z))
            )
        except ValueError:
            same_base = False

        if same_base:
            in_cell_inds = self._get_containing_cell_indexes(out_tens_mesh.cell_centers)
            # Every cell input cell is gauranteed to be a lower level than the output tenser mesh
            # therefore all weights a 1.0
            if values is not None:
                if output is None:
                    output = np.empty(out_tens_mesh.n_cells)
                output[:] = values[in_cell_inds]
                return output
            return sp.csr_matrix(
                (np.ones(out_tens_mesh.n_cells), (np.arange(out_tens_mesh.n_cells), in_cell_inds)),
                shape=(out_tens_mesh.n_cells, self.n_cells)
            )

        if self.dim == 2:
            origin = np.r_[self.origin, 0.0]
            xF = np.array([self._xs[-1], self._ys[-1], 0.0])
        else:
            origin = self._origin
            xF = np.array([self._xs[-1], self._ys[-1], self._zs[-1]])
        cdef c_Cell * in_cell

        cdef np.float64_t[:] vals = np.array([])
        cdef np.float64_t[::1, :, :] outs = np.array([[[]]])

        cdef vector[int_t] row_inds
        cdef vector[int_t] indptr
        cdef vector[double] all_weights
        cdef int_t nnz_row = 0
        cdef int_t nnz_counter = 0

        cdef double[:] nodes_x = out_tens_mesh.nodes_x
        cdef double[:] nodes_y = out_tens_mesh.nodes_y
        cdef double[:] nodes_z = np.array([0.0, 0.0])
        if self._dim==3:
            nodes_z = out_tens_mesh.nodes_z
        cdef int_t nx = len(nodes_x)-1
        cdef int_t ny = len(nodes_y)-1
        cdef int_t nz = len(nodes_z)-1

        cdef int_t build_mat = 1
        if values is not None:
            vals = values
            if output is None:
                output = np.empty((nx, ny, nz), order='F')
            else:
                output = output.reshape((nx, ny, nz), order='F')
            output[:] = 0
            outs = output

            build_mat = 0
        if build_mat:
            indptr.push_back(0)

        cdef int_t ix, iy, iz, in_cell_ind, i
        cdef int_t n_overlap
        cdef double weight_sum

        #for cell in self.tree.cells:
        for iz in range(nz):
            z1m = min(nodes_z[iz], xF[2])
            z1p = max(nodes_z[iz+1], origin[2])
            for iy in range(ny):
                y1m = min(nodes_y[iy], xF[1])
                y1p = max(nodes_y[iy+1], origin[1])
                for ix in range(nx):
                    x1m = min(nodes_x[ix], xF[0])
                    x1p = max(nodes_x[ix+1], origin[0])
                    overlapping_cell_inds = self.tree.find_overlapping_cells(x1m, x1p, y1m, y1p, z1m, z1p)
                    n_overlap = overlapping_cell_inds.size()
                    weights = <double *> malloc(n_overlap*sizeof(double))
                    i = 0
                    weight_sum = 0.0
                    nnz_row = 0
                    for in_cell_ind in overlapping_cell_inds:
                        in_cell = self.tree.cells[in_cell_ind]
                        x2m = in_cell.points[0].location[0]
                        y2m = in_cell.points[0].location[1]
                        z2m = in_cell.points[0].location[2]
                        x2p = in_cell.points[3].location[0]
                        y2p = in_cell.points[3].location[1]
                        z2p = in_cell.points[7].location[2] if self._dim==3 else 0.0

                        if x1m == xF[0] or x1p == origin[0]:
                            over_lap_vol = 1.0
                        else:
                            over_lap_vol = min(x1p, x2p) - max(x1m, x2m)
                        if y1m == xF[1] or y1p == origin[1]:
                            over_lap_vol *= 1.0
                        else:
                            over_lap_vol *= min(y1p, y2p) - max(y1m, y2m)
                        if self._dim==3:
                            if z1m == xF[2] or z1p == origin[2]:
                                over_lap_vol *= 1.0
                            else:
                                over_lap_vol *= min(z1p, z2p) - max(z1m, z2m)

                        weights[i] = over_lap_vol
                        if build_mat and weights[i] != 0.0:
                            nnz_row += 1
                            row_inds.push_back(in_cell_ind)
                        weight_sum += weights[i]
                        i += 1
                    for i in range(n_overlap):
                        weights[i] /= weight_sum
                        if build_mat and weights[i] != 0.0:
                            all_weights.push_back(weights[i])

                    if not build_mat:
                        for i in range(n_overlap):
                            outs[ix, iy, iz] += vals[overlapping_cell_inds[i]]*weights[i]
                    else:
                        nnz_counter += nnz_row
                        indptr.push_back(nnz_counter)

                    free(weights)
                    overlapping_cell_inds.clear()

        if not build_mat:
            return output.reshape(-1, order='F')
        return sp.csr_matrix((all_weights, row_inds, indptr), shape=(out_tens_mesh.n_cells, self.n_cells))

    @cython.boundscheck(False)
    @cython.cdivision(True)
    def _vol_avg_from_tens(self, in_tens_mesh, values=None, output=None):
        cdef double *weights
        cdef double over_lap_vol
        cdef double x1m, x1p, y1m, y1p, z1m, z1p
        cdef double x2m, x2p, y2m, y2p, z2m, z2p
        cdef int_t ix, ix1, ix2, iy, iy1, iy2, iz, iz1, iz2
        cdef double[:] origin = in_tens_mesh.origin
        cdef double[:] xF

        # first check if they have the same tensor base, as it makes it a lot easier...
        cdef int_t same_base
        try:
            same_base = (
                np.allclose(self.nodes_x, in_tens_mesh.nodes_x)
                and np.allclose(self.nodes_y, in_tens_mesh.nodes_y)
                and (self.dim == 2 or np.allclose(self.nodes_z, in_tens_mesh.nodes_z))
            )
        except ValueError:
            same_base = False


        if same_base:
            out_cell_inds = self._get_containing_cell_indexes(in_tens_mesh.cell_centers)
            ws = in_tens_mesh.cell_volumes/self.cell_volumes[out_cell_inds]
            if values is not None:
                if output is None:
                    output = np.empty(self.n_cells)
                output[:] = np.bincount(out_cell_inds, ws*values)
                return output
            return sp.csr_matrix(
                (ws, (out_cell_inds, np.arange(in_tens_mesh.n_cells))),
                shape=(self.n_cells, in_tens_mesh.n_cells)
            )


        cdef np.float64_t[:] nodes_x = in_tens_mesh.nodes_x
        cdef np.float64_t[:] nodes_y = in_tens_mesh.nodes_y
        cdef np.float64_t[:] nodes_z = np.array([0.0, 0.0])
        if self._dim == 3:
            nodes_z = in_tens_mesh.nodes_z
        cdef int_t nx = len(nodes_x)-1
        cdef int_t ny = len(nodes_y)-1
        cdef int_t nz = len(nodes_z)-1

        cdef double * dx
        cdef double * dy
        cdef double * dz

        if self.dim == 2:
            xF = np.array([nodes_x[-1], nodes_y[-1]])
        else:
            xF = np.array([nodes_x[-1], nodes_y[-1], nodes_z[-1]])

        cdef np.float64_t[::1, :, :] vals = np.array([[[]]])
        cdef np.float64_t[:] outs = np.array([])

        cdef int_t build_mat = 1
        if values is not None:
            vals = values.reshape((nx, ny, nz), order='F')
            if output is None:
                output = np.empty(self.n_cells)
            output[:] = 0
            outs = output

            build_mat = 0

        cdef vector[int_t] row_inds
        cdef vector[double] all_weights
        cdef np.int64_t[:] indptr = np.zeros(self.n_cells+1, dtype=np.int64)
        cdef int_t nnz_counter = 0
        cdef int_t nnz_row = 0

        cdef int_t nx_overlap, ny_overlap, nz_overlap, n_overlap
        cdef int_t i
        cdef double weight_sum

        for cell in self.tree.cells:
            x1m = min(cell.points[0].location[0], xF[0])
            y1m = min(cell.points[0].location[1], xF[1])

            x1p = max(cell.points[3].location[0], origin[0])
            y1p = max(cell.points[3].location[1], origin[1])
            if self._dim==3:
                z1m = min(cell.points[0].location[2], xF[2])
                z1p = max(cell.points[7].location[2], origin[2])
            # then need to find overlapping cells of TensorMesh...
            ix1 = max(_bisect_left(nodes_x, x1m) - 1, 0)
            ix2 = min(_bisect_right(nodes_x, x1p), nx)
            iy1 = max(_bisect_left(nodes_y, y1m) - 1, 0)
            iy2 = min(_bisect_right(nodes_y, y1p), ny)
            if self._dim==3:
                iz1 = max(_bisect_left(nodes_z, z1m) - 1, 0)
                iz2 = min(_bisect_right(nodes_z, z1p), nz)
            else:
                iz1 = 0
                iz2 = 1
            nx_overlap = ix2-ix1
            ny_overlap = iy2-iy1
            nz_overlap = iz2-iz1
            n_overlap = nx_overlap*ny_overlap*nz_overlap
            weights = <double *> malloc(n_overlap*sizeof(double))

            dx = <double *> malloc(nx_overlap*sizeof(double))
            for ix in range(ix1, ix2):
                x2m = nodes_x[ix]
                x2p = nodes_x[ix+1]
                if x1m == xF[0] or x1p == origin[0]:
                    dx[ix-ix1] = 1.0
                else:
                    dx[ix-ix1] = min(x1p, x2p) - max(x1m, x2m)

            dy = <double *> malloc(ny_overlap*sizeof(double))
            for iy in range(iy1, iy2):
                y2m = nodes_y[iy]
                y2p = nodes_y[iy+1]
                if y1m == xF[1] or y1p == origin[1]:
                    dy[iy-iy1] = 1.0
                else:
                    dy[iy-iy1] = min(y1p, y2p) - max(y1m, y2m)

            dz = <double *> malloc(nz_overlap*sizeof(double))
            for iz in range(iz1, iz2):
                z2m = nodes_z[iz]
                z2p = nodes_z[iz+1]
                if self._dim==3:
                    if z1m == xF[2] or z1p == origin[2]:
                        dz[iz-iz1] = 1.0
                    else:
                        dz[iz-iz1] = min(z1p, z2p) - max(z1m, z2m)
                else:
                    dz[iz-iz1] = 1.0

            i = 0
            weight_sum = 0.0
            nnz_row = 0
            for iz in range(iz1, iz2):
                for iy in range(iy1, iy2):
                    for ix in range(ix1, ix2):
                        in_cell_ind = ix + (iy + iz*ny)*nx
                        weights[i] = dx[ix-ix1]*dy[iy-iy1]*dz[iz-iz1]
                        if build_mat and weights[i] != 0.0:
                            nnz_row += 1
                            row_inds.push_back(in_cell_ind)

                        weight_sum += weights[i]
                        i += 1

            for i in range(n_overlap):
                weights[i] /= weight_sum
                if build_mat and weights[i] != 0.0:
                    all_weights.push_back(weights[i])

            if not build_mat:
                i = 0
                for iz in range(iz1, iz2):
                    for iy in range(iy1, iy2):
                        for ix in range(ix1, ix2):
                            outs[cell.index] += vals[ix, iy, iz]*weights[i]
                            i += 1
            else:
                nnz_counter += nnz_row
                indptr[cell.index+1] = nnz_counter

            free(weights)
            free(dx)
            free(dy)
            free(dz)

        if not build_mat:
            return output
        return sp.csr_matrix((all_weights, row_inds, indptr), shape=(self.n_cells, in_tens_mesh.n_cells))

    def get_overlapping_cells(self, rectangle):
        cdef double xm, ym, zm, xp, yp, zp
        cdef double[:] origin = self._origin
        cdef double[:] xF
        if self.dim == 2:
            xF = np.array([self._xs[-1], self._ys[-1]])
        else:
            xF = np.array([self._xs[-1], self._ys[-1], self._zs[-1]])
        xm = min(rectangle[0], xF[0])
        xp = max(rectangle[1], origin[0])
        ym = min(rectangle[2], xF[1])
        yp = max(rectangle[3], origin[1])
        if self.dim==3:
            zm = min(rectangle[4], xF[2])
            zp = max(rectangle[5], origin[2])
        else:
            zm = 0.0
            zp = 0.0
        return self.tree.find_overlapping_cells(xm, xp, ym, yp, zm, zp)


cdef inline double _clip01(double x) nogil:
    return min(1, max(x, 0))
