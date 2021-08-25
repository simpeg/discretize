import numpy as np
import scipy.sparse as sp
from discretize.utils import Identity
from discretize.base import BaseMesh
from discretize._extensions.simplex_helpers import _build_faces_edges, _build_adjacency


class SimplexMesh(BaseMesh):
    def __init__(self, *args, **kwargs):
        if len(args) == 1:
            # Assume args was a Delaunay triangulation from scipy
            triang = args[0]
            nodes = triang.points
            simplices = triang.simplices
        elif len(args) == 2:
            # Assume args was a tuple of (nodes, simplices)
            nodes, simplices = args
        else:
            raise TypeError("unsupport input")

        self._nodes = nodes
        self._simplices = simplices

        if self.cell_volumes.min() == 0.0:
            raise ValueError("Triangulation contains degenerate simplices")

        # build faces from simplices
        self._number()

        # build node to simplex lookup
        # _, first = np.unique(simplices, return_index=True)
        # self._node_to_simplex, _ = np.unravel_index(first, simplices.shape)

        # build a cKDTree for fast simplex lookup
        # self._lookup_tree = KDTree(self.cell_centers)

        # build adjacency graph
        # self._build_adjacency()

    def _number(self):
        items = _build_faces_edges(self._simplices)
        self._simplex_faces = np.array(items[0])
        self._faces = np.array(items[1])
        self._simplex_edges = np.array(items[2])
        self._edges = np.array(items[3])
        self._n_faces = self._faces.shape[0]
        self._n_edges = self._edges.shape[0]
        if self.dim == 3:
            self._face_edges = np.array(items[4])

    @property
    def neighbors(self):
        if getattr(self, "_neighbors", None) is None:
            self._neighbors = np.array(_build_adjacency(self._simplex_faces, self.n_faces))
        return self._neighbors

    @property
    def dim(self):
        return self.nodes.shape[-1]

    @property
    def n_nodes(self):
        return self._nodes.shape[0]

    @property
    def nodes(self):
        return self._nodes

    @property
    def n_cells(self):
        return self._simplices.shape[0]

    @property
    def cell_centers(self):
        return np.mean(self.nodes[self._simplices], axis=1)

    @property
    def cell_volumes(self):
        if getattr(self, "_cell_volumes", None) is None:
            simplex_nodes = self._nodes[self._simplices]
            mats = np.pad(simplex_nodes, ((0, 0), (0, 0), (0, 1)), constant_values=1)
            V1 = np.abs(np.linalg.det(mats))
            V1 /= 6 if self.dim == 3 else 2
            self._cell_volumes = V1
        return self._cell_volumes

    @property
    def n_edges(self):
        return self._n_edges

    @property
    def edges(self):
        return np.mean(self.nodes[self._edges], axis=1)

    @property
    def edge_tangents(self):
        tangents = np.diff(self.nodes[self._edges], axis=1).squeeze()
        tangents /= np.linalg.norm(tangents, axis=-1)[:, None]
        return tangents

    @property
    def edge_lengths(self):
        return np.linalg.norm(np.diff(self.nodes[self._edges], axis=1).squeeze(), axis=-1)

    @property
    def n_faces(self):
        return self._n_faces

    @property
    def faces(self):
        return np.mean(self.nodes[self._faces], axis=1)

    @property
    def face_areas(self):
        if self.dim == 2:
            return self.edge_lengths
        else:
            face_nodes = self._nodes[self._faces]
            v01 = face_nodes[:, 1] - face_nodes[:, 0]
            v02 = face_nodes[:, 2] - face_nodes[:, 0]
            areas = np.linalg.norm(np.cross(v01, v02), axis=1) / 2
            return areas

    @property
    def face_normals(self):
        if self.dim == 2:
            # Here we just choose a direction that is perpendicular to
            # the edge tangents
            tangents = self.edge_tangents
            # take the larger of the two elements as a divisor
            i_max = np.argmax(np.abs(tangents), axis=1)

            mins = np.take_along_axis(tangents, np.expand_dims(1-i_max, axis=-1), axis=-1)
            maxs = np.take_along_axis(tangents, np.expand_dims(i_max, axis=-1), axis=-1)
            normals = np.ones_like(tangents)
            np.put_along_axis(normals, np.expand_dims(i_max, axis=-1), -mins/maxs,  axis=-1)
            normals /= np.linalg.norm(normals, axis=-1)[:, None]
            return normals
        else:
            # define normal as |01 x 02|
            # therefore clockwise path about the normal is 0->1->2->0
            face_nodes = self._nodes[self._faces]
            v01 = face_nodes[:, 1] - face_nodes[:, 0]
            v02 = face_nodes[:, 2] - face_nodes[:, 0]
            normal = np.cross(v01, v02)
            normal /= np.linalg.norm(normal, axis=1)[:, None]
            return normal

    @property
    def face_divergence(self):
        areas = self.face_areas
        normals = self.face_normals

        # approx outward normals (to figure out if face normal is opposite the outward direction)
        test = self.faces[self._simplex_faces] - self.cell_centers[:, None, :]
        dirs = np.einsum('ijk,ijk->ij', normals[self._simplex_faces], test)

        Aijs = areas[self._simplex_faces] / self.cell_volumes[:, None]
        Aijs[dirs < 0] *= -1

        Aijs = Aijs.reshape(-1)
        ind_ptr = (self.dim + 1) * np.arange(self.n_cells+1)
        col_inds = self._simplex_faces.reshape(-1)
        D = sp.csr_matrix((Aijs, col_inds, ind_ptr), shape=(self.n_cells, self.n_faces))
        return D

    @property
    def nodal_gradient(self):
        ind_ptr = 2 * np.arange(self.n_edges+1)
        col_inds = self._edges.reshape(-1)
        Aijs = ((1.0/self.edge_lengths[:, None]) * [-1, 1]).reshape(-1)

        return sp.csr_matrix((Aijs, col_inds, ind_ptr), shape=(self.n_edges, self.n_nodes))

    @property
    def edge_curl(self):
        dim = self.dim
        n_edges = self.n_edges
        if dim == 2:
            face_edges = self._simplex_edges
            face_nodes = self.nodes[self._simplices]
            face_areas = self.cell_volumes
            n_faces = self.n_cells
        else:
            face_edges = self._face_edges
            face_nodes = self.nodes[self._faces]
            face_areas = self.face_areas
            n_faces = self.n_faces

        ind_ptr = 3 * np.arange(n_faces + 1)
        col_inds = face_edges.reshape(-1)

        # edge tangents point from lower node to higher node
        # clockwise about the face normal on face is path from 0 -> 1 -> 2 -> 0
        # so for each face can take the dot product of the path with the edge tangent

        l01 = face_nodes[:, 1] - face_nodes[:, 0]  # path opposite node 2
        l12 = face_nodes[:, 2] - face_nodes[:, 1]  # path opposite node 0
        l20 = face_nodes[:, 0] - face_nodes[:, 2]  # path opposite node 1

        face_path_tangents = self.edge_tangents[face_edges]

        if dim == 2:
            # need to do an adjustment in 2D for the places where the simplices
            # are not oriented counter clockwise about the +z axis
            # cp = np.cross(l01, -l20)
            # cp is a bunch of 1s (where simplices are CCW) and -1s (where simplices are CW)
            # (but we take the sign here to guard against numerical precision)
            cp = np.sign(np.cross(l20, l01))
            face_areas = face_areas * cp
            # don't due *= here

        Aijs = (np.c_[
            np.einsum('ij,ij->i', face_path_tangents[:, 0], l12),
            np.einsum('ij,ij->i', face_path_tangents[:, 1], l20),
            np.einsum('ij,ij->i', face_path_tangents[:, 2], l01),
        ] / face_areas[:, None]).reshape(-1)

        C = sp.csr_matrix((Aijs, col_inds, ind_ptr), shape=(n_faces, n_edges))

        return C

    def __validate_model(self, model, invert_model=False):
        n_cells = self.n_cells
        dim = self.dim
        # determines the tensor type of the model and reshapes it properly
        model = np.atleast_1d(model)
        n_aniso = ((dim + 1) * dim)//2
        if model.size == n_aniso * n_cells:
            # model is fully anisotropic
            # reshape it into a stack of dim x dim matrices
            if model.ndim == 1:
                model = model.reshape((-1, n_aniso), order='F')
            vals = model
            if self.dim == 2:
                model = np.stack(
                    [[vals[:, 0], vals[:, 2]], [vals[:, 2], vals[:, 1]]]
                ).transpose((2, 0, 1))
            else:
                model = np.stack(
                    [
                        [vals[:, 0], vals[:, 3], vals[:, 4]],
                        [vals[:, 3], vals[:, 1], vals[:, 5]],
                        [vals[:, 4], vals[:, 5], vals[:, 2]]
                    ]
                ).transpose((2, 0, 1))
        elif model.size == dim * n_cells:
            if model.ndim == 1:
                model = model.reshape((n_cells, dim), order='F')
            model = model.reshape(-1)

        if invert_model:
            if model.ndim == 1:
                model = 1.0/model
            else:
                model = np.linalg.inv(model)
        return model

    def get_face_inner_product(self, model=None, invert_model=False, invert_matrix=False):
        if invert_matrix:
            raise NotImplementedError(
                "The inverse of the inner product matrix with a tetrahedral mesh is not supported."
            )

        normals = self.face_normals
        dim = self.dim
        n_cells = self.n_cells
        n_faces = self.n_faces

        Ps = []

        # Precalc indptr and values for the projection matrix
        P_indptr = np.arange(dim * n_cells + 1)
        ones = np.ones(dim * n_cells)
        V = np.sqrt(self.cell_volumes / (dim + 1))

        # precalculate indices for the block diagonal matrix
        d = np.ones(dim, dtype=int)[:, None] * np.arange(dim)
        t = np.arange(n_cells)
        T_col_inds = (d + t[:, None, None] * dim).reshape(-1)
        T_ind_ptr = dim * np.arange(dim * n_cells + 1)

        for i in range(dim + 1):
            # matrix which selects the faces associated with node i of each simplex...
            face_inds = np.c_[self._simplex_faces[:, :i], self._simplex_faces[:, i+1:]]
            P_col_inds = face_inds.reshape(-1)
            P = sp.csr_matrix(
                (ones, P_col_inds, P_indptr),
                shape=(dim * n_cells, n_faces)
            )

            face_normals = normals[face_inds]
            trans_inv = V[:, None, None] * np.linalg.inv(face_normals)
            T = sp.csr_matrix(
                (trans_inv.reshape(-1), T_col_inds, T_ind_ptr),
                shape=(dim * n_cells, dim * n_cells)
            )
            Ps.append(T @ P)

        if model is None:
            Mu = Identity()
        else:
            model = self.__validate_model(model, invert_model)
            if model.size == 1:
                Mu = sp.diags((model,), (0, ), shape=(dim * n_cells, dim * n_cells))
            elif model.size == n_cells:
                Mu = sp.diags(np.repeat(model, dim))
            elif model.size == dim * n_cells:
                # diagonally anisotropic model
                Mu = sp.diags(model)
            elif model.size == (dim * dim) * n_cells:
                Mu = sp.csr_matrix(
                    (model.reshape(-1), T_col_inds, T_ind_ptr),
                    shape=(dim * n_cells, dim * n_cells)
                )
            else:
                raise ValueError("Unrecognized size of model vector")

        A = np.sum([P.T @ Mu @ P for P in Ps])
        return A

    def get_edge_inner_product(self, model=None, invert_model=False, invert_matrix=False):
        if invert_matrix:
            raise NotImplementedError(
                "The inverse of the SimplexMesh's inner product matrix is not supported."
            )

        tangents = self.edge_tangents
        dim = self.dim
        n_cells = self.n_cells
        n_edges = self.n_edges
        Ps = []

        # Precalc indptr and values for the projection matrix
        P_indptr = np.arange(dim * n_cells + 1)
        ones = np.ones(dim * n_cells)

        if dim == 2:
            node_edges = np.array([
                [1, 2],
                [0, 2],
                [0, 1]
            ])
        elif dim == 3:
            node_edges = np.array([
                [1, 2, 3],
                [0, 2, 4],
                [0, 1, 5],
                [3, 4, 5]
            ])

        V = np.sqrt(self.cell_volumes/(dim + 1))
        # precalculate indices for the block diagonal matrix
        d = np.ones(dim, dtype=int)[:, None] * np.arange(dim)
        t = np.arange(n_cells)
        T_col_inds = (d + t[:, None, None]*dim).reshape(-1)
        T_ind_ptr = dim * np.arange(dim * n_cells + 1)

        for i in range(dim + 1):
            # matrix which selects the edges associated with node i of each simplex...
            edge_inds = np.take(self._simplex_edges, node_edges[i], axis=1)
            P_col_inds = edge_inds.reshape(-1)
            P = sp.csr_matrix((ones, P_col_inds, P_indptr), shape=(dim * n_cells, n_edges))

            edge_tangents = tangents[edge_inds]
            trans_inv = V[:, None, None] * np.linalg.inv(edge_tangents)
            T = sp.csr_matrix(
                (trans_inv.reshape(-1), T_col_inds, T_ind_ptr),
                shape=(dim * n_cells, dim * n_cells)
            )

            Ps.append(T @ P)

        if model is None:
            Mu = Identity()
        else:
            model = self.__validate_model(model, invert_model)
            if model.size == 1:
                Mu = sp.diags((model,), (0, ), shape=(dim * n_cells, dim * n_cells))
            elif model.size == n_cells:
                Mu = sp.diags(np.repeat(model, dim))
            elif model.size == dim * n_cells:
                # diagonally anisotropic model
                Mu = sp.diags(model)
            elif model.size == (dim * dim) * n_cells:
                Mu = sp.csr_matrix(
                    (model.reshape(-1), T_col_inds, T_ind_ptr),
                    shape=(dim * n_cells, dim * n_cells)
                )
            else:
                raise ValueError("Unrecognized size of model vector")

        A = np.sum([P.T @ Mu @ P for P in Ps])
        return A

    @property
    def average_node_to_cell(self):
        nodes_per_cell = self.dim + 1
        n_cells = self.n_cells

        ind_ptr = nodes_per_cell * np.arange(n_cells + 1)
        col_inds = self._simplices.reshape(-1)
        Aij = np.full(nodes_per_cell * n_cells, 1/nodes_per_cell)
        return sp.csr_matrix((Aij, col_inds, ind_ptr), shape=(n_cells, self.n_nodes))

    @property
    def average_face_to_cell_vector(self):
        normals = self.face_normals
        dim = self.dim
        n_cells = self.n_cells
        n_faces = self.n_faces

        Av = sp.csr_matrix(shape=(dim * n_cells, n_faces))
        # Precalc indptr and values for the projection matrix
        P_indptr = np.arange(dim * n_cells + 1)
        ones = np.ones(dim * n_cells)

        # precalculate indices for the block diagonal matrix
        d = np.ones(dim, dtype=int)[:, None] * np.arange(dim)
        t = np.arange(n_cells)
        T_col_inds = (d + t[:, None, None] * dim).reshape(-1)
        T_ind_ptr = dim * np.arange(dim * n_cells + 1)

        for i in range(dim + 1):
            # matrix which selects the faces associated with node i of each simplex...
            face_inds = np.c_[self._simplex_faces[:, :i], self._simplex_faces[:, i+1:]]
            P_col_inds = face_inds.reshape(-1)
            P = sp.csr_matrix(
                (ones, P_col_inds, P_indptr),
                shape=(dim * n_cells, n_faces)
            )

            face_normals = normals[face_inds]
            #face normals is now a block of dim x dim matrices...
            trans_inv = np.linalg.inv(face_normals) / (dim + 1)

            T = sp.csr_matrix(
                (trans_inv.reshape(-1), T_col_inds, T_ind_ptr),
                shape=(dim * n_cells, dim * n_cells)
            )
            Av = Av + (T @ P)
        return Av

    @property
    def average_edge_to_cell_vector(self):
        tangents = self.edge_tangents
        dim = self.dim
        n_cells = self.n_cells
        n_edges = self.n_edges

        Av = sp.csr_matrix(shape=(dim * n_cells, n_edges))
        # Precalc indptr and values for the projection matrix
        P_indptr = np.arange(dim * n_cells + 1)
        ones = np.ones(dim * n_cells)

        if dim == 2:
            node_edges = np.array([
                [1, 2],
                [0, 2],
                [0, 1]
            ])
        elif dim == 3:
            node_edges = np.array([
                [1, 2, 3],
                [0, 2, 4],
                [0, 1, 5],
                [3, 4, 5]
            ])

        # precalculate indices for the block diagonal matrix
        d = np.ones(dim, dtype=int)[:, None] * np.arange(dim)
        t = np.arange(n_cells)
        T_col_inds = (d + t[:, None, None]*dim).reshape(-1)
        T_ind_ptr = dim * np.arange(dim * n_cells + 1)

        for i in range(dim + 1):
            # matrix which selects the edges associated with node i of each simplex...
            edge_inds = np.take(self._simplex_edges, node_edges[i], axis=1)
            P_col_inds = edge_inds.reshape(-1)
            P = sp.csr_matrix((ones, P_col_inds, P_indptr), shape=(dim * n_cells, n_edges))

            edge_tangents = tangents[edge_inds]
            trans_inv = np.linalg.inv(edge_tangents) / (dim + 1)
            T = sp.csr_matrix(
                (trans_inv.reshape(-1), T_col_inds, T_ind_ptr),
                shape=(dim * n_cells, dim * n_cells)
            )
            Av = Av + T @ P
        return Av
