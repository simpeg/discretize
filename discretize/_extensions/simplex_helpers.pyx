# distutils: language=c++
# cython: embedsignature=True, language_level=3
# cython: linetrace=True

from libcpp.pair cimport pair
from libcpp.unordered_map cimport unordered_map
from cython.operator cimport dereference
cimport cython
cimport numpy as np

cdef extern from "triplet.h":
    cdef cppclass triplet[T, U, V]:
        T v1
        U v2
        V v3
        triplet()
        triplet(T, U, V)
import numpy as np

ctypedef fused ints:
    size_t
    np.int32_t
    np.int64_t

ctypedef fused pointers:
    size_t
    np.intp_t
    np.int32_t
    np.int64_t

@cython.boundscheck(False)
def _build_faces_edges(ints[:, :] simplices):
    # the node index in each simplex must be in increasing order
    cdef:
        int dim = simplices.shape[1] - 1
        ints n_simplex = simplices.shape[0]
        ints[:] simplex
        ints v1, v2, v3

        unordered_map[pair[ints, ints], ints] edges
        pair[ints, ints] edge
        ints n_edges = 0
        ints edges_per_simplex = 3 if dim==2 else 6
        ints[:, :] simplex_edges

        unordered_map[triplet[ints, ints, ints], ints] faces
        triplet[ints, ints, ints] face
        ints n_faces = 0
        ints faces_per_simplex = dim + 1
        ints[:, :] simplex_faces

    if ints is size_t:
        int_type = np.uintp
    elif ints is np.int32_t:
        int_type = np.int32
    elif ints is np.int64_t:
        int_type = np.int64

    simplex_edges = np.empty((n_simplex, edges_per_simplex), dtype=int_type)

    if dim == 3:
        simplex_faces = np.empty((n_simplex, faces_per_simplex), dtype=int_type)
    else:
        simplex_faces = simplex_edges

    cdef ints[:,:] edge_pairs = np.array(
        [[1, 2], [0, 2], [0, 1], [0, 3], [1, 3], [2, 3]],
        dtype=int_type
    )

    for i_simp in range(n_simplex):
        simplex = simplices[i_simp]

        # build edges
        for i_edge in range(edges_per_simplex):

            v1 = simplex[edge_pairs[i_edge, 0]]
            v2 = simplex[edge_pairs[i_edge, 1]]
            edge = pair[ints, ints](v1, v2)

            edge_search = edges.find(edge)
            if edge_search != edges.end():
                ind = dereference(edge_search).second
            else:
                ind = n_edges
                edges[edge] = ind
                n_edges += 1
            simplex_edges[i_simp, i_edge] = ind

        # build faces in 3D
        if dim == 3:
            for i_face in range(4):
                if i_face == 0:
                    v1 = simplex[1]
                    v2 = simplex[2]
                    v3 = simplex[3]
                elif i_face == 1:
                    v1 = simplex[0]
                    v2 = simplex[2]
                    v3 = simplex[3]
                elif i_face == 2:
                    v1 = simplex[0]
                    v2 = simplex[1]
                    v3 = simplex[3]
                else:
                    v1 = simplex[0]
                    v2 = simplex[1]
                    v3 = simplex[2]
                face = triplet[ints, ints, ints](v1, v2, v3)

                face_search = faces.find(face)
                if face_search != faces.end():
                    ind = dereference(face_search).second
                else:
                    ind = n_faces
                    faces[face] = ind
                    n_faces += 1
                simplex_faces[i_simp, i_face] = ind

    cdef ints[:, :] _edges = np.empty((n_edges, 2), dtype=int_type)
    for edge_it in edges:
        _edges[edge_it.second, 0] = edge_it.first.first
        _edges[edge_it.second, 1] = edge_it.first.second

    cdef ints[:, :] _faces
    if dim == 3:
        _faces =  np.empty((n_faces, 3), dtype=int_type)
        for face_it in faces:
            _faces[face_it.second, 0] = face_it.first.v1
            _faces[face_it.second, 1] = face_it.first.v2
            _faces[face_it.second, 2] = face_it.first.v3
    else:
        _faces = _edges

    cdef ints[:, :] face_edges
    cdef ints[:] _face

    if dim == 3:
        face_edges = np.empty((n_faces, 3), dtype=int_type)
        for i_face in range(n_faces):
            _face = _faces[i_face]
            # get indices of each edge in the face
            # 3 edges per face
            for i_edge in range(3):
                if i_edge == 0:
                    v1 = _face[1]
                    v2 = _face[2]
                elif i_edge == 1:
                    v1 = _face[0]
                    v2 = _face[2]
                elif i_edge == 2:
                    v1 = _face[0]
                    v2 = _face[1]
                # because of how faces were constructed, v1 < v2 always
                edge = pair[ints, ints](v1, v2)
                ind = edges[edge]
                face_edges[i_face, i_edge] = ind
    else:
        face_edges = np.empty((1, 1), dtype=int_type)

    return simplex_faces, _faces, simplex_edges, _edges, face_edges

@cython.boundscheck(False)
def _build_adjacency(ints[:, :] simplex_faces, n_faces):
    cdef:
        size_t n_cells = simplex_faces.shape[0]
        int dim = simplex_faces.shape[1] - 1
        np.int64_t[:, :] neighbors
        np.int64_t[:] visited
        ints[:] simplex

        ints i_cell, j, k, i_face, i_other

    if ints is size_t:
        int_type = np.uintp
    elif ints is np.int32_t:
        int_type = np.int32
    elif ints is np.int64_t:
        int_type = np.int64

    neighbors = np.full((n_cells, dim + 1), -1, dtype=np.int64)
    visited = np.full((n_faces), -1, dtype=np.int64)

    for i_cell in range(n_cells):
        simplex = simplex_faces[i_cell]
        for j in range(dim + 1):
            i_face = simplex[j]
            i_other = visited[i_face]
            if i_other == -1:
                visited[i_face] = i_cell
            else:
                neighbors[i_cell, j] = i_other
                k = 0
                while (k < dim + 1) and (simplex_faces[i_other, k] != i_face):
                    k += 1
                neighbors[i_other, k] = i_cell
    return neighbors

@cython.boundscheck(False)
@cython.linetrace(False)
cdef void _compute_bary_coords(
    np.float64_t[:] point,
    np.float64_t[:, :] Tinv,
    np.float64_t[:] shift,
    np.float64_t * bary
) nogil:
    cdef:
        int dim = point.shape[0]
        int i, j

    bary[dim] = 1.0
    for i in range(dim):
        bary[i] = 0.0
        for j in range(dim):
            bary[i] += Tinv[i, j] * (point[j] - shift[j])
        bary[dim] -= bary[i]

@cython.boundscheck(False)
def _directed_search(
    np.float64_t[:, :] locs,
    pointers[:] nearest_cc,
    np.float64_t[:, :] nodes,
    ints[:, :] simplex_nodes,
    np.int64_t[:, :] neighbors,
    np.float64_t[:, :, :] transform,
    np.float64_t[:, :] shift,
    bint return_bary=True
):
    cdef:
        int i, j
        pointers i_simp
        int n_locs = locs.shape[0], dim = locs.shape[1]
        int max_directed = 1 + simplex_nodes.shape[0] // 4
        int i_directed
        np.float64_t eps = 1E-15
        np.int64_t[:] inds = np.full(len(locs), -1, dtype=np.int64)
        np.float64_t[:, :] all_barys = np.empty((1, 1), dtype=np.float64)
        np.float64_t barys[4]
        np.float64_t[:] loc
        np.float64_t[:, :] Tinv
        np.float64_t[:] rD
    if return_bary:
        all_barys = np.empty((len(locs), dim+1), dtype=np.float64)

    for i in range(n_locs):
        loc = locs[i]

        i_simp = nearest_cc[i]  # start at the nearest cell center
        i_directed = 0
        while i_directed < max_directed:
            Tinv = transform[i_simp]
            rD = shift[i_simp]
            _compute_bary_coords(loc, Tinv, rD, barys)
            j = 0
            is_inside = True
            while j <= dim:
                if barys[j] < -eps:
                    is_inside = False
                    # if not -1, move towards neighbor
                    if neighbors[i_simp, j] != -1:
                        i_simp = neighbors[i_simp, j]
                        break
                j += 1
            # If inside, I found my container
            if is_inside:
                break
            # Else, if I cycled through every bary
            # without breaking out of the above loop, that means I'm completely outside
            elif j == dim + 1:
                i_simp = -1
                break
            i_directed += 1

        if i_directed == max_directed:
            # made it through the whole loop without breaking out
            # Mark as failed
            i_simp = -2
        inds[i] = i_simp
        if return_bary:
            for j in range(dim+1):
                all_barys[i, j] = barys[j]

    if return_bary:
        return np.array(inds), np.array(all_barys)
    return np.array(inds)
