import numpy as np
import discretize


def test_pizza_slice_counting():
    # pizza slice includes r=0, but doesn't wrap around azimuth
    # i.e. domain is a triangular pizza slice.
    mesh = discretize.CylindricalMesh([4, 3 * [np.pi / 6], 2])

    assert not mesh.is_wrapped
    assert mesh.includes_zero
    assert not mesh.is_symmetric

    assert mesh.shape_nodes == (5, 4, 3)
    assert mesh.n_nodes == 4 * 4 * 3 + 3
    assert mesh._shape_total_nodes == (5, 4, 3)
    assert mesh._n_total_nodes == 5 * 4 * 3

    assert mesh.shape_faces_x == (4, 3, 2)
    assert mesh.n_faces_x == 4 * 3 * 2
    assert mesh._shape_total_faces_x == (5, 3, 2)
    assert mesh._n_total_faces_x == 5 * 3 * 2
    assert mesh._n_hanging_faces_x == 3 * 2

    assert mesh.shape_faces_y == (4, 4, 2)
    assert mesh.n_faces_y == 4 * 4 * 2
    assert mesh._shape_total_faces_y == (4, 4, 2)
    assert mesh._n_total_faces_y == 4 * 4 * 2
    assert mesh._n_hanging_faces_y == 0

    assert mesh.shape_faces_z == (4, 3, 3)
    assert mesh.n_faces_z == 4 * 3 * 3
    assert mesh._shape_total_faces_z == (4, 3, 3)
    assert mesh._n_total_faces_z == 4 * 3 * 3
    assert mesh._n_hanging_faces_z == 0

    assert mesh.shape_edges_x == (4, 4, 3)
    assert mesh.n_edges_x == 4 * 4 * 3
    assert mesh._shape_total_edges_x == (4, 4, 3)
    assert mesh._n_total_edges_x == 4 * 4 * 3

    assert mesh.shape_edges_y == (4, 3, 3)
    assert mesh.n_edges_y == 4 * 3 * 3
    assert mesh._shape_total_edges_y == (5, 3, 3)
    assert mesh._n_total_edges_y == 5 * 3 * 3

    assert mesh.shape_edges_z == (5, 4, 2)
    assert mesh.n_edges_z == 4 * 4 * 2 + 2
    assert mesh._shape_total_edges_z == (5, 4, 2)
    assert mesh._n_total_edges_z == 5 * 4 * 2

    assert mesh.n_nodes == mesh.nodes.shape[0]
    assert mesh.n_faces == mesh.n_faces_x + mesh.n_faces_y + mesh.n_faces_z
    assert mesh.n_faces == mesh.faces.shape[0]
    assert mesh.n_edges == mesh.n_edges_x + mesh.n_edges_y + mesh.n_edges_z
    assert mesh.n_edges == mesh.edges.shape[0]

    assert mesh.n_faces == mesh.face_areas.shape[0]
    assert mesh.n_edges == mesh.edge_lengths.shape[0]
    assert mesh.n_cells == mesh.cell_volumes.shape[0]


def test_ring_counting():
    # domain fully discretizes the azimuthal dimension, but
    # does not include zero
    mesh = discretize.CylindricalMesh([4, 3, 2], origin=[1, 0, 0])

    assert mesh.is_wrapped
    assert not mesh.includes_zero
    assert not mesh.is_symmetric

    assert mesh.shape_nodes == (5, 3, 3)
    assert mesh.n_nodes == 5 * 3 * 3
    assert mesh._shape_total_nodes == (5, 4, 3)
    assert mesh._n_total_nodes == 5 * 4 * 3

    assert mesh.shape_faces_x == (5, 3, 2)
    assert mesh.n_faces_x == 5 * 3 * 2
    assert mesh._shape_total_faces_x == (5, 3, 2)
    assert mesh._n_total_faces_x == 5 * 3 * 2
    assert mesh._n_hanging_faces_x == 0

    assert mesh.shape_faces_y == (4, 3, 2)
    assert mesh.n_faces_y == 4 * 3 * 2
    assert mesh._shape_total_faces_y == (4, 4, 2)
    assert mesh._n_total_faces_y == 4 * 4 * 2
    assert mesh._n_hanging_faces_y == 4 * 2

    assert mesh.shape_faces_z == (4, 3, 3)
    assert mesh.n_faces_z == 4 * 3 * 3
    assert mesh._shape_total_faces_z == (4, 3, 3)
    assert mesh._n_total_faces_z == 4 * 3 * 3
    assert mesh._n_hanging_faces_z == 0

    assert mesh.shape_edges_x == (4, 3, 3)
    assert mesh.n_edges_x == 4 * 3 * 3
    assert mesh._shape_total_edges_x == (4, 4, 3)
    assert mesh._n_total_edges_x == 4 * 4 * 3

    assert mesh.shape_edges_y == (5, 3, 3)
    assert mesh.n_edges_y == 5 * 3 * 3
    assert mesh._shape_total_edges_y == (5, 3, 3)
    assert mesh._n_total_edges_y == 5 * 3 * 3

    assert mesh.shape_edges_z == (5, 3, 2)
    assert mesh.n_edges_z == 5 * 3 * 2
    assert mesh._shape_total_edges_z == (5, 4, 2)
    assert mesh._n_total_edges_z == 5 * 4 * 2

    assert mesh.n_nodes == mesh.nodes.shape[0]
    assert mesh.n_faces == mesh.n_faces_x + mesh.n_faces_y + mesh.n_faces_z
    assert mesh.n_faces == mesh.faces.shape[0]
    assert mesh.n_edges == mesh.n_edges_x + mesh.n_edges_y + mesh.n_edges_z
    assert mesh.n_edges == mesh.edges.shape[0]

    assert mesh.n_faces == mesh.face_areas.shape[0]
    assert mesh.n_edges == mesh.edge_lengths.shape[0]
    assert mesh.n_cells == mesh.cell_volumes.shape[0]


def test_cyl_tensor_counting():
    mesh = discretize.CylindricalMesh([4, 3 * [np.pi / 6], 2], origin=[1, 0, 0])

    assert not mesh.is_wrapped
    assert not mesh.includes_zero
    assert not mesh.is_symmetric

    assert mesh.shape_nodes == (5, 4, 3)
    assert mesh.n_nodes == 5 * 4 * 3
    assert mesh._shape_total_nodes == (5, 4, 3)
    assert mesh._n_total_nodes == 5 * 4 * 3

    assert mesh.shape_faces_x == (5, 3, 2)
    assert mesh.n_faces_x == 5 * 3 * 2
    assert mesh._shape_total_faces_x == (5, 3, 2)
    assert mesh._n_total_faces_x == 5 * 3 * 2
    assert mesh._n_hanging_faces_x == 0

    assert mesh.shape_faces_y == (4, 4, 2)
    assert mesh.n_faces_y == 4 * 4 * 2
    assert mesh._shape_total_faces_y == (4, 4, 2)
    assert mesh._n_total_faces_y == 4 * 4 * 2
    assert mesh._n_hanging_faces_y == 0

    assert mesh.shape_faces_z == (4, 3, 3)
    assert mesh.n_faces_z == 4 * 3 * 3
    assert mesh._shape_total_faces_z == (4, 3, 3)
    assert mesh._n_total_faces_z == 4 * 3 * 3
    assert mesh._n_hanging_faces_z == 0

    assert mesh.shape_edges_x == (4, 4, 3)
    assert mesh.n_edges_x == 4 * 4 * 3
    assert mesh._shape_total_edges_x == (4, 4, 3)
    assert mesh._n_total_edges_x == 4 * 4 * 3

    assert mesh.shape_edges_y == (5, 3, 3)
    assert mesh.n_edges_y == 5 * 3 * 3
    assert mesh._shape_total_edges_y == (5, 3, 3)
    assert mesh._n_total_edges_y == 5 * 3 * 3

    assert mesh.shape_edges_z == (5, 4, 2)
    assert mesh.n_edges_z == 5 * 4 * 2
    assert mesh._shape_total_edges_z == (5, 4, 2)
    assert mesh._n_total_edges_z == 5 * 4 * 2

    assert mesh.n_nodes == mesh.nodes.shape[0]
    assert mesh.n_faces == mesh.n_faces_x + mesh.n_faces_y + mesh.n_faces_z
    assert mesh.n_faces == mesh.faces.shape[0]
    assert mesh.n_edges == mesh.n_edges_x + mesh.n_edges_y + mesh.n_edges_z
    assert mesh.n_edges == mesh.edges.shape[0]

    assert mesh.n_faces == mesh.face_areas.shape[0]
    assert mesh.n_edges == mesh.edge_lengths.shape[0]
    assert mesh.n_cells == mesh.cell_volumes.shape[0]


def test_sym_ring_counting():
    # domain fully discretizes the azimuthal dimension, but
    # does not include zero
    mesh = discretize.CylindricalMesh([4, 1, 2], origin=[1, 0, 0])

    assert mesh.is_wrapped
    assert not mesh.includes_zero
    assert mesh.is_symmetric

    assert mesh.shape_nodes == (5, 0, 3)
    assert mesh.n_nodes == 0
    assert mesh._shape_total_nodes == (5, 1, 3)
    assert mesh._n_total_nodes == 5 * 1 * 3

    assert mesh.shape_faces_x == (5, 1, 2)
    assert mesh.n_faces_x == 5 * 1 * 2
    assert mesh._shape_total_faces_x == (5, 1, 2)
    assert mesh._n_total_faces_x == 5 * 1 * 2
    assert mesh._n_hanging_faces_x == 0

    assert mesh.shape_faces_y == (4, 0, 2)
    assert mesh.n_faces_y == 0
    assert mesh._shape_total_faces_y == (4, 1, 2)
    assert mesh._n_total_faces_y == 4 * 1 * 2
    assert mesh._n_hanging_faces_y == 4 * 2

    assert mesh.shape_faces_z == (4, 1, 3)
    assert mesh.n_faces_z == 4 * 1 * 3
    assert mesh._shape_total_faces_z == (4, 1, 3)
    assert mesh._n_total_faces_z == 4 * 1 * 3
    assert mesh._n_hanging_faces_z == 0

    assert mesh.shape_edges_x == (4, 0, 3)
    assert mesh.n_edges_x == 0
    assert mesh._shape_total_edges_x == (4, 1, 3)
    assert mesh._n_total_edges_x == 4 * 1 * 3

    assert mesh.shape_edges_y == (5, 1, 3)
    assert mesh.n_edges_y == 5 * 1 * 3
    assert mesh._shape_total_edges_y == (5, 1, 3)
    assert mesh._n_total_edges_y == 5 * 1 * 3

    assert mesh.shape_edges_z == (5, 0, 2)
    assert mesh.n_edges_z == 0
    assert mesh._shape_total_edges_z == (5, 1, 2)
    assert mesh._n_total_edges_z == 5 * 1 * 2

    assert mesh._n_total_nodes == mesh.nodes.shape[0]
    assert mesh.n_faces == mesh.n_faces_x + mesh.n_faces_y + mesh.n_faces_z
    assert mesh.n_faces == mesh.faces.shape[0]
    assert mesh.n_edges == mesh.n_edges_x + mesh.n_edges_y + mesh.n_edges_z
    assert mesh.n_edges == mesh.edges.shape[0]

    assert mesh.n_faces == mesh.face_areas.shape[0]
    assert mesh.n_edges == mesh.edge_lengths.shape[0]
    assert mesh.n_cells == mesh.cell_volumes.shape[0]


def test_sym_full_counting():
    mesh = discretize.CylindricalMesh([3, 1, 2])

    assert mesh.dim == 3
    assert mesh.shape_cells == (3, 1, 2)

    assert mesh.is_wrapped
    assert mesh.includes_zero
    assert mesh.is_symmetric

    assert mesh.shape_nodes == (3, 0, 3)
    assert mesh.n_nodes == 0
    assert mesh._shape_total_nodes == (3, 1, 3)
    assert mesh._n_total_nodes == 3 * 1 * 3

    assert mesh.shape_faces_x == (3, 1, 2)
    assert mesh.n_faces_x == 3 * 1 * 2
    assert mesh._shape_total_faces_x == (3, 1, 2)
    assert mesh._n_total_faces_x == 3 * 1 * 2
    assert mesh._n_hanging_faces_x == 1 * 2

    assert mesh.shape_faces_y == (3, 0, 2)
    assert mesh.n_faces_y == 3 * 0 * 2
    assert mesh._shape_total_faces_y == (3, 1, 2)
    assert mesh._n_total_faces_y == 3 * 1 * 2
    assert mesh._n_hanging_faces_y == 3 * 1 * 2

    assert mesh.shape_faces_z == (3, 1, 3)
    assert mesh.n_faces_z == 3 * 1 * 3
    assert mesh._shape_total_faces_z == (3, 1, 3)
    assert mesh._n_total_faces_z == 3 * 1 * 3
    assert mesh._n_hanging_faces_z == 0

    assert mesh.shape_edges_x == (3, 0, 3)
    assert mesh.n_edges_x == 3 * 0 * 3
    assert mesh._shape_total_edges_x == (3, 1, 3)
    assert mesh._n_total_edges_x == 3 * 1 * 3

    assert mesh.shape_edges_y == (3, 1, 3)
    assert mesh.n_edges_y == 3 * 1 * 3
    assert mesh._shape_total_edges_y == (3, 1, 3)
    assert mesh._n_total_edges_y == 3 * 1 * 3

    assert mesh.shape_edges_z == (3, 0, 2)
    assert mesh.n_edges_z == 0
    assert mesh._shape_total_edges_z == (3, 1, 2)
    assert mesh._n_total_edges_z == 3 * 1 * 2

    assert mesh._n_total_nodes == mesh.nodes.shape[0]
    assert mesh.n_faces == mesh.n_faces_x + mesh.n_faces_y + mesh.n_faces_z
    assert mesh.n_faces == mesh.faces.shape[0]
    assert mesh.n_edges == mesh.n_edges_x + mesh.n_edges_y + mesh.n_edges_z
    assert mesh.n_edges == mesh.edges.shape[0]

    assert mesh.n_faces == mesh.face_areas.shape[0]
    assert mesh.n_edges == mesh.edge_lengths.shape[0]
    assert mesh.n_cells == mesh.cell_volumes.shape[0]


def test_wrapped_counting():
    mesh = discretize.CylindricalMesh([3, 2, 2])

    assert mesh.dim == 3
    assert mesh.shape_cells == (3, 2, 2)

    assert mesh.is_wrapped
    assert mesh.includes_zero
    assert not mesh.is_symmetric

    assert mesh.shape_nodes == (4, 2, 3)
    assert mesh.n_nodes == 3 * 2 * 3 + 3
    assert mesh._shape_total_nodes == (4, 3, 3)
    assert mesh._n_total_nodes == 4 * 3 * 3

    assert mesh.shape_faces_x == (3, 2, 2)
    assert mesh.n_faces_x == 3 * 2 * 2
    assert mesh._shape_total_faces_x == (4, 2, 2)
    assert mesh._n_total_faces_x == 4 * 2 * 2
    assert mesh._n_hanging_faces_x == 2 * 2

    assert mesh.shape_faces_y == (3, 2, 2)
    assert mesh.n_faces_y == 3 * 2 * 2
    assert mesh._shape_total_faces_y == (3, 3, 2)
    assert mesh._n_total_faces_y == 3 * 3 * 2
    assert mesh._n_hanging_faces_y == 3 * 2

    assert mesh.shape_faces_z == (3, 2, 3)
    assert mesh.n_faces_z == 3 * 2 * 3
    assert mesh._shape_total_faces_z == (3, 2, 3)
    assert mesh._n_total_faces_z == 3 * 2 * 3
    assert mesh._n_hanging_faces_z == 0

    assert mesh.shape_edges_x == (3, 2, 3)
    assert mesh.n_edges_x == 3 * 2 * 3
    assert mesh._shape_total_edges_x == (3, 3, 3)
    assert mesh._n_total_edges_x == 3 * 3 * 3

    assert mesh.shape_edges_y == (3, 2, 3)
    assert mesh.n_edges_y == 3 * 2 * 3
    assert mesh._shape_total_edges_y == (4, 2, 3)
    assert mesh._n_total_edges_y == 4 * 2 * 3

    assert mesh.shape_edges_z == (4, 2, 2)
    assert mesh.n_edges_z == 3 * 2 * 2 + 2
    assert mesh._shape_total_edges_z == (4, 3, 2)
    assert mesh._n_total_edges_z == 4 * 3 * 2

    assert mesh.n_nodes == mesh.nodes.shape[0]
    assert mesh.n_faces == mesh.n_faces_x + mesh.n_faces_y + mesh.n_faces_z
    assert mesh.n_faces == mesh.faces.shape[0]
    assert mesh.n_edges == mesh.n_edges_x + mesh.n_edges_y + mesh.n_edges_z
    assert mesh.n_edges == mesh.edges.shape[0]

    assert mesh.n_faces == mesh.face_areas.shape[0]
    assert mesh.n_edges == mesh.edge_lengths.shape[0]
    assert mesh.n_cells == mesh.cell_volumes.shape[0]
