import discretize
import numpy as np


def check_for_diag_unbalance(mesh):
    avg = mesh.average_node_to_cell.T.tocsr()
    bad_nodes = []
    for i in range(mesh.n_nodes):
        cells_around_node = avg[i, :].indices
        levels = np.atleast_1d(mesh.cell_levels_by_index(cells_around_node))
        level_diff = max(levels) - min(levels)
        if level_diff >= 2:
            bad_nodes.append(i)
    bad_nodes = np.asarray(bad_nodes)
    return bad_nodes


def test_insert_cells_2D():
    mesh1 = discretize.TreeMesh([64, 64])
    mesh1.insert_cells([0.09, 0.09], -1, finalize=False)
    mesh1.insert_cells([0.09, 0.91], -1, finalize=False)
    mesh1.insert_cells([0.91, 0.09], -1, finalize=False)
    mesh1.insert_cells([0.91, 0.91], -1)

    bad_nodes = check_for_diag_unbalance(mesh1)
    assert len(bad_nodes) == 8

    mesh2 = discretize.TreeMesh([64, 64], diagonal_balance=True)
    mesh2.insert_cells([0.09, 0.09], -1, finalize=False)
    mesh2.insert_cells([0.09, 0.91], -1, finalize=False)
    mesh2.insert_cells([0.91, 0.09], -1, finalize=False)
    mesh2.insert_cells([0.91, 0.91], -1)

    bad_nodes = check_for_diag_unbalance(mesh2)
    assert len(bad_nodes) == 0


def test_insert_cells_3D():
    mesh1 = discretize.TreeMesh([64, 64, 64])
    mesh1.insert_cells([0.09, 0.09, 0.09], -1, finalize=False)
    mesh1.insert_cells([0.09, 0.91, 0.09], -1, finalize=False)
    mesh1.insert_cells([0.91, 0.09, 0.09], -1, finalize=False)
    mesh1.insert_cells([0.91, 0.91, 0.09], -1, finalize=False)
    mesh1.insert_cells([0.09, 0.09, 0.91], -1, finalize=False)
    mesh1.insert_cells([0.09, 0.91, 0.91], -1, finalize=False)
    mesh1.insert_cells([0.91, 0.09, 0.91], -1, finalize=False)
    mesh1.insert_cells([0.91, 0.91, 0.91], -1)

    bad_nodes = check_for_diag_unbalance(mesh1)
    assert len(bad_nodes) == 64

    mesh2 = discretize.TreeMesh([64, 64, 64], diagonal_balance=True)
    mesh2.insert_cells([0.09, 0.09, 0.09], -1, finalize=False)
    mesh2.insert_cells([0.09, 0.91, 0.09], -1, finalize=False)
    mesh2.insert_cells([0.91, 0.09, 0.09], -1, finalize=False)
    mesh2.insert_cells([0.91, 0.91, 0.09], -1, finalize=False)
    mesh2.insert_cells([0.09, 0.09, 0.91], -1, finalize=False)
    mesh2.insert_cells([0.09, 0.91, 0.91], -1, finalize=False)
    mesh2.insert_cells([0.91, 0.09, 0.91], -1, finalize=False)
    mesh2.insert_cells([0.91, 0.91, 0.91], -1)

    bad_nodes = check_for_diag_unbalance(mesh2)
    assert len(bad_nodes) == 0


def test_refine():
    mesh1 = discretize.TreeMesh([64, 64])

    def refine(cell):
        if np.sqrt(((np.r_[cell.center] - 0.5) ** 2).sum()) < 0.2:
            return 5
        return 2

    mesh1.refine(refine)
    bad_nodes = check_for_diag_unbalance(mesh1)

    assert len(bad_nodes) == 4

    mesh2 = discretize.TreeMesh([64, 64], diagonal_balance=True)
    mesh2.refine(refine)
    bad_nodes = check_for_diag_unbalance(mesh2)

    assert len(bad_nodes) == 0


def test_refine_box():
    mesh1 = discretize.TreeMesh([64, 64])
    mesh1.refine_box([0.4, 0.4], [0.6, 0.6], -1)
    bad_nodes = check_for_diag_unbalance(mesh1)

    assert len(bad_nodes) == 8

    mesh2 = discretize.TreeMesh([64, 64], diagonal_balance=True)
    mesh2.refine_box([0.4, 0.4], [0.6, 0.6], -1)
    bad_nodes = check_for_diag_unbalance(mesh2)

    assert len(bad_nodes) == 0


def test_refine_ball():
    mesh1 = discretize.TreeMesh([64, 64])
    mesh1.refine_ball([0.5, 0.5], [0.1], -1)
    bad_nodes = check_for_diag_unbalance(mesh1)

    assert len(bad_nodes) == 12

    mesh2 = discretize.TreeMesh([64, 64], diagonal_balance=True)
    mesh2.refine_ball([0.5, 0.5], [0.1], -1)
    bad_nodes = check_for_diag_unbalance(mesh2)

    assert len(bad_nodes) == 0


def test_refine_line():
    segments = np.array([[0.1, 0.3], [0.3, 0.9], [0.8, 0.9]])

    mesh1 = discretize.TreeMesh([64, 64])
    mesh1.refine_line(segments, -1)
    bad_nodes = check_for_diag_unbalance(mesh1)

    assert len(bad_nodes) == 7

    mesh2 = discretize.TreeMesh([64, 64], diagonal_balance=True)
    mesh2.refine_line(segments, -1)
    bad_nodes = check_for_diag_unbalance(mesh2)

    assert len(bad_nodes) == 0


def test_refine_triangle():
    triangle = np.array([[0.14, 0.31], [0.32, 0.96], [0.23, 0.87]])
    mesh1 = discretize.TreeMesh([64, 64], diagonal_balance=False)
    mesh1.refine_triangle(triangle, -1)
    bad_nodes = check_for_diag_unbalance(mesh1)

    assert len(bad_nodes) == 5

    mesh2 = discretize.TreeMesh([64, 64], diagonal_balance=True)
    mesh2.refine_triangle(triangle, -1)
    bad_nodes = check_for_diag_unbalance(mesh2)

    assert len(bad_nodes) == 0


def test_balance_out_unbalance_in():
    mesh1 = discretize.TreeMesh([64, 64], diagonal_balance=True)
    mesh1.insert_cells([0.09, 0.09], -1, finalize=False)
    mesh1.insert_cells([0.09, 0.91], -1, finalize=False)
    mesh1.insert_cells([0.91, 0.09], -1, finalize=False)
    mesh1.insert_cells([0.91, 0.91], -1)

    bad_nodes = check_for_diag_unbalance(mesh1)
    assert len(bad_nodes) == 0

    srl = mesh1.to_dict()
    mesh2 = discretize.TreeMesh.deserialize(srl)

    assert mesh1.equals(mesh2)
