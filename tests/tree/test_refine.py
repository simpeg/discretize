import discretize
import numpy as np
import pytest


def test_2d_line():
    segments = np.array([[0.1, 0.3], [0.3, 0.9]])

    mesh = discretize.TreeMesh([64, 64])
    mesh.refine_line(segments, mesh.max_level)

    cells = mesh.get_cells_along_line(segments[0], segments[1])
    levels = mesh.cell_levels_by_index(cells)

    np.testing.assert_equal(levels, mesh.max_level)


def test_3d_line():
    segments = np.array([[0.1, 0.3, 0.2], [0.3, 0.9, 0.7]])

    mesh = discretize.TreeMesh([64, 64, 64])
    mesh.refine_line(segments, mesh.max_level)

    cells = mesh.get_cells_along_line(segments[0], segments[1])
    levels = mesh.cell_levels_by_index(cells)

    np.testing.assert_equal(levels, mesh.max_level)


def test_line_errors():
    mesh = discretize.TreeMesh([64, 64])
    segments2D = np.array([[0.1, 0.3], [0.3, 0.9]])
    segments3D = np.array([[0.1, 0.3, 0.2], [0.3, 0.9, 0.7]])

    # incorrect dimension
    with pytest.raises(ValueError):
        mesh.refine_line(segments3D, mesh.max_level, finalize=False)

    # incorrect number of levels
    with pytest.raises(ValueError):
        mesh.refine_line(segments2D, [mesh.max_level, 3], finalize=False)


def test_box_errors():
    mesh = discretize.TreeMesh([64, 64])
    x0s = np.array([0.1, 0.2])
    x0s2d = np.array([[0.1, 0.1], [0.5, 0.5]])
    x1s2d = np.array([[0.2, 0.3], [0.8, 0.9]])

    x0s3d = np.array([[0.1, 0.1, 0.1], [0.5, 0.5, 0.5]])
    x1s3d = np.array([[0.2, 0.3, 0.1], [0.8, 0.9, 0.75]])

    # incorrect dimension on x0
    with pytest.raises(ValueError):
        mesh.refine_box(x0s3d, x1s3d, [5, 5], finalize=False)

    # incorrect dimension on x1
    with pytest.raises(ValueError):
        mesh.refine_box(x0s2d, x1s3d, [5, 5], finalize=False)

    # incompatible shapes
    with pytest.raises(ValueError):
        mesh.refine_box(x0s, x1s2d, [5, 5], finalize=False)

    # incorrect number of levels
    with pytest.raises(ValueError):
        mesh.refine_box(x0s2d, x1s2d, [mesh.max_level], finalize=False)


def test_ball_errors():
    mesh = discretize.TreeMesh([64, 64])
    x0s2d = np.array([[0.1, 0.1], [0.5, 0.5]])
    x0s3d = np.array([[0.1, 0.1, 0.1], [0.5, 0.5, 0.5]])

    # incorrect dimension on x0
    with pytest.raises(ValueError):
        mesh.refine_ball(x0s3d, [1, 1], [5, 5], finalize=False)

    # incorrect number of radii
    with pytest.raises(ValueError):
        mesh.refine_ball(x0s2d, [1, 1, 2], [5, 5], finalize=False)

    # incorrect number of levels
    with pytest.raises(ValueError):
        mesh.refine_ball(x0s2d, [1, 1], [5, 5, 4], finalize=False)


def test_insert_errors():
    mesh = discretize.TreeMesh([64, 64])
    x0s2d = np.array([[0.1, 0.1], [0.5, 0.5]])
    x0s3d = np.array([[0.1, 0.1, 0.1], [0.5, 0.5, 0.5]])

    # incorrect dimension on points
    with pytest.raises(ValueError):
        mesh.insert_cells(x0s3d, [1, 1], finalize=False)

    # incorrect number of levels
    with pytest.raises(ValueError):
        mesh.insert_cells(x0s2d, [1, 1, 3], finalize=False)
