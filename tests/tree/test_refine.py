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


def test_triangle2d():
    # define a slower function that is surely accurate
    triangle = np.array([[0.14, 0.31], [0.32, 0.96], [0.23, 0.87]])
    edges = np.stack(
        [
            triangle[1] - triangle[0],
            triangle[2] - triangle[1],
            triangle[2] - triangle[0],
        ]
    )

    def project_min_max(points, axis):
        ps = points @ axis
        return ps.min(), ps.max()

    def refine_triangle2d(cell):
        # The underlying C functions are more optimized
        # but this is more explicit
        x0 = cell.origin
        xF = x0 + cell.h

        mins = triangle.min(axis=0)
        if np.any(mins > xF):
            return 0
        maxs = triangle.max(axis=0)
        if np.any(maxs < x0):
            return 0

        box_points = np.array(
            [
                [x0[0], x0[1]],
                [x0[0], xF[1]],
                [xF[0], x0[1]],
                [xF[0], xF[1]],
            ]
        )
        for i in range(3):
            axis = [-edges[i, 1], edges[i, 0]]
            bmin, bmax = project_min_max(box_points, axis)
            tmin, tmax = project_min_max(triangle, axis)
            if bmax < tmin or bmin > tmax:
                return 0
        return -1

    mesh1 = discretize.TreeMesh([64, 64])
    mesh1.refine(refine_triangle2d)

    mesh2 = discretize.TreeMesh([64, 64])
    mesh2.refine_triangle(triangle, -1)

    assert mesh1.equals(mesh2)


def test_triangle3d():
    # define a slower function that is surely accurate
    triangle = np.array([[0.14, 0.31, 0.23], [0.32, 0.96, 0.41], [0.23, 0.87, 0.72]])
    edges = np.stack(
        [
            triangle[1] - triangle[0],
            triangle[2] - triangle[1],
            triangle[2] - triangle[0],
        ]
    )
    triangle_norm = np.cross(edges[0], edges[1])
    triangle_proj = triangle[0] @ triangle_norm

    def project_min_max(points, axis):
        ps = points @ axis
        return ps.min(), ps.max()

    box_normals = np.eye(3)

    def refine_triangle(cell):
        # The underlying C functions are more optimized
        # but this is more explicit
        x0 = cell.origin
        xF = x0 + cell.h

        mins = triangle.min(axis=0)
        if np.any(mins > xF):
            return 0
        maxs = triangle.max(axis=0)
        if np.any(maxs < x0):
            return 0

        box_points = np.array(
            [
                [x0[0], x0[1], x0[2]],
                [x0[0], xF[1], x0[2]],
                [xF[0], x0[1], x0[2]],
                [xF[0], xF[1], x0[2]],
                [x0[0], x0[1], xF[2]],
                [x0[0], xF[1], xF[2]],
                [xF[0], x0[1], xF[2]],
                [xF[0], xF[1], xF[2]],
            ]
        )
        for i in range(3):
            for j in range(3):
                axis = np.cross(edges[i], box_normals[j])
                bmin, bmax = project_min_max(box_points, axis)
                tmin, tmax = project_min_max(triangle, axis)
                if bmax < tmin or bmin > tmax:
                    return 0
        bmin, bmax = project_min_max(box_points, triangle_norm)
        if bmax < triangle_proj or bmin > triangle_proj:
            return 0
        return -1

    mesh1 = discretize.TreeMesh([64, 64, 64])
    mesh1.refine(refine_triangle)

    mesh2 = discretize.TreeMesh([64, 64, 64])
    mesh2.refine_triangle(triangle, -1)

    assert mesh1.equals(mesh2)


def test_triangle_errors():
    not_triangles_array = np.random.rand(4, 2, 2)
    triangle3 = np.random.rand(3, 3)
    triangles2 = np.random.rand(10, 3, 2)
    levels = np.full(8, -1)

    mesh1 = discretize.TreeMesh([64, 64])

    # not passing 3 points on the second to last dimension to make a triangle
    with pytest.raises(ValueError):
        mesh1.refine_triangle(not_triangles_array, -1, finalize=False)

    # incorrect dimension
    with pytest.raises(ValueError):
        mesh1.refine_triangle(triangle3, -1, finalize=False)

    # bad number of levels and triangles
    with pytest.raises(ValueError):
        mesh1.refine_triangle(triangles2, levels, finalize=False)


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
