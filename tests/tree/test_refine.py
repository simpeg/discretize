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
    not_triangles_array = np.empty((4, 2, 2))
    triangle3 = np.empty((3, 3))
    triangles2 = np.empty((10, 3, 2))
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


def test_tetra2d():
    # It actually calls triangle refine... just double check that works
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
    mesh2.refine_tetrahedron(triangle, -1)

    assert mesh1.equals(mesh2)


def test_tetra3d():
    # define a slower function that is surely accurate
    simplex = np.array(
        [[0.32, 0.21, 0.15], [0.82, 0.19, 0.34], [0.14, 0.82, 0.29], [0.32, 0.27, 0.83]]
    )
    edges = np.stack(
        [
            simplex[1] - simplex[0],
            simplex[2] - simplex[0],
            simplex[2] - simplex[1],
            simplex[3] - simplex[0],
            simplex[3] - simplex[1],
            simplex[3] - simplex[2],
        ]
    )

    def project_min_max(points, axis):
        ps = points @ axis
        return ps.min(), ps.max()

    box_normals = np.eye(3)

    def refine_simplex(cell):
        x0 = cell.origin
        xF = x0 + cell.h
        simp = simplex

        # Bounding box tests
        # 3(x2) box face normals
        mins = simp.min(axis=0)
        if np.any(mins > xF):
            return 0
        maxs = simp.max(axis=0)
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
        # 3 box edges tangents and 6 simplex edge tangents
        for i in range(6):
            for j in range(3):
                axis = np.cross(edges[i], box_normals[j])
                bmin, bmax = project_min_max(box_points, axis)
                tmin, tmax = project_min_max(simp, axis)
                if bmax < tmin or bmin > tmax:
                    return 0

        # 4 simplex faces
        axis = np.cross(edges[0], edges[1])
        tmin, tmax = project_min_max(simp, axis)
        bmin, bmax = project_min_max(box_points, axis)
        if bmax < tmin or bmin > tmax:
            return 0
        axis = np.cross(edges[0], edges[3])
        tmin, tmax = project_min_max(simp, axis)
        bmin, bmax = project_min_max(box_points, axis)
        if bmax < tmin or bmin > tmax:
            return 0
        axis = np.cross(edges[1], edges[4])
        tmin, tmax = project_min_max(simp, axis)
        bmin, bmax = project_min_max(box_points, axis)
        if bmax < tmin or bmin > tmax:
            return 0
        axis = np.cross(edges[2], edges[5])
        tmin, tmax = project_min_max(simp, axis)
        bmin, bmax = project_min_max(box_points, axis)
        if bmax < tmin or bmin > tmax:
            return 0
        return -1

    mesh1 = discretize.TreeMesh([32, 32, 32])
    mesh1.refine(refine_simplex)

    mesh2 = discretize.TreeMesh([32, 32, 32])
    mesh2.refine_tetrahedron(simplex, -1)

    assert mesh1.equals(mesh2)


def test_tetra_errors():
    not_simplex_array = np.empty((4, 3, 3))
    simplex = np.empty((4, 2))
    simplices = np.empty((10, 4, 3))
    levels = np.full(8, -1)

    mesh1 = discretize.TreeMesh([32, 32, 32])

    # not passing 4 points on the second to last dimension to make a simplex
    with pytest.raises(ValueError):
        mesh1.refine_tetrahedron(not_simplex_array, -1, finalize=False)

    # incorrect dimension
    with pytest.raises(ValueError):
        mesh1.refine_tetrahedron(simplex, -1, finalize=False)

    # bad number of levels and triangles
    with pytest.raises(ValueError):
        mesh1.refine_tetrahedron(simplices, levels, finalize=False)


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
        mesh.refine_box(x0s2d, x1s2d, [-1, -1, -1], finalize=False)


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


def test_refine_triang_prism():
    xyz = np.array(
        [
            [0.41, 0.21, 0.11],
            [0.21, 0.61, 0.22],
            [0.71, 0.71, 0.31],
        ]
    )
    h = 0.48

    simps = np.array([[0, 1, 2]])

    n_ps = len(xyz)
    simps1 = np.c_[simps[:, 0], simps[:, 1], simps[:, 2], simps[:, 0]] + [0, 0, 0, n_ps]
    simps2 = np.c_[simps[:, 0], simps[:, 1], simps[:, 2], simps[:, 1]] + [
        n_ps,
        n_ps,
        n_ps,
        0,
    ]
    simps3 = np.c_[simps[:, 1], simps[:, 2], simps[:, 0], simps[:, 2]] + [
        0,
        0,
        n_ps,
        n_ps,
    ]
    simps = np.r_[simps1, simps2, simps3]

    points = np.r_[xyz, xyz + [0, 0, h]]
    mesh1 = discretize.TreeMesh([32, 32, 32])
    mesh1.refine_tetrahedron(points[simps], -1)

    mesh2 = discretize.TreeMesh([32, 32, 32])
    mesh2.refine_vertical_trianglular_prism(xyz, h, -1)

    assert mesh1.equals(mesh2)


def test_refine_triang_prism_errors():
    xyz = np.array(
        [
            [0.41, 0.21, 0.11],
            [0.21, 0.61, 0.22],
            [0.71, 0.71, 0.31],
        ]
    )
    h = 0.48

    mesh = discretize.TreeMesh([32, 32])
    # Not implemented for 2D
    with pytest.raises(NotImplementedError):
        mesh.refine_vertical_trianglular_prism(xyz, h, -1)

    mesh = discretize.TreeMesh([32, 32, 32])
    # incorrect triangle dimensions
    with pytest.raises(ValueError):
        mesh.refine_vertical_trianglular_prism(xyz[:, :-1], h, -1)

    # incorrect levels and triangles
    ps = np.empty((10, 3, 3))
    with pytest.raises(ValueError):
        mesh.refine_vertical_trianglular_prism(ps, h, [-1, -2])

    # incorrect heights and triangles
    ps = np.empty((10, 3, 3))
    with pytest.raises(ValueError):
        mesh.refine_vertical_trianglular_prism(ps, [h, h], -1)

    # negative heights
    ps = np.empty((10, 3, 3))
    with pytest.raises(ValueError):
        mesh.refine_vertical_trianglular_prism(ps, -h, -1)


def test_bounding_box():
    # No padding
    rng = np.random.default_rng(51623978)
    xyz = rng.random((20, 2)) * 0.25 + 3 / 8
    mesh1 = discretize.TreeMesh([32, 32])
    mesh1.refine_bounding_box(xyz, -1, None)

    x0 = xyz.min(axis=0)
    xF = xyz.max(axis=0)

    mesh2 = discretize.TreeMesh([32, 32])
    mesh2.refine_box(x0, xF, -1)

    assert mesh1.equals(mesh2)

    # Padding at all levels
    n_cell_pad = 2
    x0 = xyz.min(axis=0)
    xF = xyz.max(axis=0)

    mesh1 = discretize.TreeMesh([32, 32])
    mesh1.refine_bounding_box(xyz, -1, n_cell_pad)

    mesh2 = discretize.TreeMesh([32, 32])
    for lv in range(mesh2.max_level, 1, -1):
        padding = n_cell_pad * (2 ** (mesh2.max_level - lv) / 32)
        x0 -= padding
        xF += padding
        mesh2.refine_box(x0, xF, lv, finalize=False)
    mesh2.finalize()

    assert mesh1.equals(mesh2)


def test_bounding_box_errors():
    mesh1 = discretize.TreeMesh([32, 32])

    xyz = np.empty((20, 3))
    # incorrect padding shape
    with pytest.raises(ValueError):
        mesh1.refine_bounding_box(xyz, -1, [[2, 3, 4]])

    # bad level
    with pytest.raises(IndexError):
        mesh1.refine_bounding_box(xyz, 20)


def test_refine_points():
    mesh1 = discretize.TreeMesh([32, 32])
    point = [0.5, 0.5]
    level = -1
    n_cell_pad = 3
    mesh1.refine_points(point, level, None)

    mesh2 = discretize.TreeMesh([32, 32])
    mesh2.insert_cells(point, -1)

    assert mesh1.equals(mesh2)

    mesh1 = discretize.TreeMesh([32, 32])
    mesh1.refine_points(point, level, n_cell_pad)

    mesh2 = discretize.TreeMesh([32, 32])
    ball_rad = 0.0
    for lv in range(mesh2.max_level, 1, -1):
        ball_rad += 2 ** (mesh2.max_level - lv) / 32 * n_cell_pad
        mesh2.refine_ball(point, ball_rad, lv, finalize=False)
    mesh2.finalize()

    assert mesh1.equals(mesh2)


def test_refine_points_errors():
    mesh1 = discretize.TreeMesh([32, 32])

    point = [0.5, 0.5]

    with pytest.raises(IndexError):
        mesh1.refine_points(point, 20)


def test_refine_surface2D():
    mesh1 = discretize.TreeMesh([32, 32])
    points = [[0.3, 0.3], [0.7, 0.3]]
    mesh1.refine_surface(points, -1, None, pad_up=True, pad_down=True)

    mesh2 = discretize.TreeMesh([32, 32])
    x0 = [0.3, 0.3]
    xF = [0.7, 0.3]
    mesh2.refine_box(x0, xF, -1)

    assert mesh1.equals(mesh2)

    mesh1 = discretize.TreeMesh([32, 32])
    points = [[0.3, 0.3], [0.7, 0.3]]
    n_cell_pad = 2
    mesh1.refine_surface(points, -1, n_cell_pad, pad_up=True, pad_down=True)

    mesh2 = discretize.TreeMesh([32, 32])
    x0 = np.r_[0.3, 0.3]
    xF = np.r_[0.7, 0.3]
    for lv in range(mesh2.max_level, 1, -1):
        pad = 2 ** (mesh2.max_level - lv) / 32 * n_cell_pad
        x0 -= pad
        xF += pad
        mesh2.refine_box(x0, xF, lv, finalize=False)
    mesh2.finalize()

    assert mesh1.equals(mesh2)


def test_refine_surface3D():
    mesh1 = discretize.TreeMesh([32, 32, 32])
    points = [
        [0.3, 0.3, 0.5],
        [0.7, 0.3, 0.5],
        [0.3, 0.7, 0.5],
        [0.7, 0.7, 0.5],
    ]
    mesh1.refine_surface(points, -1, [[1, 2, 3]], pad_up=True, pad_down=True)

    mesh2 = discretize.TreeMesh([32, 32, 32])
    pad = np.array([1, 2, 3]) / 32
    x0 = [0.3, 0.3, 0.5] - pad
    xF = [0.7, 0.7, 0.5] + pad
    mesh2.refine_box(x0, xF, -1)

    assert mesh1.equals(mesh2)

    mesh3 = discretize.TreeMesh([32, 32, 32])
    simps = [[0, 1, 2], [1, 2, 3]]
    mesh3.refine_surface((points, simps), -1, [[1, 2, 3]], pad_up=True, pad_down=True)

    assert mesh1.equals(mesh3)


def test_refine_surface_errors():
    mesh = discretize.TreeMesh([32, 32])
    points = [[0.3, 0.3], [0.7, 0.3]]

    with pytest.raises(ValueError):
        mesh.refine_surface(points, -1, [[0, 1, 2, 3]])

    with pytest.raises(IndexError):
        mesh.refine_surface(points, 20)
