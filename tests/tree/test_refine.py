import discretize
import numpy as np
import pytest
from discretize.tests import assert_cell_intersects_geometric


def test_2d_line():
    segments = np.array([[0.12, 0.33], [0.32, 0.93]])

    mesh1 = discretize.TreeMesh([64, 64])
    mesh1.refine_line(segments, -1)

    def refine_line(cell):
        return assert_cell_intersects_geometric(
            cell, segments, edges=[0, 1], as_refine=True
        )

    mesh2 = discretize.TreeMesh([64, 64])
    mesh2.refine(refine_line)

    assert mesh2.equals(mesh1)


def test_3d_line():
    segments = np.array([[0.12, 0.33, 0.19], [0.32, 0.93, 0.68]])

    mesh1 = discretize.TreeMesh([64, 64, 64])
    mesh1.refine_line(segments, -1)

    def refine_line(cell):
        return assert_cell_intersects_geometric(
            cell, segments, edges=[0, 1], as_refine=True
        )

    mesh2 = discretize.TreeMesh([64, 64, 64])
    mesh2.refine(refine_line)

    assert mesh2.equals(mesh1)


def test_line_errors():
    mesh = discretize.TreeMesh([64, 64])
    rng = np.random.default_rng(512)
    segments2D = rng.random((5, 2))
    segments3D = rng.random((5, 3))

    # incorrect dimension
    with pytest.raises(ValueError):
        mesh.refine_line(segments3D, mesh.max_level, finalize=False)

    # incorrect number of levels
    # 4 segments won't broadcast to 2 levels
    with pytest.raises(ValueError):
        mesh.refine_line(segments2D, [mesh.max_level, 3], finalize=False)


def test_triangle2d():
    triangle = np.array([[0.14, 0.31], [0.32, 0.96], [0.23, 0.87]])
    edges = [[0, 1], [0, 2], [1, 2]]

    def refine_triangle(cell):
        return assert_cell_intersects_geometric(
            cell, triangle, edges=edges, as_refine=True
        )

    mesh1 = discretize.TreeMesh([64, 64])
    mesh1.refine(refine_triangle)

    mesh2 = discretize.TreeMesh([64, 64])
    mesh2.refine_triangle(triangle, -1)

    assert mesh1.equals(mesh2)


def test_triangle3d():
    triangle = np.array([[0.14, 0.31, 0.23], [0.32, 0.96, 0.41], [0.23, 0.87, 0.72]])
    edges = [[0, 1], [0, 2], [1, 2]]
    faces = [0, 1, 2]

    def refine_triangle(cell):
        return assert_cell_intersects_geometric(
            cell, triangle, edges=edges, faces=faces, as_refine=True
        )

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


def test_tetra2d():
    # It actually calls triangle refine... just double check that works
    # define a slower function that is surely accurate
    triangle = np.array([[0.14, 0.31], [0.32, 0.96], [0.23, 0.87]])
    edges = [[0, 1], [0, 2], [1, 2]]

    def refine_triangle(cell):
        return assert_cell_intersects_geometric(
            cell, triangle, edges=edges, as_refine=True
        )

    mesh1 = discretize.TreeMesh([64, 64])
    mesh1.refine(refine_triangle)

    mesh2 = discretize.TreeMesh([64, 64])
    mesh2.refine_tetrahedron(triangle, -1)

    assert mesh1.equals(mesh2)


def test_tetra3d():
    simplex = np.array(
        [[0.32, 0.21, 0.15], [0.82, 0.19, 0.34], [0.14, 0.82, 0.29], [0.32, 0.27, 0.83]]
    )
    edges = [[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 3]]
    faces = [
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
    ]

    def refine_simplex(cell):
        return assert_cell_intersects_geometric(
            cell, simplex, edges=edges, faces=faces, as_refine=True
        )

    mesh1 = discretize.TreeMesh([32, 32, 32])
    mesh1.refine(refine_simplex)

    mesh2 = discretize.TreeMesh([32, 32, 32])
    mesh2.refine_tetrahedron(simplex, -1)

    assert mesh1.equals(mesh2)


def test_tetra_errors():
    not_simplex_array = np.random.rand(4, 3, 3)
    simplex = np.random.rand(4, 2)
    simplices = np.random.rand(10, 4, 3)
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
    rng = np.random.default_rng(32)
    x0s = rng.random((3, 2))
    x0s2d = 0.5 * rng.random((2, 2))
    x1s2d = 0.5 * rng.random((2, 2)) + 0.5

    x0s3d = 0.5 * rng.random((2, 3))
    x1s3d = 0.5 * rng.random((2, 3)) + 0.5

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

    all_points = np.concatenate([xyz, xyz + [0, 0, h]])
    # only need to define the unique edge tangents (minus axis-aligned ones)
    edges = [
        [0, 1],
        [0, 2],
        [1, 2],
    ]

    # and define unique face normals (absent any face parallel to an axis,
    # or with normal defined by an axis and an edge above.)
    faces = [
        [0, 1, 2],
    ]

    def refine_vert(cell):
        return assert_cell_intersects_geometric(
            cell, all_points, edges=edges, faces=faces, as_refine=True
        )

    mesh1 = discretize.TreeMesh([32, 32, 32])
    mesh1.refine(refine_vert)

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
    ps = np.random.rand(10, 3, 3)
    with pytest.raises(ValueError):
        mesh.refine_vertical_trianglular_prism(ps, h, [-1, -2])

    # incorrect heights and triangles
    ps = np.random.rand(10, 3, 3)
    with pytest.raises(ValueError):
        mesh.refine_vertical_trianglular_prism(ps, [h, h], -1)

    # negative heights
    ps = np.random.rand(10, 3, 3)
    with pytest.raises(ValueError):
        mesh.refine_vertical_trianglular_prism(ps, -h, -1)


def test_bounding_box():
    # No padding
    xyz = np.random.rand(20, 2) * 0.25 + 3 / 8
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

    xyz = np.random.rand(20, 3)
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


def test_refine_plane2D():
    p0 = [2, 2]
    normal = [-1, 1]
    p1 = [-2, -2]

    mesh1 = discretize.TreeMesh([64, 64])
    mesh1.refine_plane(p0, normal, -1)

    mesh2 = discretize.TreeMesh([64, 64])
    mesh2.refine_line(np.stack([p0, p1]), -1)

    assert mesh1.equals(mesh2)


def test_refine_plane3D():
    p0 = [20, 20, 20]
    normal = [-1, -1, 2]
    # define 4 corner points (including p0) of a plane to create triangles
    # to verify the refine functionallity
    p1 = [20, -20, 0]
    p2 = [-20, 20, 0]
    p3 = [-20, -20, -20]
    tris = np.stack([[p0, p1, p2], [p1, p2, p3]])

    mesh1 = discretize.TreeMesh([64, 64, 64])
    mesh1.refine_plane(p0, normal, -1)

    mesh2 = discretize.TreeMesh([64, 64, 64])
    mesh2.refine_triangle(tris, -1)

    assert mesh1.equals(mesh2)
