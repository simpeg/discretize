import discretize
import numpy as np
import pytest
from discretize.tests import assert_cell_intersects_geometric

pytestmark = pytest.mark.parametrize("dim", [2, 3])


def test_point_locator(dim):
    point = [0.44] * dim
    mesh = discretize.TreeMesh([32] * dim)

    mesh.insert_cells(point, -1)

    ind = mesh.get_containing_cells(point)
    cell = mesh[ind]
    assert_cell_intersects_geometric(cell, point)


def test_ball_locator(dim):
    center = [0.44] * dim
    radius = 0.12

    mesh = discretize.TreeMesh([32] * dim)
    mesh.refine_ball(center, radius, -1)

    cells = mesh.get_cells_within_ball(center, radius)

    r2 = radius * radius

    def ball_intersects(cell):
        a = cell.origin
        b = a + cell.h
        dr = np.maximum(a, np.minimum(center, b)) - center
        r2_test = np.sum(dr * dr)
        return r2_test < r2

    # ensure it found all of the cells by using a brute force search
    test = []
    for cell in mesh:
        if ball_intersects(cell):
            test.append(cell.index)
    np.testing.assert_array_equal(test, cells)


def test_line_locator(dim):
    segment = np.array([[0.12, 0.33, 0.19], [0.32, 0.93, 0.68]])[:, :dim]

    mesh = discretize.TreeMesh([32] * dim)
    mesh.refine_line(segment, -1)

    with pytest.raises(ValueError):
        mesh.get_cells_along_line(segment[:-1])

    cell_inds = mesh.get_cells_along_line(segment)

    # ensure it found all of the cells by using a brute force search
    test = []
    for cell in mesh:
        try:
            assert_cell_intersects_geometric(cell, segment, edges=[0, 1])
            test.append(cell.index)
        except AssertionError:
            pass
    np.testing.assert_array_equal(test, cell_inds)


def test_box_locator(dim):
    xmin = [0.2] * dim
    xmax = [0.4] * dim
    points = np.stack([xmin, xmax])

    mesh = discretize.TreeMesh([32] * dim)
    mesh.refine_box(xmin, xmax, -1)

    cell_inds = mesh.get_cells_within_aabb(xmin, xmax)

    # ensure it found all of the cells by using a brute force search
    test = []
    for cell in mesh:
        try:
            assert_cell_intersects_geometric(cell, points)
            test.append(cell.index)
        except AssertionError:
            pass
    np.testing.assert_array_equal(test, cell_inds)


def test_plane_locator(dim):
    if dim == 2:
        p0 = [2, 2]
        normal = [-1, 1]
        p1 = [-2, -2]
        points = np.stack([p0, p1])
        edges = [0, 1]
        faces = None
    elif dim == 3:
        p0 = [20, 20, 20]
        normal = [-1, -1, 2]
        # define 4 corner points (including p0) of a plane to create triangles
        # to verify the refine functionallity
        p1 = [20, -20, 0]
        p2 = [-20, 20, 0]
        p3 = [-20, -20, -20]
        points = np.stack([p0, p1, p2, p3])
        edges = [[0, 1], [0, 2]]
        faces = [[0, 1, 2]]

    mesh = discretize.TreeMesh([16] * dim)
    mesh.refine_plane(p0, normal, -1)

    cell_inds = mesh.get_cells_along_plane(p0, normal)

    # ensure it found all of the cells by using a brute force search
    test = []
    for cell in mesh:
        try:
            assert_cell_intersects_geometric(cell, points, edges=edges, faces=faces)
            test.append(cell.index)
        except AssertionError:
            pass
    np.testing.assert_array_equal(test, cell_inds)


def test_triangle_locator(dim):
    triangle = np.array([[0.14, 0.31, 0.23], [0.32, 0.96, 0.41], [0.23, 0.87, 0.72]])[
        :, :dim
    ]
    edges = [[0, 1], [0, 2], [1, 2]]
    faces = [0, 1, 2]

    mesh = discretize.TreeMesh([32] * dim)
    mesh.refine_triangle(triangle, -1)

    cell_inds = mesh.get_cells_within_triangle(triangle)

    # test it throws an error without giving enough points to triangle.
    with pytest.raises(ValueError):
        mesh.get_cells_within_triangle(triangle[:-1])

    # ensure it found all of the cells by using a brute force search
    test = []
    for cell in mesh:
        try:
            assert_cell_intersects_geometric(cell, triangle, edges=edges, faces=faces)
            test.append(cell.index)
        except AssertionError:
            pass
    np.testing.assert_array_equal(test, cell_inds)


def test_vert_tri_prism_locator(dim):
    xyz = np.array(
        [
            [0.41, 0.21, 0.11],
            [0.21, 0.61, 0.22],
            [0.71, 0.71, 0.31],
        ]
    )
    h = 0.48

    points = np.concatenate([xyz, xyz + [0, 0, h]])
    # only need to define the unique edge tangents (minus axis-aligned ones)
    edges = [
        [0, 1],
        [0, 2],
        [1, 2],
    ]
    faces = [
        [0, 1, 2],
    ]

    mesh = discretize.TreeMesh([16] * dim)
    if dim == 2:
        with pytest.raises(NotImplementedError):
            mesh.refine_vertical_trianglular_prism(xyz, h, -1)
    else:
        mesh.refine_vertical_trianglular_prism(xyz, h, -1)

        # test it throws an error on incorrect number of points for the triangle
        with pytest.raises(ValueError):
            mesh.get_cells_within_vertical_trianglular_prism(xyz[:-1], h)

        cell_inds = mesh.get_cells_within_vertical_trianglular_prism(xyz, h)

        # ensure it found all of the cells by using a brute force search
        test = []
        for cell in mesh:
            try:
                assert_cell_intersects_geometric(cell, points, edges=edges, faces=faces)
                test.append(cell.index)
            except AssertionError:
                pass
        np.testing.assert_array_equal(test, cell_inds)


def test_tetrahedron_locator(dim):
    simplex = np.array(
        [[0.32, 0.21, 0.15], [0.82, 0.19, 0.34], [0.14, 0.82, 0.29], [0.32, 0.27, 0.83]]
    )[: dim + 1, :dim]
    edges = [[0, 1], [0, 2], [1, 2], [0, 3], [1, 3], [2, 3]][: (dim - 1) * 3]
    faces = [
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
    ][: 3 * dim - 5]

    mesh = discretize.TreeMesh([16] * dim)
    mesh.refine_tetrahedron(simplex, -1)

    cell_inds = mesh.get_cells_within_tetrahedron(simplex)

    # test it throws an error without giving enough points to triangle.
    with pytest.raises(ValueError):
        mesh.get_cells_within_tetrahedron(simplex[:-1])

    # ensure it found all of the cells by using a brute force search
    test = []
    for cell in mesh:
        try:
            assert_cell_intersects_geometric(cell, simplex, edges=edges, faces=faces)
            test.append(cell.index)
        except AssertionError:
            pass
    np.testing.assert_array_equal(test, cell_inds)
