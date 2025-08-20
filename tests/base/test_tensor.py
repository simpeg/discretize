import pytest
import numpy as np
import numpy.testing as npt
import unittest
import discretize
from scipy.sparse.linalg import spsolve

TOL = 1e-10

gen = np.random.default_rng(123)


class BasicTensorMeshTests(unittest.TestCase):
    def setUp(self):
        a = np.array([1, 1, 1])
        b = np.array([1, 2])
        c = np.array([1, 4])
        self.mesh2 = discretize.TensorMesh([a, b], [3, 5])
        self.mesh3 = discretize.TensorMesh([a, b, c])

    def test_gridded_2D(self):
        H = self.mesh2.h_gridded
        test_hx = np.all(H[:, 0] == np.r_[1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        test_hy = np.all(H[:, 1] == np.r_[1.0, 1.0, 1.0, 2.0, 2.0, 2.0])
        self.assertTrue(test_hx and test_hy)

    def test_gridded_3D(self):
        H = self.mesh3.h_gridded
        test_hx = np.all(
            H[:, 0] == np.r_[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )
        test_hy = np.all(
            H[:, 1] == np.r_[1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
        )
        test_hz = np.all(
            H[:, 2] == np.r_[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
        )
        self.assertTrue(test_hx and test_hy and test_hz)

    def test_vectorN_2D(self):
        testNx = np.array([3, 4, 5, 6])
        testNy = np.array([5, 6, 8])
        xtest = np.all(self.mesh2.nodes_x == testNx)
        ytest = np.all(self.mesh2.nodes_y == testNy)
        self.assertTrue(xtest and ytest)

    def test_vectorCC_2D(self):
        testNx = np.array([3.5, 4.5, 5.5])
        testNy = np.array([5.5, 7])

        xtest = np.all(self.mesh2.cell_centers_x == testNx)
        ytest = np.all(self.mesh2.cell_centers_y == testNy)
        self.assertTrue(xtest and ytest)

    def test_area_3D(self):
        test_area = np.array(
            [
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                4,
                4,
                4,
                4,
                8,
                8,
                8,
                8,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                1,
                1,
                1,
                2,
                2,
                2,
                1,
                1,
                1,
                2,
                2,
                2,
                1,
                1,
                1,
                2,
                2,
                2,
            ]
        )
        t1 = np.all(self.mesh3.face_areas == test_area)
        self.assertTrue(t1)

    def test_vol_3D(self):
        test_vol = np.array([1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8])
        t1 = np.all(self.mesh3.cell_volumes == test_vol)
        self.assertTrue(t1)

    def test_vol_2D(self):
        test_vol = np.array([1, 1, 1, 2, 2, 2])
        t1 = np.all(self.mesh2.cell_volumes == test_vol)
        self.assertTrue(t1)

    def test_edge_3D(self):
        test_edge = np.array(
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
            ]
        )
        t1 = np.all(self.mesh3.edge_lengths == test_edge)
        self.assertTrue(t1)

    def test_edge_2D(self):
        test_edge = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2])
        t1 = np.all(self.mesh2.edge_lengths == test_edge)
        self.assertTrue(t1)

    def test_oneCell(self):
        hx = np.array([1e-5])
        M = discretize.TensorMesh([hx])
        self.assertTrue(M.nC == 1)

    def test_printing(self):
        print(discretize.TensorMesh([10]))
        print(discretize.TensorMesh([10, 10]))
        print(discretize.TensorMesh([10, 10, 10]))

    def test_centering(self):
        M1d = discretize.TensorMesh([10], x0="C")
        M2d = discretize.TensorMesh([10, 10], x0="CC")
        M3d = discretize.TensorMesh([10, 10, 10], x0="CCC")
        self.assertLess(np.abs(M1d.x0 + 0.5).sum(), TOL)
        self.assertLess(np.abs(M2d.x0 + 0.5).sum(), TOL)
        self.assertLess(np.abs(M3d.x0 + 0.5).sum(), TOL)

    def test_negative(self):
        M1d = discretize.TensorMesh([10], x0="N")
        self.assertRaises(Exception, discretize.TensorMesh, [10], "F")
        M2d = discretize.TensorMesh([10, 10], x0="NN")
        M3d = discretize.TensorMesh([10, 10, 10], x0="NNN")
        self.assertLess(np.abs(M1d.x0 + 1.0).sum(), TOL)
        self.assertLess(np.abs(M2d.x0 + 1.0).sum(), TOL)
        self.assertLess(np.abs(M3d.x0 + 1.0).sum(), TOL)

    def test_cent_neg(self):
        M3d = discretize.TensorMesh([10, 10, 10], x0="C0N")
        self.assertLess(np.abs(M3d.x0 + np.r_[0.5, 0, 1.0]).sum(), TOL)

    def test_tensor(self):
        M = discretize.TensorMesh([[(10.0, 2)]])
        self.assertLess(np.abs(M.h[0] - np.r_[10.0, 10.0]).sum(), TOL)

    def test_serialization(self):
        mesh = discretize.TensorMesh.deserialize(self.mesh2.serialize())
        self.assertTrue(np.all(self.mesh2.x0 == mesh.x0))
        self.assertTrue(np.all(self.mesh2.shape_cells == mesh.shape_cells))
        self.assertTrue(np.all(self.mesh2.h[0] == mesh.h[0]))
        self.assertTrue(np.all(self.mesh2.h[1] == mesh.h[1]))
        self.assertTrue(np.all(self.mesh2.gridCC == mesh.gridCC))


class TestTensorMeshProperties:
    """
    Test some of the properties in TensorMesh
    """

    @pytest.fixture(params=[1, 2, 3], ids=["dims-1", "dims-2", "dims-3"])
    def mesh(self, request):
        """Sample TensorMesh."""
        if request.param == 1:
            h = [10]
            origin = (-35.5,)
        elif request.param == 2:
            h = [10, 15]
            origin = (-35.5, 105.3)
        else:
            h = [10, 15, 20]
            origin = (-35.5, 105.3, -27.3)
        return discretize.TensorMesh(h, origin=origin)

    def test_cell_nodes(self, mesh):
        """Test TensorMesh.cell_nodes."""
        expected_cell_nodes = np.array([cell.nodes for cell in mesh])
        np.testing.assert_equal(mesh.cell_nodes, expected_cell_nodes)

    def test_cell_bounds(self, mesh):
        """Test TensorMesh.cell_bounds."""
        expected_cell_bounds = np.array([cell.bounds for cell in mesh])
        np.testing.assert_equal(mesh.cell_bounds, expected_cell_bounds)


class TestPoissonEqn(discretize.tests.OrderTest):
    name = "Poisson Equation"
    meshSizes = [10, 16, 20]

    def getError(self):
        # Create some functions to integrate
        fun = (
            lambda x: np.sin(2 * np.pi * x[:, 0])
            * np.sin(2 * np.pi * x[:, 1])
            * np.sin(2 * np.pi * x[:, 2])
        )
        sol = lambda x: -3.0 * ((2 * np.pi) ** 2) * fun(x)

        self.M.set_cell_gradient_BC("dirichlet")

        D = self.M.face_divergence
        G = self.M.cell_gradient
        if self.forward:
            sA = sol(self.M.gridCC)
            sN = D * G * fun(self.M.gridCC)
            err = np.linalg.norm((sA - sN), np.inf)
        else:
            fA = fun(self.M.gridCC)
            fN = spsolve(D * G, sol(self.M.gridCC))
            err = np.linalg.norm((fA - fN), np.inf)
        return err

    def test_orderForward(self):
        self.name = "Poisson Equation - Forward"
        self.forward = True
        self.orderTest()

    def test_orderBackward(self):
        self.name = "Poisson Equation - Backward"
        self.forward = False
        self.orderTest()


@pytest.fixture(params=[1, 2, 3], ids=["dims-1", "dims-2", "dims-3"])
def random_tensor_mesh(request):
    dim = request.param
    rng = np.random.default_rng(440122)
    shape = rng.integers(5, 10, dim)
    cell_widths = [rng.uniform(3.0, 872634.321, n) for n in shape]
    origin = rng.uniform(-101.031, 33.2, dim)

    return discretize.TensorMesh(cell_widths, origin)


def test_tensor_point2index_inside_points(random_tensor_mesh):
    mesh = random_tensor_mesh
    dim = mesh.dim
    m_origin = mesh.origin
    m_extent = np.atleast_1d(np.max(mesh.nodes, axis=0))

    nd = 15
    points = np.stack(np.meshgrid(*np.linspace(m_origin, m_extent, nd).T), axis=-1)
    points = points.reshape((-1, dim))

    npt.assert_array_equal(mesh.is_inside(points), True)

    cell_inds = mesh.point2index(points)
    for icell, p in zip(cell_inds, points):
        cell = mesh[icell]
        c_origin, c_extent = cell.bounds.reshape((dim, 2)).T
        dim_test = (p >= c_origin) & (p <= c_extent)
        npt.assert_equal(dim_test, True)


def test_tensor_point2index_outside_points(random_tensor_mesh):
    mesh = random_tensor_mesh
    dim = mesh.dim
    m_origin = mesh.origin
    m_extent = np.atleast_1d(np.max(mesh.nodes, axis=0))
    m_width = m_extent - m_origin

    nd = 15
    points = np.stack(
        np.meshgrid(*np.linspace(m_origin - m_width * 2, m_extent + m_width * 2, nd).T),
        axis=-1,
    )
    points = points.reshape((-1, dim))
    outside_points = points[~mesh.is_inside(points)]

    npt.assert_array_equal(mesh.is_inside(outside_points), False)

    # manually check each point that is outside
    cell_inds = mesh.point2index(outside_points)
    for icell, p in zip(cell_inds, outside_points):
        cell = mesh[icell]
        c_origin, c_extent = cell.bounds.reshape((dim, 2)).T
        dim_test = np.zeros(dim, bool)
        for i in range(dim):
            p_d = p[i]
            if p_d < m_origin[i]:
                dim_test[i] = p_d < c_origin[i]
            elif p_d > m_extent[i]:
                dim_test[i] = p_d > c_extent[i]
            else:
                dim_test[i] = p_d >= c_origin[i] and p_d <= c_extent[i]
        npt.assert_equal(dim_test, True)


if __name__ == "__main__":
    unittest.main()
