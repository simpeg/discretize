import unittest
import numpy as np
import scipy.sparse as sp
from discretize.utils import (
    sdiag,
    sub2ind,
    ndgrid,
    mkvc,
    is_scalar,
    inverse_2x2_block_diagonal,
    inverse_3x3_block_diagonal,
    inverse_property_tensor,
    make_property_tensor,
    index_cube,
    ind2sub,
    as_array_n_by_dim,
    TensorType,
    Zero,
    Identity,
    extract_core_mesh,
    active_from_xyz,
    mesh_builder_xyz,
    refine_tree_xyz,
    unpack_widths,
)
import discretize

TOL = 1e-8


class TestSequenceFunctions(unittest.TestCase):
    def setUp(self):
        self.a = np.array([1, 2, 3])
        self.b = np.array([1, 2])
        self.c = np.array([1, 2, 3, 4])

    def test_mkvc1(self):
        x = mkvc(self.a)
        self.assertTrue(x.shape, (3,))

    def test_mkvc2(self):
        x = mkvc(self.a, 2)
        self.assertTrue(x.shape, (3, 1))

    def test_mkvc3(self):
        x = mkvc(self.a, 3)
        self.assertTrue(x.shape, (3, 1, 1))

    def test_ndgrid_2D(self):
        XY = ndgrid([self.a, self.b])

        X1_test = np.array([1, 2, 3, 1, 2, 3])
        X2_test = np.array([1, 1, 1, 2, 2, 2])

        self.assertTrue(np.all(XY[:, 0] == X1_test))
        self.assertTrue(np.all(XY[:, 1] == X2_test))

    def test_ndgrid_3D(self):
        XYZ = ndgrid([self.a, self.b, self.c])

        X1_test = np.array(
            [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]
        )
        X2_test = np.array(
            [1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2]
        )
        X3_test = np.array(
            [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4]
        )

        self.assertTrue(np.all(XYZ[:, 0] == X1_test))
        self.assertTrue(np.all(XYZ[:, 1] == X2_test))
        self.assertTrue(np.all(XYZ[:, 2] == X3_test))

    def test_sub2ind(self):
        x = np.ones((5, 2))
        self.assertTrue(np.all(sub2ind(x.shape, [0, 0]) == [0]))
        self.assertTrue(np.all(sub2ind(x.shape, [4, 0]) == [4]))
        self.assertTrue(np.all(sub2ind(x.shape, [0, 1]) == [5]))
        self.assertTrue(np.all(sub2ind(x.shape, [4, 1]) == [9]))
        self.assertTrue(np.all(sub2ind(x.shape, [[4, 1]]) == [9]))
        self.assertTrue(
            np.all(sub2ind(x.shape, [[0, 0], [4, 0], [0, 1], [4, 1]]) == [0, 4, 5, 9])
        )

    def test_ind2sub(self):
        x = np.ones((5, 2))
        self.assertTrue(np.all(ind2sub(x.shape, [0, 4, 5, 9])[0] == [0, 4, 0, 4]))
        self.assertTrue(np.all(ind2sub(x.shape, [0, 4, 5, 9])[1] == [0, 0, 1, 1]))

    def test_index_cube_2D(self):
        nN = np.array([3, 3])
        self.assertTrue(np.all(index_cube("A", nN) == np.array([0, 1, 3, 4])))
        self.assertTrue(np.all(index_cube("B", nN) == np.array([3, 4, 6, 7])))
        self.assertTrue(np.all(index_cube("C", nN) == np.array([4, 5, 7, 8])))
        self.assertTrue(np.all(index_cube("D", nN) == np.array([1, 2, 4, 5])))

    def test_index_cube_3D(self):
        nN = np.array([3, 3, 3])
        self.assertTrue(
            np.all(index_cube("A", nN) == np.array([0, 1, 3, 4, 9, 10, 12, 13]))
        )
        self.assertTrue(
            np.all(index_cube("B", nN) == np.array([3, 4, 6, 7, 12, 13, 15, 16]))
        )
        self.assertTrue(
            np.all(index_cube("C", nN) == np.array([4, 5, 7, 8, 13, 14, 16, 17]))
        )
        self.assertTrue(
            np.all(index_cube("D", nN) == np.array([1, 2, 4, 5, 10, 11, 13, 14]))
        )
        self.assertTrue(
            np.all(index_cube("E", nN) == np.array([9, 10, 12, 13, 18, 19, 21, 22]))
        )
        self.assertTrue(
            np.all(index_cube("F", nN) == np.array([12, 13, 15, 16, 21, 22, 24, 25]))
        )
        self.assertTrue(
            np.all(index_cube("G", nN) == np.array([13, 14, 16, 17, 22, 23, 25, 26]))
        )
        self.assertTrue(
            np.all(index_cube("H", nN) == np.array([10, 11, 13, 14, 19, 20, 22, 23]))
        )

    def test_invXXXBlockDiagonal(self):
        rng = np.random.default_rng(78352)
        a = [rng.random((5, 1)) for i in range(4)]

        B = inverse_2x2_block_diagonal(*a)

        A = sp.vstack(
            (
                sp.hstack((sdiag(a[0]), sdiag(a[1]))),
                sp.hstack((sdiag(a[2]), sdiag(a[3]))),
            )
        )

        Z2 = B * A - sp.identity(10)
        self.assertTrue(np.linalg.norm(Z2.todense().ravel(), 2) < TOL)

        a = [rng.random((5, 1)) for i in range(9)]
        B = inverse_3x3_block_diagonal(*a)

        A = sp.vstack(
            (
                sp.hstack((sdiag(a[0]), sdiag(a[1]), sdiag(a[2]))),
                sp.hstack((sdiag(a[3]), sdiag(a[4]), sdiag(a[5]))),
                sp.hstack((sdiag(a[6]), sdiag(a[7]), sdiag(a[8]))),
            )
        )

        Z3 = B * A - sp.identity(15)

        self.assertTrue(np.linalg.norm(Z3.todense().ravel(), 2) < TOL)

    def test_inverse_property_tensor2D(self):
        rng = np.random.default_rng(763)
        M = discretize.TensorMesh([6, 6])
        a1 = rng.random(M.nC)
        a2 = rng.random(M.nC)
        a3 = rng.random(M.nC)
        prop1 = a1
        prop2 = np.c_[a1, a2]
        prop3 = np.c_[a1, a2, a3]

        for prop in [4, prop1, prop2, prop3]:
            b = inverse_property_tensor(M, prop)
            A = make_property_tensor(M, prop)
            B1 = make_property_tensor(M, b)
            B2 = inverse_property_tensor(M, prop, return_matrix=True)

            Z = B1 * A - sp.identity(M.nC * 2)
            self.assertTrue(np.linalg.norm(Z.todense().ravel(), 2) < TOL)
            Z = B2 * A - sp.identity(M.nC * 2)
            self.assertTrue(np.linalg.norm(Z.todense().ravel(), 2) < TOL)

    def test_TensorType2D(self):
        rng = np.random.default_rng(8546)
        M = discretize.TensorMesh([6, 6])
        a1 = rng.random(M.nC)
        a2 = rng.random(M.nC)
        a3 = rng.random(M.nC)
        prop1 = a1
        prop2 = np.c_[a1, a2]
        prop3 = np.c_[a1, a2, a3]

        for ii, prop in enumerate([4, prop1, prop2, prop3]):
            self.assertTrue(TensorType(M, prop) == ii)

        self.assertRaises(Exception, TensorType, M, np.c_[a1, a2, a3, a3])
        self.assertTrue(TensorType(M, None) == -1)

    def test_TensorType3D(self):
        rng = np.random.default_rng(78352)
        M = discretize.TensorMesh([6, 6, 7])
        a1 = rng.random(M.nC)
        a2 = rng.random(M.nC)
        a3 = rng.random(M.nC)
        a4 = rng.random(M.nC)
        a5 = rng.random(M.nC)
        a6 = rng.random(M.nC)
        prop1 = a1
        prop2 = np.c_[a1, a2, a3]
        prop3 = np.c_[a1, a2, a3, a4, a5, a6]

        for ii, prop in enumerate([4, prop1, prop2, prop3]):
            self.assertTrue(TensorType(M, prop) == ii)

        self.assertRaises(Exception, TensorType, M, np.c_[a1, a2, a3, a3])
        self.assertTrue(TensorType(M, None) == -1)

    def test_inverse_property_tensor3D(self):
        rng = np.random.default_rng(78352)
        M = discretize.TensorMesh([6, 6, 6])
        a1 = rng.random(M.nC)
        a2 = rng.random(M.nC)
        a3 = rng.random(M.nC)
        a4 = rng.random(M.nC)
        a5 = rng.random(M.nC)
        a6 = rng.random(M.nC)
        prop1 = a1
        prop2 = np.c_[a1, a2, a3]
        prop3 = np.c_[a1, a2, a3, a4, a5, a6]

        for prop in [4, prop1, prop2, prop3]:
            b = inverse_property_tensor(M, prop)
            A = make_property_tensor(M, prop)
            B1 = make_property_tensor(M, b)
            B2 = inverse_property_tensor(M, prop, return_matrix=True)

            Z = B1 * A - sp.identity(M.nC * 3)
            self.assertTrue(np.linalg.norm(Z.todense().ravel(), 2) < TOL)
            Z = B2 * A - sp.identity(M.nC * 3)
            self.assertTrue(np.linalg.norm(Z.todense().ravel(), 2) < TOL)

    def test_is_scalar(self):
        self.assertTrue(is_scalar(1.0))
        self.assertTrue(is_scalar(1))
        self.assertTrue(is_scalar(1j))
        self.assertTrue(is_scalar(np.r_[1.0]))
        self.assertTrue(is_scalar(np.r_[1]))
        self.assertTrue(is_scalar(np.r_[1j]))

    def test_as_array_n_by_dim(self):
        true = np.array([[1, 2, 3]])

        listArray = as_array_n_by_dim([1, 2, 3], 3)
        self.assertTrue(np.all(true == listArray))
        self.assertTrue(true.shape == listArray.shape)

        listArray = as_array_n_by_dim(np.r_[1, 2, 3], 3)
        self.assertTrue(np.all(true == listArray))
        self.assertTrue(true.shape == listArray.shape)

        listArray = as_array_n_by_dim(np.array([[1, 2, 3.0]]), 3)
        self.assertTrue(np.all(true == listArray))
        self.assertTrue(true.shape == listArray.shape)

        true = np.array([[1, 2], [4, 5]])

        listArray = as_array_n_by_dim([[1, 2], [4, 5]], 2)
        self.assertTrue(np.all(true == listArray))
        self.assertTrue(true.shape == listArray.shape)


class TestZero(unittest.TestCase):
    def test_zero(self):
        z = Zero()
        assert z == 0
        assert not (z < 0)
        assert z <= 0
        assert not (z > 0)
        assert z >= 0
        assert +z == z
        assert -z == z
        assert z + 1 == 1
        assert z + 3 + z == 3
        assert z - 3 == -3
        assert z - 3 - z == -3
        assert 3 * z == 0
        assert z * 3 == 0
        assert z / 3 == 0

        a = 1
        a += z
        assert a == 1
        a = 1
        a += z
        assert a == 1
        self.assertRaises(ZeroDivisionError, lambda: 3 / z)

        assert mkvc(z) == 0
        assert sdiag(z) * a == 0
        assert z.T == 0
        assert z.transpose() == 0

    def test_mat_zero(self):
        z = Zero()
        S = sdiag(np.r_[2, 3])
        assert S * z == 0

    def test_numpy_multiply(self):
        z = Zero()
        x = np.r_[1, 2, 3]
        a = x * z
        assert isinstance(a, Zero)

        z = Zero()
        x = np.r_[1, 2, 3]
        a = z * x
        assert isinstance(a, Zero)

    def test_one(self):
        o = Identity()
        assert o == 1
        assert not (o < 1)
        assert o <= 1
        assert not (o > 1)
        assert o >= 1
        o = -o
        assert o == -1
        assert not (o < -1)
        assert o <= -1
        assert not (o > -1)
        assert o >= -1
        assert -1.0 * (-o) * o == -o
        o = Identity()
        assert +o == o
        assert -o == -o
        assert o * 3 == 3
        assert -o * 3 == -3
        assert -o * o == -1
        assert -o * o * -o == 1
        assert -o + 3 == 2
        assert 3 + -o == 2

        assert -o - 3 == -4
        assert o - 3 == -2
        assert 3 - -o == 4
        assert 3 - o == 2

        assert o // 2 == 0
        assert o / 2.0 == 0.5
        assert -o // 2 == -1
        assert -o / 2.0 == -0.5
        assert 2 / o == 2
        assert 2 // -o == -2
        assert 2.3 // o == 2
        assert 2.3 // -o == -3

        assert o.T == 1
        assert o.transpose() == 1

    def test_mat_one(self):
        o = Identity()
        S = sdiag(np.r_[2, 3])

        def check(exp, ans):
            assert np.all((exp).todense() == ans)

        check(S * o, [[2, 0], [0, 3]])
        check(o * S, [[2, 0], [0, 3]])
        check(S * -o, [[-2, 0], [0, -3]])
        check(-o * S, [[-2, 0], [0, -3]])
        check(S / o, [[2, 0], [0, 3]])
        check(S / -o, [[-2, 0], [0, -3]])
        self.assertRaises(NotImplementedError, lambda: o / S)

        check(S + o, [[3, 0], [0, 4]])
        check(o + S, [[3, 0], [0, 4]])
        check(S - o, [[1, 0], [0, 2]])

        check(S + -o, [[1, 0], [0, 2]])
        check(-o + S, [[1, 0], [0, 2]])

    def test_mat_shape(self):
        o = Identity()
        S = sdiag(np.r_[2, 3])[:1, :]
        self.assertRaises(ValueError, lambda: S + o)

        def check(exp, ans):
            assert np.all((exp).todense() == ans)

        check(S * o, [[2, 0]])
        check(S * -o, [[-2, 0]])

    def test_numpy_one(self):
        o = Identity()
        n = np.r_[2.0, 3]

        assert np.all(n + 1 == n + o)
        assert np.all(1 + n == o + n)
        assert np.all(n - 1 == n - o)
        assert np.all(1 - n == o - n)
        assert np.all(n / 1 == n / o)
        assert np.all(n / -1 == n / -o)
        assert np.all(1 / n == o / n)
        assert np.all(-1 / n == -o / n)
        assert np.all(n * 1 == n * o)
        assert np.all(n * -1 == n * -o)
        assert np.all(1 * n == o * n)
        assert np.all(-1 * n == -o * n)

    def test_both(self):
        z = Zero()
        o = Identity()
        assert o * z == 0
        assert o * z + o == 1
        assert o - z == 1


class TestMeshUtils(unittest.TestCase):
    def test_extract_core_mesh(self):
        # 1D Test on TensorMesh
        meshtest1d = discretize.TensorMesh([[(50.0, 10)]])
        xzlim1d = np.r_[[[0.0, 250.0]]]
        actind1d, meshCore1d = extract_core_mesh(xzlim1d, meshtest1d)

        self.assertEqual(len(actind1d), meshtest1d.nC)
        self.assertEqual(meshCore1d.nC, np.count_nonzero(actind1d))
        self.assertGreater(meshCore1d.cell_centers_x.min(), xzlim1d[0, :].min())
        self.assertLess(meshCore1d.cell_centers_x.max(), xzlim1d[0, :].max())

        # 2D Test on TensorMesh
        meshtest2d = discretize.TensorMesh([[(50.0, 10)], [(25.0, 10)]])
        xzlim2d = np.r_[[[0.0, 200.0], [0.0, 200.0]]]
        actind2d, meshCore2d = extract_core_mesh(xzlim2d, meshtest2d)

        self.assertEqual(len(actind2d), meshtest2d.nC)
        self.assertEqual(meshCore2d.nC, np.count_nonzero(actind2d))
        self.assertGreater(meshCore2d.cell_centers_x.min(), xzlim2d[0, :].min())
        self.assertLess(meshCore2d.cell_centers_x.max(), xzlim2d[0, :].max())
        self.assertGreater(meshCore2d.cell_centers_y.min(), xzlim2d[1, :].min())
        self.assertLess(meshCore2d.cell_centers_y.max(), xzlim2d[1, :].max())

        # 3D Test on TensorMesh
        meshtest3d = discretize.TensorMesh([[(50.0, 10)], [(25.0, 10)], [(5.0, 40)]])
        xzlim3d = np.r_[[[0.0, 250.0], [0.0, 200.0], [0.0, 150]]]
        actind3d, meshCore3d = extract_core_mesh(xzlim3d, meshtest3d)

        self.assertEqual(len(actind3d), meshtest3d.nC)
        self.assertEqual(meshCore3d.nC, np.count_nonzero(actind3d))
        self.assertGreater(meshCore3d.cell_centers_x.min(), xzlim3d[0, :].min())
        self.assertLess(meshCore3d.cell_centers_x.max(), xzlim3d[0, :].max())
        self.assertGreater(meshCore3d.cell_centers_y.min(), xzlim3d[1, :].min())
        self.assertLess(meshCore3d.cell_centers_y.max(), xzlim3d[1, :].max())
        self.assertGreater(meshCore3d.cell_centers_z.min(), xzlim3d[2, :].min())
        self.assertLess(meshCore3d.cell_centers_z.max(), xzlim3d[2, :].max())

    def test_active_from_xyz(self):
        # Create 3D topo
        [xx, yy] = np.meshgrid(np.linspace(-200, 200, 50), np.linspace(-200, 200, 50))
        b = 50
        A = 50
        zz = A * np.exp(-0.5 * ((xx / b) ** 2.0 + (yy / b) ** 2.0))

        h = [5.0, 5.0, 5.0]

        # Test 1D Mesh
        topo1D = zz[25, :].ravel()
        mesh1D = discretize.TensorMesh([np.ones(10) * 20], x0="C")

        indtopoCC = active_from_xyz(
            mesh1D, topo1D, grid_reference="CC", method="nearest"
        )
        indtopoN = active_from_xyz(mesh1D, topo1D, grid_reference="N", method="nearest")

        self.assertEqual(indtopoCC.sum(), 3)
        self.assertEqual(indtopoN.sum(), 2)

        # Test 2D Tensor mesh
        topo2D = np.c_[xx[25, :].ravel(), zz[25, :].ravel()]

        mesh_tensor = discretize.TensorMesh([[(h[0], 24)], [(h[1], 20)]], x0="CC")

        indtopoCC = active_from_xyz(
            mesh_tensor, topo2D, grid_reference="CC", method="nearest"
        )
        indtopoN = active_from_xyz(
            mesh_tensor, topo2D, grid_reference="N", method="nearest"
        )

        self.assertEqual(indtopoCC.sum(), 434)
        self.assertEqual(indtopoN.sum(), 412)

        # test 2D Curvilinear mesh
        nodes_x = mesh_tensor.gridN[:, 0].reshape(mesh_tensor.vnN, order="F")
        nodes_y = mesh_tensor.gridN[:, 1].reshape(mesh_tensor.vnN, order="F")
        mesh_curvi = discretize.CurvilinearMesh([nodes_x, nodes_y])

        indtopoCC = active_from_xyz(
            mesh_curvi, topo2D, grid_reference="CC", method="nearest"
        )
        indtopoN = active_from_xyz(
            mesh_curvi, topo2D, grid_reference="N", method="nearest"
        )

        self.assertEqual(indtopoCC.sum(), 434)
        self.assertEqual(indtopoN.sum(), 412)

        # Test 2D Tree mesh
        mesh_tree = mesh_builder_xyz(topo2D, h[:2], mesh_type="TREE")
        mesh_tree = refine_tree_xyz(
            mesh_tree,
            topo2D,
            method="surface",
            octree_levels=[1],
            octree_levels_padding=None,
            finalize=True,
        )
        indtopoCC = active_from_xyz(
            mesh_tree, topo2D, grid_reference="CC", method="nearest"
        )
        indtopoN = active_from_xyz(
            mesh_tree, topo2D, grid_reference="N", method="nearest"
        )

        self.assertEqual(indtopoCC.sum(), 167)
        self.assertEqual(indtopoN.sum(), 119)

        # Test 3D Tensor meshes
        topo3D = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

        mesh_tensor = discretize.TensorMesh(
            [[(h[0], 24)], [(h[1], 20)], [(h[2], 30)]], x0="CCC"
        )

        indtopoCC = active_from_xyz(
            mesh_tensor, topo3D, grid_reference="CC", method="nearest"
        )
        indtopoN = active_from_xyz(
            mesh_tensor, topo3D, grid_reference="N", method="nearest"
        )

        self.assertEqual(indtopoCC.sum(), 10496)
        self.assertEqual(indtopoN.sum(), 10084)

        # test 3D Curvilinear mesh
        nodes_x = mesh_tensor.gridN[:, 0].reshape(mesh_tensor.vnN, order="F")
        nodes_y = mesh_tensor.gridN[:, 1].reshape(mesh_tensor.vnN, order="F")
        nodes_z = mesh_tensor.gridN[:, 2].reshape(mesh_tensor.vnN, order="F")
        mesh_curvi = discretize.CurvilinearMesh([nodes_x, nodes_y, nodes_z])

        indtopoCC = active_from_xyz(
            mesh_curvi, topo3D, grid_reference="CC", method="nearest"
        )
        indtopoN = active_from_xyz(
            mesh_curvi, topo3D, grid_reference="N", method="nearest"
        )

        self.assertEqual(indtopoCC.sum(), 10496)
        self.assertEqual(indtopoN.sum(), 10084)

        # Test 3D Tree mesh
        mesh_tree = mesh_builder_xyz(topo3D, h, mesh_type="TREE")
        mesh_tree = refine_tree_xyz(
            mesh_tree,
            topo3D,
            method="surface",
            octree_levels=[1],
            octree_levels_padding=None,
            finalize=True,
        )
        indtopoCC = active_from_xyz(
            mesh_tree, topo3D, grid_reference="CC", method="nearest"
        )
        indtopoN = active_from_xyz(
            mesh_tree, topo3D, grid_reference="N", method="nearest"
        )

        self.assertIn(indtopoCC.sum(), [6292, 6299])
        self.assertIn(indtopoN.sum(), [4632, 4639])

        # Test 3D CYL Mesh
        ncr = 10  # number of mesh cells in r
        ncz = 15  # number of mesh cells in z
        dr = 15  # cell width r
        dz = 10  # cell width z
        npad_r = 4  # number of padding cells in r
        npad_z = 4  # number of padding cells in z
        exp_r = 1.25  # expansion rate of padding cells in r
        exp_z = 1.25  # expansion rate of padding cells in z

        hr = [(dr, ncr), (dr, npad_r, exp_r)]
        hz = [(dz, npad_z, -exp_z), (dz, ncz), (dz, npad_z, exp_z)]

        # A value of 1 is used to define the discretization in phi for this case.
        mesh_cyl = discretize.CylindricalMesh([hr, 1, hz], x0="00C")

        indtopoCC = active_from_xyz(
            mesh_cyl, topo3D, grid_reference="CC", method="nearest"
        )
        indtopoN = active_from_xyz(
            mesh_cyl, topo3D, grid_reference="N", method="nearest"
        )

        self.assertEqual(indtopoCC.sum(), 183)
        self.assertEqual(indtopoN.sum(), 171)

        htheta = unpack_widths([(1.0, 4)])
        htheta = htheta * 2 * np.pi / htheta.sum()

        mesh_cyl2 = discretize.CylindricalMesh([hr, htheta, hz], x0="00C")
        with self.assertRaises(NotImplementedError):
            indtopoCC = active_from_xyz(
                mesh_cyl2, topo3D, grid_reference="CC", method="nearest"
            )


if __name__ == "__main__":
    unittest.main()
