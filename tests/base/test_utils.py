from __future__ import print_function
import unittest
import numpy as np
import scipy.sparse as sp
from discretize.utils import (
    sdiag, sub2ind, ndgrid, mkvc, isScalar,
    inv2X2BlockDiagonal, inv3X3BlockDiagonal,
    invPropertyTensor, makePropertyTensor, indexCube,
    ind2sub, asArray_N_x_Dim, TensorType, Zero, Identity,
    ExtractCoreMesh, active_from_xyz, mesh_builder_xyz, refine_tree_xyz
)
from discretize.Tests import checkDerivative
import discretize
import sys

TOL = 1e-8


class TestCheckDerivative(unittest.TestCase):

    def test_simplePass(self):
        def simplePass(x):
            return np.sin(x), sdiag(np.cos(x))
        passed = checkDerivative(simplePass, np.random.randn(5), plotIt=False)
        self.assertTrue(passed, True)

    def test_simpleFunction(self):
        def simpleFunction(x):
            return np.sin(x), lambda xi: sdiag(np.cos(x))*xi
        passed = checkDerivative(simpleFunction, np.random.randn(5), plotIt=False)
        self.assertTrue(passed, True)

    def test_simpleFail(self):
        def simpleFail(x):
            return np.sin(x), -sdiag(np.cos(x))
        passed = checkDerivative(simpleFail, np.random.randn(5), plotIt=False)
        self.assertTrue(not passed, True)


class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        self.a = np.array([1, 2, 3])
        self.b = np.array([1, 2])
        self.c = np.array([1, 2, 3, 4])

    def test_mkvc1(self):
        x = mkvc(self.a)
        self.assertTrue(x.shape, (3, ))

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

        X1_test = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
        X2_test = np.array([1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 2, 2])
        X3_test = np.array([1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4])

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
        self.assertTrue(np.all(sub2ind(x.shape, [[0, 0], [4, 0], [0, 1], [4, 1]]) == [0, 4, 5, 9]))

    def test_ind2sub(self):
        x = np.ones((5, 2))
        self.assertTrue(np.all(ind2sub(x.shape, [0, 4, 5, 9])[0] == [0, 4, 0, 4]))
        self.assertTrue(np.all(ind2sub(x.shape, [0, 4, 5, 9])[1] == [0, 0, 1, 1]))

    def test_indexCube_2D(self):
        nN = np.array([3, 3])
        self.assertTrue(np.all(indexCube('A', nN) == np.array([0, 1, 3, 4])))
        self.assertTrue(np.all(indexCube('B', nN) == np.array([3, 4, 6, 7])))
        self.assertTrue(np.all(indexCube('C', nN) == np.array([4, 5, 7, 8])))
        self.assertTrue(np.all(indexCube('D', nN) == np.array([1, 2, 4, 5])))

    def test_indexCube_3D(self):
        nN = np.array([3, 3, 3])
        self.assertTrue(np.all(indexCube('A', nN) == np.array([0, 1, 3, 4, 9, 10, 12, 13])))
        self.assertTrue(np.all(indexCube('B', nN) == np.array([3, 4, 6, 7, 12, 13, 15, 16])))
        self.assertTrue(np.all(indexCube('C', nN) == np.array([4, 5, 7, 8, 13, 14, 16, 17])))
        self.assertTrue(np.all(indexCube('D', nN) == np.array([1, 2, 4, 5, 10, 11, 13, 14])))
        self.assertTrue(np.all(indexCube('E', nN) == np.array([9, 10, 12, 13, 18, 19, 21, 22])))
        self.assertTrue(np.all(indexCube('F', nN) == np.array([12, 13, 15, 16, 21, 22, 24, 25])))
        self.assertTrue(np.all(indexCube('G', nN) == np.array([13, 14, 16, 17, 22, 23, 25, 26])))
        self.assertTrue(np.all(indexCube('H', nN) == np.array([10, 11, 13, 14, 19, 20, 22, 23])))

    def test_invXXXBlockDiagonal(self):
        a = [np.random.rand(5, 1) for i in range(4)]

        B = inv2X2BlockDiagonal(*a)

        A = sp.vstack((sp.hstack((sdiag(a[0]), sdiag(a[1]))),
                       sp.hstack((sdiag(a[2]), sdiag(a[3])))))

        Z2 = B*A - sp.identity(10)
        self.assertTrue(np.linalg.norm(Z2.todense().ravel(), 2) < TOL)

        a = [np.random.rand(5, 1) for i in range(9)]
        B = inv3X3BlockDiagonal(*a)

        A = sp.vstack((sp.hstack((sdiag(a[0]), sdiag(a[1]),  sdiag(a[2]))),
                       sp.hstack((sdiag(a[3]), sdiag(a[4]),  sdiag(a[5]))),
                       sp.hstack((sdiag(a[6]), sdiag(a[7]),  sdiag(a[8])))))

        Z3 = B*A - sp.identity(15)

        self.assertTrue(np.linalg.norm(Z3.todense().ravel(), 2) < TOL)

    def test_invPropertyTensor2D(self):
        M = discretize.TensorMesh([6, 6])
        a1 = np.random.rand(M.nC)
        a2 = np.random.rand(M.nC)
        a3 = np.random.rand(M.nC)
        prop1 = a1
        prop2 = np.c_[a1, a2]
        prop3 = np.c_[a1, a2, a3]

        for prop in [4, prop1, prop2, prop3]:
            b = invPropertyTensor(M, prop)
            A = makePropertyTensor(M, prop)
            B1 = makePropertyTensor(M, b)
            B2 = invPropertyTensor(M, prop, returnMatrix=True)

            Z = B1*A - sp.identity(M.nC*2)
            self.assertTrue(np.linalg.norm(Z.todense().ravel(), 2) < TOL)
            Z = B2*A - sp.identity(M.nC*2)
            self.assertTrue(np.linalg.norm(Z.todense().ravel(), 2) < TOL)

    def test_TensorType2D(self):
        M = discretize.TensorMesh([6, 6])
        a1 = np.random.rand(M.nC)
        a2 = np.random.rand(M.nC)
        a3 = np.random.rand(M.nC)
        prop1 = a1
        prop2 = np.c_[a1, a2]
        prop3 = np.c_[a1, a2, a3]

        for ii, prop in enumerate([4, prop1, prop2, prop3]):
            self.assertTrue(TensorType(M, prop) == ii)

        self.assertRaises(Exception, TensorType, M, np.c_[a1, a2, a3, a3])
        self.assertTrue(TensorType(M, None) == -1)

    def test_TensorType3D(self):
        M = discretize.TensorMesh([6, 6, 7])
        a1 = np.random.rand(M.nC)
        a2 = np.random.rand(M.nC)
        a3 = np.random.rand(M.nC)
        a4 = np.random.rand(M.nC)
        a5 = np.random.rand(M.nC)
        a6 = np.random.rand(M.nC)
        prop1 = a1
        prop2 = np.c_[a1, a2, a3]
        prop3 = np.c_[a1, a2, a3, a4, a5, a6]

        for ii, prop in enumerate([4, prop1, prop2, prop3]):
            self.assertTrue(TensorType(M, prop) == ii)

        self.assertRaises(Exception, TensorType, M, np.c_[a1, a2, a3, a3])
        self.assertTrue(TensorType(M, None) == -1)

    def test_invPropertyTensor3D(self):
        M = discretize.TensorMesh([6, 6, 6])
        a1 = np.random.rand(M.nC)
        a2 = np.random.rand(M.nC)
        a3 = np.random.rand(M.nC)
        a4 = np.random.rand(M.nC)
        a5 = np.random.rand(M.nC)
        a6 = np.random.rand(M.nC)
        prop1 = a1
        prop2 = np.c_[a1, a2, a3]
        prop3 = np.c_[a1, a2, a3, a4, a5, a6]

        for prop in [4, prop1, prop2, prop3]:
            b = invPropertyTensor(M, prop)
            A = makePropertyTensor(M, prop)
            B1 = makePropertyTensor(M, b)
            B2 = invPropertyTensor(M, prop, returnMatrix=True)

            Z = B1*A - sp.identity(M.nC*3)
            self.assertTrue(np.linalg.norm(Z.todense().ravel(), 2) < TOL)
            Z = B2*A - sp.identity(M.nC*3)
            self.assertTrue(np.linalg.norm(Z.todense().ravel(), 2) < TOL)

    def test_isScalar(self):
        self.assertTrue(isScalar(1.))
        self.assertTrue(isScalar(1))
        self.assertTrue(isScalar(1j))
        if sys.version_info < (3, ):
            self.assertTrue(isScalar(long(1)))
        self.assertTrue(isScalar(np.r_[1.]))
        self.assertTrue(isScalar(np.r_[1]))
        self.assertTrue(isScalar(np.r_[1j]))

    def test_asArray_N_x_Dim(self):

        true = np.array([[1, 2, 3]])

        listArray = asArray_N_x_Dim([1, 2, 3], 3)
        self.assertTrue(np.all(true == listArray))
        self.assertTrue(true.shape == listArray.shape)

        listArray = asArray_N_x_Dim(np.r_[1, 2, 3], 3)
        self.assertTrue(np.all(true == listArray))
        self.assertTrue(true.shape == listArray.shape)

        listArray = asArray_N_x_Dim(np.array([[1, 2, 3.]]), 3)
        self.assertTrue(np.all(true == listArray))
        self.assertTrue(true.shape == listArray.shape)

        true = np.array([[1, 2], [4, 5]])

        listArray = asArray_N_x_Dim([[1, 2], [4, 5]], 2)
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
        assert 3*z == 0
        assert z*3 == 0
        assert z/3 == 0

        a = 1
        a += z
        assert a == 1
        a = 1
        a += z
        assert a == 1
        self.assertRaises(ZeroDivisionError, lambda: 3/z)

        assert mkvc(z) == 0
        assert sdiag(z)*a == 0
        assert z.T == 0
        assert z.transpose() == 0

    def test_mat_zero(self):
        z = Zero()
        S = sdiag(np.r_[2, 3])
        assert S*z == 0

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
        assert -1.*(-o)*o == -o
        o = Identity()
        assert +o == o
        assert -o == -o
        assert o*3 == 3
        assert -o*3 == -3
        assert -o*o == -1
        assert -o*o*-o == 1
        assert -o + 3 == 2
        assert 3 + -o == 2

        assert -o - 3 == -4
        assert o - 3 == -2
        assert 3 - -o == 4
        assert 3 - o == 2

        assert o//2 == 0
        assert o/2. == 0.5
        assert -o//2 == -1
        assert -o/2. == -0.5
        assert 2/o == 2
        assert 2/-o == -2

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
        check(S/o, [[2, 0], [0, 3]])
        check(S/-o, [[-2, 0], [0, -3]])
        self.assertRaises(NotImplementedError, lambda: o/S)

        check(S + o, [[3, 0], [0, 4]])
        check(o + S, [[3, 0], [0, 4]])
        check(S - o, [[1, 0], [0, 2]])

        check(S + - o, [[1, 0], [0, 2]])
        check(- o + S, [[1, 0], [0, 2]])

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
        n = np.r_[2., 3]

        assert np.all(n+1 == n+o)
        assert np.all(1+n == o+n)
        assert np.all(n-1 == n-o)
        assert np.all(1-n == o-n)
        assert np.all(n/1 == n/o)
        assert np.all(n/-1 == n/-o)
        assert np.all(1/n == o/n)
        assert np.all(-1/n == -o/n)
        assert np.all(n*1 == n*o)
        assert np.all(n*-1 == n*-o)
        assert np.all(1*n == o*n)
        assert np.all(-1*n == -o*n)

    def test_both(self):
        z = Zero()
        o = Identity()
        assert o*z == 0
        assert o*z + o == 1
        assert o-z == 1


class TestMeshUtils(unittest.TestCase):

    def test_ExtractCoreMesh(self):

        # 1D Test on TensorMesh
        meshtest1d = discretize.TensorMesh([[(50., 10)]])
        xzlim1d = np.r_[[[0., 250.]]]
        actind1d, meshCore1d = ExtractCoreMesh(xzlim1d, meshtest1d)

        assert len(actind1d) == meshtest1d.nC
        assert meshCore1d.nC == np.count_nonzero(actind1d)
        assert meshCore1d.vectorCCx.min() > xzlim1d[0, :].min()
        assert meshCore1d.vectorCCx.max() < xzlim1d[0, :].max()

        # 2D Test on TensorMesh
        meshtest2d = discretize.TensorMesh([[(50., 10)], [(25., 10)]])
        xzlim2d = xyzlim = np.r_[[[0., 200.], [0., 200.]]]
        actind2d, meshCore2d = ExtractCoreMesh(xzlim2d, meshtest2d)

        assert len(actind2d) == meshtest2d.nC
        assert meshCore2d.nC == np.count_nonzero(actind2d)
        assert meshCore2d.vectorCCx.min() > xzlim2d[0, :].min()
        assert meshCore2d.vectorCCx.max() < xzlim2d[0, :].max()
        assert meshCore2d.vectorCCy.min() > xzlim2d[1, :].min()
        assert meshCore2d.vectorCCy.max() < xzlim2d[1, :].max()

        # 3D Test on TensorMesh
        meshtest3d = discretize.TensorMesh([[(50., 10)], [(25., 10)], [(5., 40)]])
        xzlim3d = np.r_[[[0., 250.], [0., 200.], [0., 150]]]
        actind3d, meshCore3d = ExtractCoreMesh(xzlim3d, meshtest3d)

        assert len(actind3d) == meshtest3d.nC
        assert meshCore3d.nC == np.count_nonzero(actind3d)
        assert meshCore3d.vectorCCx.min() > xzlim3d[0, :].min()
        assert meshCore3d.vectorCCx.max() < xzlim3d[0, :].max()
        assert meshCore3d.vectorCCy.min() > xzlim3d[1, :].min()
        assert meshCore3d.vectorCCy.max() < xzlim3d[1, :].max()
        assert meshCore3d.vectorCCz.min() > xzlim3d[2, :].min()
        assert meshCore3d.vectorCCz.max() < xzlim3d[2, :].max()

    def test_active_from_xyz(self):

        # Create 3D topo
        [xx, yy] = np.meshgrid(np.linspace(-200, 200, 50), np.linspace(-200, 200, 50))
        b = 50
        A = 50
        zz = A * np.exp(-0.5 * ((xx / b) ** 2. + (yy / b) ** 2.))

        h = [5., 5., 5.]

        # Test 1D Mesh
        topo1D = zz[25, :].ravel()
        mesh1D = discretize.TensorMesh(
            [np.ones(10) * 20],
            x0='C'
        )

        indtopoCC = active_from_xyz(mesh1D, topo1D, grid_reference='CC', method='nearest')
        indtopoN = active_from_xyz(mesh1D, topo1D, grid_reference='N', method='nearest')

        assert indtopoCC.sum() == 3
        assert indtopoN.sum() == 2
        #
        # plt.figure()
        # axs = plt.subplot()
        # axs.step(mesh1D.gridCC, indtopoN)
        # axs.step(mesh1D.gridCC, indtopoCC)

        # Test 2D Tensor mesh
        topo2D = np.c_[xx[25, :].ravel(), zz[25, :].ravel()]

        mesh_tensor = discretize.TensorMesh([
            [(h[0], 24)],
            [(h[1], 20)]
        ],
            x0='CC')

        indtopoCC = active_from_xyz(mesh_tensor, topo2D, grid_reference='CC', method='nearest')
        indtopoN = active_from_xyz(mesh_tensor, topo2D, grid_reference='N', method='nearest')

        assert indtopoCC.sum() == 434
        assert indtopoN.sum() == 412
        # plt.figure()
        # ax1 = plt.subplot()
        # mesh_tensor.plotImage(indtopoCC, grid=True, ax=ax1)
        # ax1.plot(topo2D[:, 0], topo2D[:, 1])

        # Test 2D Tree mesh
        mesh_tree = mesh_builder_xyz(topo2D, h[:2], mesh_type='TREE')
        mesh_tree = refine_tree_xyz(
            mesh_tree, topo2D,
            method="surface",
            octree_levels=[1],
            octree_levels_padding=None,
            finalize=True
        )
        indtopoCC = active_from_xyz(mesh_tree, topo2D, grid_reference='CC', method='nearest')
        indtopoN = active_from_xyz(mesh_tree, topo2D, grid_reference='N', method='nearest')

        assert indtopoCC.sum() == 167
        assert indtopoN.sum() == 119
        # plt.figure()
        # ax1 = plt.subplot(1,2,1)
        # mesh_tree.plotImage(indtopoCC, grid=True, ax=ax1)
        # ax1.plot(topo2D[:, 0], topo2D[:, 1])
        # ax2 = plt.subplot(1,2,2)
        # mesh_tree.plotImage(indtopoN, grid=True, ax=ax2)
        # ax2.plot(topo2D[:, 0], topo2D[:, 1])

        # assert len(np.where(indtopoCC)[0]) == 8729
        # assert len(np.where(indtopoN)[0]) == 8212

        # Test 3D Tensor meshes
        topo3D = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

        mesh_tensor = discretize.TensorMesh([
            [(h[0], 24)],
            [(h[1], 20)],
            [(h[2], 30)]
        ],
            x0='CCC')

        indtopoCC = active_from_xyz(mesh_tensor, topo3D, grid_reference='CC', method='nearest')
        indtopoN = active_from_xyz(mesh_tensor, topo3D, grid_reference='N', method='nearest')

        assert indtopoCC.sum() == 10496
        assert indtopoN.sum() == 10084
        # plt.figure()
        # ax1 = plt.subplot()
        # mesh_tensor.plotSlice(indtopoCC+indtopoN, normal='Y', grid=True, ax=ax1)
        # ax1.set_aspect('equal')

        # Test 3D Tree mesh
        mesh_tree = mesh_builder_xyz(topo3D, h, mesh_type='TREE')
        mesh_tree = refine_tree_xyz(
            mesh_tree, topo3D,
            method="surface",
            octree_levels=[1],
            octree_levels_padding=None,
            finalize=True
        )
        indtopoCC = active_from_xyz(mesh_tree, topo3D, grid_reference='CC', method='nearest')
        indtopoN = active_from_xyz(mesh_tree, topo3D, grid_reference='N', method='nearest')

        assert indtopoCC.sum() == 6292
        assert indtopoN.sum() == 4632
        # plt.figure()
        # axs = plt.subplot(1,2,1)
        # mesh_tree.plotSlice(indtopoCC, normal='Y', grid=True, ax=axs)
        # axs = plt.subplot(1,2,2)
        # mesh_tree.plotSlice(indtopoN, normal='Y', grid=True, ax=axs)

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
        mesh_cyl = discretize.CylMesh([hr, 1, hz], x0='00C')

        # The bottom end of the vertical axis of rotational symmetry
        x0 = mesh_cyl.x0

        # The total number of cells
        nC = mesh_cyl.nC

        # An (nC, 3) array containing the cell-center locations
        cc = mesh_cyl.gridCC

        # Plot the cell volumes.
        v = mesh_cyl.vol

        indtopoCC = active_from_xyz(mesh_cyl, topo3D, grid_reference='CC', method='nearest')
        indtopoN = active_from_xyz(mesh_cyl, topo3D, grid_reference='N', method='nearest')

        assert indtopoCC.sum() == 183
        assert indtopoN.sum() == 171
        # plt.figure()
        # axs = plt.subplot(1,2,1)
        # mesh_cyl.plotImage(indtopoCC, grid=True, ax=axs)
        # axs = plt.subplot(1,2,2)
        # mesh_cyl.plotImage(indtopoN, grid=True, ax=axs)


if __name__ == '__main__':
    unittest.main()
