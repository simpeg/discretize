from __future__ import print_function
import unittest
import numpy as np
import scipy.sparse as sp
from discretize.utils import (
    sdiag, sub2ind, ndgrid, mkvc, isScalar,
    inv2X2BlockDiagonal, inv3X3BlockDiagonal,
    invPropertyTensor, makePropertyTensor, indexCube,
    ind2sub, asArray_N_x_Dim, TensorType, Zero, Identity,
    ExtractCoreMesh
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
        if sys.version_info < (3, ):
            self.assertTrue(isScalar(long(1)))
        self.assertTrue(isScalar(np.r_[1.]))
        self.assertTrue(isScalar(np.r_[1]))

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

    def test_get_domain():
        # Test default values (and therefore skindepth etc)
        h1, d1 = utils.get_domain()
        assert_allclose(h1, 55.133753)
        assert_allclose(d1, [-1378.343816, 1378.343816])

        # Ensure fact_min/fact_neg/fact_pos
        h2, d2 = utils.get_domain(fact_min=1, fact_neg=10, fact_pos=20)
        assert h2 == 5*h1
        assert 2*d1[0] == d2[0]
        assert -2*d2[0] == d2[1]

        # Check limits and min_width
        h3, d3 = utils.get_domain(limits=[-10000, 10000], min_width=[1, 10])
        assert h3 == 10
        assert np.sum(d3) == 0

    def test_get_stretched_h(capsys):
        # Test min_space bigger (11) then required (10)
        h1 = utils.get_stretched_h(11, [0, 100], nx=10)
        assert_allclose(np.ones(10)*10, h1)
        out, _ = capsys.readouterr()  # Empty capsys
        assert "extent :        0.0 - 100.0" in out
        assert "min/max width:   10.0 - 10.0  ; stretching: 1.000" in out

        # Test with range, wont end at 100
        h2 = utils.get_stretched_h(10, [-100, 100], nx=10, x0=20, x1=60)
        assert_allclose(np.ones(4)*10, h2[5:9])
        assert -100+np.sum(h2) != 100

        # Now ensure 100
        h3 = utils.get_stretched_h(10, [-100, 100], nx=10, x0=20, x1=60,
                                resp_domain=True)
        assert -100+np.sum(h3) == 100

        out, _ = capsys.readouterr()  # Empty capsys
        _ = utils.get_stretched_h(10, [-100, 100], nx=5, x0=20, x1=60)
        out, _ = capsys.readouterr()
        assert "Warning :: Not enough points for non-stretched part" in out

    def test_get_hx():
        # Test alpha <= 0
        hx1 = utils.get_hx(-.5, [0, 10], 5, 3.33)
        assert_allclose(np.ones(5)*2, hx1)

        # Test x0 on domain
        hx2a = utils.get_hx(0.1, [0, 10], 5, 0)
        assert_allclose(np.ones(4)*1.1, hx2a[1:]/hx2a[:-1])
        hx2b = utils.get_hx(0.1, [0, 10], 5, 10)
        assert_allclose(np.ones(4)/1.1, hx2b[1:]/hx2b[:-1])
        assert np.sum(hx2b) == 10.0

        # Test resp_domain
        hx3 = utils.get_hx(0.1, [0, 10], 3, 8, False)
        assert np.sum(hx3) != 10.0

if __name__ == '__main__':
    unittest.main()
