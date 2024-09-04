import numpy as np
import unittest

import discretize

MESHTYPES = ["uniformTensorMesh", "randomTensorMesh"]
TOLERANCES = [0.9, 0.5, 0.5]
call1 = lambda fun, xyz: fun(xyz)
call2 = lambda fun, xyz: fun(xyz[:, 0], xyz[:, -1])
call3 = lambda fun, xyz: fun(xyz[:, 0], xyz[:, 1], xyz[:, 2])
cart_row2 = lambda g, xfun, yfun: np.c_[call2(xfun, g), call2(yfun, g)]
cart_row3 = lambda g, xfun, yfun, zfun: np.c_[
    call3(xfun, g), call3(yfun, g), call3(zfun, g)
]
cartF2 = lambda M, fx, fy: np.vstack(
    (cart_row2(M.gridFx, fx, fy), cart_row2(M.gridFy, fx, fy))
)
cartF2Cyl = lambda M, fx, fy: np.vstack(
    (cart_row2(M.gridFx, fx, fy), cart_row2(M.gridFz, fx, fy))
)
cartE2 = lambda M, ex, ey: np.vstack(
    (cart_row2(M.gridEx, ex, ey), cart_row2(M.gridEy, ex, ey))
)
cartE2Cyl = lambda M, ex, ey: cart_row2(M.gridEy, ex, ey)
cartF3 = lambda M, fx, fy, fz: np.vstack(
    (
        cart_row3(M.gridFx, fx, fy, fz),
        cart_row3(M.gridFy, fx, fy, fz),
        cart_row3(M.gridFz, fx, fy, fz),
    )
)
cartE3 = lambda M, ex, ey, ez: np.vstack(
    (
        cart_row3(M.gridEx, ex, ey, ez),
        cart_row3(M.gridEy, ex, ey, ez),
        cart_row3(M.gridEz, ex, ey, ez),
    )
)

TOL = 1e-7


class TestInterpolation1D(discretize.tests.OrderTest):
    name = "Interpolation 1D"
    meshTypes = MESHTYPES
    tolerance = TOLERANCES
    meshDimension = 1
    meshSizes = [8, 16, 32, 64, 128]
    random_seed = np.random.default_rng(55124)
    LOCS = random_seed.random(50) * 0.6 + 0.2

    def getError(self):
        funX = lambda x: np.cos(2 * np.pi * x)

        ana = call1(funX, self.LOCS)

        if "CC" == self.type:
            grid = call1(funX, self.M.gridCC)
        elif "N" == self.type:
            grid = call1(funX, self.M.gridN)

        comp = self.M.get_interpolation_matrix(self.LOCS, self.type) * grid

        err = np.linalg.norm((comp - ana), 2)
        return err

    def test_orderCC(self):
        self.type = "CC"
        self.name = "Interpolation 1D: CC"
        self.orderTest()

    def test_orderN(self):
        self.type = "N"
        self.name = "Interpolation 1D: N"
        self.orderTest()


class TestOutliersInterp1D(unittest.TestCase):
    def setUp(self):
        pass

    def test_outliers(self):
        M = discretize.TensorMesh([4])
        Q = M.get_interpolation_matrix(
            np.array([[0], [0.126], [0.127]]), "CC", zeros_outside=True
        )
        x = np.arange(4) + 1
        self.assertTrue(np.linalg.norm(Q * x - np.r_[1, 1.004, 1.008]) < TOL)
        Q = M.get_interpolation_matrix(
            np.array([[-1], [0.126], [0.127]]), "CC", zeros_outside=True
        )
        self.assertTrue(np.linalg.norm(Q * x - np.r_[0, 1.004, 1.008]) < TOL)


class TestInterpolation2d(discretize.tests.OrderTest):
    name = "Interpolation 2D"
    meshTypes = MESHTYPES
    tolerance = TOLERANCES
    meshDimension = 2
    meshSizes = [8, 16, 32, 64]
    random_seed = np.random.default_rng(2457)
    LOCS = random_seed.random((50, 2)) * 0.6 + 0.2

    def getError(self):
        funX = lambda x, y: np.cos(2 * np.pi * y)
        funY = lambda x, y: np.cos(2 * np.pi * x)

        if "x" in self.type:
            ana = call2(funX, self.LOCS)
        elif "y" in self.type:
            ana = call2(funY, self.LOCS)
        else:
            ana = call2(funX, self.LOCS)

        if "F" in self.type:
            Fc = cartF2(self.M, funX, funY)
            grid = self.M.project_face_vector(Fc)
        elif "E" in self.type:
            Ec = cartE2(self.M, funX, funY)
            grid = self.M.project_edge_vector(Ec)
        elif "CC" == self.type:
            grid = call2(funX, self.M.gridCC)
        elif "N" == self.type:
            grid = call2(funX, self.M.gridN)

        comp = self.M.get_interpolation_matrix(self.LOCS, self.type) * grid

        err = np.linalg.norm((comp - ana), np.inf)
        return err

    def test_orderCC(self):
        self.type = "CC"
        self.name = "Interpolation 2D: CC"
        self.orderTest()

    def test_orderN(self):
        self.type = "N"
        self.name = "Interpolation 2D: N"
        self.orderTest()

    def test_orderFx(self):
        self.type = "Fx"
        self.name = "Interpolation 2D: Fx"
        self.orderTest()

    def test_orderFy(self):
        self.type = "Fy"
        self.name = "Interpolation 2D: Fy"
        self.orderTest()

    def test_orderEx(self):
        self.type = "Ex"
        self.name = "Interpolation 2D: Ex"
        self.orderTest()

    def test_orderEy(self):
        self.type = "Ey"
        self.name = "Interpolation 2D: Ey"
        self.orderTest()


class TestInterpolationSymmetricCyl_Simple(unittest.TestCase):
    def test_simpleInter(self):
        M = discretize.CylindricalMesh([4, 1, 1])
        locs = np.r_[0, 0, 0.5]
        fx = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        self.assertTrue(np.all(fx == M.get_interpolation_matrix(locs, "Fx").todense()))
        fz = np.array([[0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]])
        self.assertTrue(np.all(fz == M.get_interpolation_matrix(locs, "Fz").todense()))

    def test_exceptions(self):
        M = discretize.CylindricalMesh([4, 1, 1])
        locs = np.r_[0, 0, 0.5]
        self.assertRaises(Exception, lambda: M.get_interpolation_matrix(locs, "Fy"))
        self.assertRaises(Exception, lambda: M.get_interpolation_matrix(locs, "Ex"))
        self.assertRaises(Exception, lambda: M.get_interpolation_matrix(locs, "Ez"))


class TestInterpolationSymCyl(discretize.tests.OrderTest):
    name = "Interpolation Symmetric 3D"
    meshTypes = ["uniform_symmetric_CylMesh"]  # MESHTYPES +
    tolerance = 0.6
    meshDimension = 3
    meshSizes = [32, 64, 128, 256]
    random_seed = np.random.default_rng(81756234)
    LOCS = np.c_[
        random_seed.random(4) * 0.6 + 0.2,
        np.zeros(4),
        random_seed.random(4) * 0.6 + 0.2,
    ]

    def getError(self):
        funX = lambda x, y: np.cos(2 * np.pi * y)
        funY = lambda x, y: np.cos(2 * np.pi * x)

        if "x" in self.type:
            ana = call2(funX, self.LOCS)
        elif "y" in self.type:
            ana = call2(funY, self.LOCS)
        elif "z" in self.type:
            ana = call2(funY, self.LOCS)
        else:
            ana = call2(funX, self.LOCS)

        if "Fx" == self.type:
            Fc = cartF2Cyl(self.M, funX, funY)
            Fc = np.c_[Fc[:, 0], np.zeros(self.M.nF), Fc[:, 1]]
            grid = self.M.project_face_vector(Fc)
        elif "Fz" == self.type:
            Fc = cartF2Cyl(self.M, funX, funY)
            Fc = np.c_[Fc[:, 0], np.zeros(self.M.nF), Fc[:, 1]]

            grid = self.M.project_face_vector(Fc)
        elif "E" in self.type:
            Ec = cartE2Cyl(self.M, funX, funY)
            grid = Ec[:, 1]
        elif "CC" == self.type:
            grid = call2(funX, self.M.gridCC)
        elif "N" == self.type:
            grid = call2(funX, self.M.gridN)

        comp = self.M.get_interpolation_matrix(self.LOCS, self.type) * grid

        err = np.linalg.norm((comp - ana), np.inf)
        return err

    def test_orderCC(self):
        self.type = "CC"
        self.name = "Interpolation 3D Symmetric CYLMESH: CC"
        self.orderTest()

    def test_orderFx(self):
        self.type = "Fx"
        self.name = "Interpolation 3D Symmetric CYLMESH: Fx"
        self.orderTest()

    def test_orderFz(self):
        self.type = "Fz"
        self.name = "Interpolation 3D Symmetric CYLMESH: Fz"
        self.orderTest()

    def test_orderEy(self):
        self.type = "Ey"
        self.name = "Interpolation 3D Symmetric CYLMESH: Ey"
        self.orderTest()


class TestInterpolationCyl(discretize.tests.OrderTest):
    name = "Interpolation Cylindrical 3D"
    meshTypes = ["uniformCylMesh", "randomCylMesh"]  # MESHTYPES +
    meshDimension = 3
    meshSizes = [8, 16, 32, 64]
    random_seed = np.random.default_rng(876234)
    LOCS = np.c_[
        random_seed.random(20) * 0.6 + 0.2,
        2 * np.pi * (random_seed.random(20) * 0.6 + 0.2),
        random_seed.random(20) * 0.6 + 0.2,
    ]

    def getError(self):
        func = lambda x, y, z: np.cos(2 * np.pi * x) + np.cos(y) + np.cos(2 * np.pi * z)
        ana = func(*self.LOCS.T)
        mesh = self.M

        if "F" in self.type:
            v = func(*mesh.faces.T)
        elif "E" in self.type:
            v = func(*mesh.edges.T)
        elif "CC" == self.type:
            v = func(*mesh.cell_centers.T)
        elif "N" == self.type:
            v = func(*mesh.nodes.T)

        comp = mesh.get_interpolation_matrix(self.LOCS, self.type) * v

        err = np.linalg.norm((comp - ana), np.inf)
        return err

    def test_orderCC(self):
        self.type = "CC"
        self.name = "Interpolation 3D CYLMESH: CC"
        self.orderTest()

    def test_orderN(self):
        self.type = "N"
        self.name = "Interpolation 3D CYLMESH: N"
        self.orderTest()

    def test_orderFx(self):
        self.type = "Fx"
        self.name = "Interpolation 3D CYLMESH: Fx"
        self.orderTest()

    def test_orderFy(self):
        self.type = "Fy"
        self.name = "Interpolation 3D CYLMESH: Fy"
        self.orderTest()

    def test_orderFz(self):
        self.type = "Fz"
        self.name = "Interpolation 3D CYLMESH: Fz"
        self.orderTest()

    def test_orderEx(self):
        self.type = "Ex"
        self.name = "Interpolation 3D CYLMESH: Ex"
        self.orderTest()

    def test_orderEy(self):
        self.type = "Ey"
        self.name = "Interpolation 3D CYLMESH: Ey"
        self.orderTest()

    def test_orderEz(self):
        self.type = "Ez"
        self.name = "Interpolation 3D CYLMESH: Ez"
        self.orderTest()


class TestInterpolation3D(discretize.tests.OrderTest):
    random_seed = np.random.default_rng(234)
    name = "Interpolation"
    LOCS = random_seed.random((50, 3)) * 0.6 + 0.2
    meshTypes = MESHTYPES
    tolerance = TOLERANCES
    meshDimension = 3
    meshSizes = [8, 16, 32, 64]

    def getError(self):
        funX = lambda x, y, z: np.cos(2 * np.pi * y)
        funY = lambda x, y, z: np.cos(2 * np.pi * z)
        funZ = lambda x, y, z: np.cos(2 * np.pi * x)

        if "x" in self.type:
            ana = call3(funX, self.LOCS)
        elif "y" in self.type:
            ana = call3(funY, self.LOCS)
        elif "z" in self.type:
            ana = call3(funZ, self.LOCS)
        else:
            ana = call3(funX, self.LOCS)

        if "F" in self.type:
            Fc = cartF3(self.M, funX, funY, funZ)
            grid = self.M.project_face_vector(Fc)
        elif "E" in self.type:
            Ec = cartE3(self.M, funX, funY, funZ)
            grid = self.M.project_edge_vector(Ec)
        elif "CC" == self.type:
            grid = call3(funX, self.M.gridCC)
        elif "N" == self.type:
            grid = call3(funX, self.M.gridN)

        comp = self.M.get_interpolation_matrix(self.LOCS, self.type) * grid

        err = np.linalg.norm((comp - ana), np.inf)
        return err

    def test_orderCC(self):
        self.type = "CC"
        self.name = "Interpolation 3D: CC"
        self.orderTest()

    def test_orderN(self):
        self.type = "N"
        self.name = "Interpolation 3D: N"
        self.orderTest()

    def test_orderFx(self):
        self.type = "Fx"
        self.name = "Interpolation 3D: Fx"
        self.orderTest()

    def test_orderFy(self):
        self.type = "Fy"
        self.name = "Interpolation 3D: Fy"
        self.orderTest()

    def test_orderFz(self):
        self.type = "Fz"
        self.name = "Interpolation 3D: Fz"
        self.orderTest()

    def test_orderEx(self):
        self.type = "Ex"
        self.name = "Interpolation 3D: Ex"
        self.orderTest()

    def test_orderEy(self):
        self.type = "Ey"
        self.name = "Interpolation 3D: Ey"
        self.orderTest()

    def test_orderEz(self):
        self.type = "Ez"
        self.name = "Interpolation 3D: Ez"
        self.orderTest()


if __name__ == "__main__":
    unittest.main()
