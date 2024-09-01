import numpy as np
import unittest
import discretize

# Tolerance
TOL = 1e-14

gen = np.random.default_rng(1)

MESHTYPES = [
    "uniformTensorMesh",
    # 'randomTensorMesh',
    "uniformCurv",
    "rotateCurv",
]
ORDERS = np.r_[2.0, 2.0, 2.0]
call2 = lambda fun, xyz: fun(xyz[:, 0], xyz[:, 1])
call3 = lambda fun, xyz: fun(xyz[:, 0], xyz[:, 1], xyz[:, 2])
cart_row2 = lambda g, xfun, yfun: np.c_[call2(xfun, g), call2(yfun, g)]
cart_row3 = lambda g, xfun, yfun, zfun: np.c_[
    call3(xfun, g), call3(yfun, g), call3(zfun, g)
]
cartF2 = lambda M, fx, fy: np.vstack(
    (cart_row2(M.gridFx, fx, fy), cart_row2(M.gridFy, fx, fy))
)
cartE2 = lambda M, ex, ey: np.vstack(
    (cart_row2(M.gridEx, ex, ey), cart_row2(M.gridEy, ex, ey))
)
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


class TestCurl(discretize.tests.OrderTest):
    name = "Curl"
    meshTypes = MESHTYPES

    def getError(self):
        # fun: i (cos(y)) + j (cos(z)) + k (cos(x))
        # sol: i (sin(z)) + j (sin(x)) + k (sin(y))

        funX = lambda x, y, z: np.cos(2 * np.pi * y)
        funY = lambda x, y, z: np.cos(2 * np.pi * z)
        funZ = lambda x, y, z: np.cos(2 * np.pi * x)

        solX = lambda x, y, z: 2 * np.pi * np.sin(2 * np.pi * z)
        solY = lambda x, y, z: 2 * np.pi * np.sin(2 * np.pi * x)
        solZ = lambda x, y, z: 2 * np.pi * np.sin(2 * np.pi * y)

        Ec = cartE3(self.M, funX, funY, funZ)
        E = self.M.project_edge_vector(Ec)

        Fc = cartF3(self.M, solX, solY, solZ)
        curlE_ana = self.M.project_face_vector(Fc)

        curlE = self.M.edge_curl.dot(E)
        if self._meshType == "rotateCurv":
            # Really it is the integration we should be caring about:
            # So, let us look at the l2 norm.
            err = np.linalg.norm(self.M.face_areas * (curlE - curlE_ana), 2)
        else:
            err = np.linalg.norm((curlE - curlE_ana), np.inf)
        return err

    def test_order(self):
        self.orderTest()


class TestCurl2D(discretize.tests.OrderTest):
    name = "Cell Grad 2D - Dirichlet"
    meshTypes = ["uniformTensorMesh"]
    meshDimension = 2
    meshSizes = [8, 16, 32, 64]

    def getError(self):
        # Test function
        ex = lambda x, y: np.cos(y)
        ey = lambda x, y: np.cos(x)
        sol = lambda x, y: -np.sin(x) + np.sin(y)

        sol_curl2d = call2(sol, self.M.gridCC)
        Ec = cartE2(self.M, ex, ey)
        sol_ana = self.M.edge_curl * self.M.project_face_vector(Ec)
        err = np.linalg.norm((sol_curl2d - sol_ana), np.inf)

        return err

    def test_order(self):
        self.orderTest()


# class TestCellGrad1D_InhomogeneousDirichlet(discretize.tests.OrderTest):
#     name = "Cell Grad 1D - Dirichlet"
#     meshTypes = ["uniformTensorMesh"]
#     meshDimension = 1
#     expectedOrders = (
#         1  # because of the averaging involved in the ghost point. u_b = (u_n + u_g)/2
#     )
#     meshSizes = [8, 16, 32, 64]
#
#     def getError(self):
#         # Test function
#         fx = lambda x: -2 * np.pi * np.sin(2 * np.pi * x)
#         sol = lambda x: np.cos(2 * np.pi * x)
#
#         xc = sol(self.M.gridCC)
#
#         gradX_ana = fx(self.M.gridFx)
#
#         bc = np.array([1, 1])
#         self.M.set_cell_gradient_BC("dirichlet")
#         gradX = self.M.cell_gradient.dot(xc) + self.M.cellGradBC * bc
#
#         err = np.linalg.norm((gradX - gradX_ana), np.inf)
#
#         return err
#
#     def test_order(self):
#         self.orderTest()


class TestCellGrad2D_Dirichlet(discretize.tests.OrderTest):
    name = "Cell Grad 2D - Dirichlet"
    meshTypes = ["uniformTensorMesh"]
    meshDimension = 2
    meshSizes = [8, 16, 32, 64]

    def getError(self):
        # Test function
        fx = lambda x, y: 2 * np.pi * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
        fy = lambda x, y: 2 * np.pi * np.cos(2 * np.pi * y) * np.sin(2 * np.pi * x)
        sol = lambda x, y: np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

        xc = call2(sol, self.M.gridCC)

        Fc = cartF2(self.M, fx, fy)
        gradX_ana = self.M.project_face_vector(Fc)

        self.M.set_cell_gradient_BC("dirichlet")
        gradX = self.M.cell_gradient.dot(xc)

        err = np.linalg.norm((gradX - gradX_ana), np.inf)

        return err

    def test_order(self):
        self.orderTest()


class TestCellGrad3D_Dirichlet(discretize.tests.OrderTest):
    name = "Cell Grad 3D - Dirichlet"
    meshTypes = ["uniformTensorMesh"]
    meshDimension = 3
    meshSizes = [8, 16, 32]

    def getError(self):
        # Test function
        fx = (
            lambda x, y, z: 2
            * np.pi
            * np.cos(2 * np.pi * x)
            * np.sin(2 * np.pi * y)
            * np.sin(2 * np.pi * z)
        )
        fy = (
            lambda x, y, z: 2
            * np.pi
            * np.sin(2 * np.pi * x)
            * np.cos(2 * np.pi * y)
            * np.sin(2 * np.pi * z)
        )
        fz = (
            lambda x, y, z: 2
            * np.pi
            * np.sin(2 * np.pi * x)
            * np.sin(2 * np.pi * y)
            * np.cos(2 * np.pi * z)
        )
        sol = (
            lambda x, y, z: np.sin(2 * np.pi * x)
            * np.sin(2 * np.pi * y)
            * np.sin(2 * np.pi * z)
        )

        xc = call3(sol, self.M.gridCC)

        Fc = cartF3(self.M, fx, fy, fz)
        gradX_ana = self.M.project_face_vector(Fc)

        self.M.set_cell_gradient_BC("dirichlet")
        gradX = self.M.cell_gradient.dot(xc)

        err = np.linalg.norm((gradX - gradX_ana), np.inf)

        return err

    def test_order(self):
        self.orderTest()


class TestCellGrad2D_Neumann(discretize.tests.OrderTest):
    name = "Cell Grad 2D - Neumann"
    meshTypes = ["uniformTensorMesh"]
    meshDimension = 2
    meshSizes = [8, 16, 32, 64]

    def getError(self):
        # Test function
        fx = lambda x, y: -2 * np.pi * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        fy = lambda x, y: -2 * np.pi * np.sin(2 * np.pi * y) * np.cos(2 * np.pi * x)
        sol = lambda x, y: np.cos(2 * np.pi * x) * np.cos(2 * np.pi * y)

        xc = call2(sol, self.M.gridCC)

        Fc = cartF2(self.M, fx, fy)
        gradX_ana = self.M.project_face_vector(Fc)

        self.M.set_cell_gradient_BC("neumann")
        gradX = self.M.cell_gradient.dot(xc)

        err = np.linalg.norm((gradX - gradX_ana), np.inf)

        return err

    def test_order(self):
        self.orderTest()


class TestCellGrad3D_Neumann(discretize.tests.OrderTest):
    name = "Cell Grad 3D - Neumann"
    meshTypes = ["uniformTensorMesh"]
    meshDimension = 3
    meshSizes = [8, 16, 32]

    def getError(self):
        # Test function
        fx = (
            lambda x, y, z: -2
            * np.pi
            * np.sin(2 * np.pi * x)
            * np.cos(2 * np.pi * y)
            * np.cos(2 * np.pi * z)
        )
        fy = (
            lambda x, y, z: -2
            * np.pi
            * np.cos(2 * np.pi * x)
            * np.sin(2 * np.pi * y)
            * np.cos(2 * np.pi * z)
        )
        fz = (
            lambda x, y, z: -2
            * np.pi
            * np.cos(2 * np.pi * x)
            * np.cos(2 * np.pi * y)
            * np.sin(2 * np.pi * z)
        )
        sol = (
            lambda x, y, z: np.cos(2 * np.pi * x)
            * np.cos(2 * np.pi * y)
            * np.cos(2 * np.pi * z)
        )

        xc = call3(sol, self.M.gridCC)

        Fc = cartF3(self.M, fx, fy, fz)
        gradX_ana = self.M.project_face_vector(Fc)

        self.M.set_cell_gradient_BC("neumann")
        gradX = self.M.cell_gradient.dot(xc)

        err = np.linalg.norm((gradX - gradX_ana), np.inf)

        return err

    def test_order(self):
        self.orderTest()


class TestFaceDiv3D(discretize.tests.OrderTest):
    name = "Face Divergence 3D"
    meshTypes = MESHTYPES
    meshSizes = [8, 16, 32, 64]

    def getError(self):
        # Test function
        fx = lambda x, y, z: np.sin(2 * np.pi * x)
        fy = lambda x, y, z: np.sin(2 * np.pi * y)
        fz = lambda x, y, z: np.sin(2 * np.pi * z)
        sol = lambda x, y, z: (
            2 * np.pi * np.cos(2 * np.pi * x)
            + 2 * np.pi * np.cos(2 * np.pi * y)
            + 2 * np.pi * np.cos(2 * np.pi * z)
        )

        Fc = cartF3(self.M, fx, fy, fz)
        F = self.M.project_face_vector(Fc)

        divF = self.M.face_divergence.dot(F)
        divF_ana = call3(sol, self.M.gridCC)

        if self._meshType == "rotateCurv":
            # Really it is the integration we should be caring about:
            # So, let us look at the l2 norm.
            err = np.linalg.norm(self.M.cell_volumes * (divF - divF_ana), 2)
        else:
            err = np.linalg.norm((divF - divF_ana), np.inf)
        return err

    def test_order(self):
        self.orderTest()


class TestFaceDiv2D(discretize.tests.OrderTest):
    name = "Face Divergence 2D"
    meshTypes = MESHTYPES
    meshDimension = 2
    meshSizes = [8, 16, 32, 64]

    def getError(self):
        # Test function
        fx = lambda x, y: np.sin(2 * np.pi * x)
        fy = lambda x, y: np.sin(2 * np.pi * y)
        sol = lambda x, y: 2 * np.pi * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))

        Fc = cartF2(self.M, fx, fy)
        F = self.M.project_face_vector(Fc)

        divF = self.M.face_divergence.dot(F)
        divF_ana = call2(sol, self.M.gridCC)

        err = np.linalg.norm((divF - divF_ana), np.inf)

        return err

    def test_order(self):
        self.orderTest()


class TestNodalGrad(discretize.tests.OrderTest):
    name = "Nodal Gradient"
    meshTypes = MESHTYPES

    def getError(self):
        # Test function
        fun = lambda x, y, z: (np.cos(x) + np.cos(y) + np.cos(z))
        # i (sin(x)) + j (sin(y)) + k (sin(z))
        solX = lambda x, y, z: -np.sin(x)
        solY = lambda x, y, z: -np.sin(y)
        solZ = lambda x, y, z: -np.sin(z)

        phi = call3(fun, self.M.gridN)
        gradE = self.M.nodal_gradient.dot(phi)

        Ec = cartE3(self.M, solX, solY, solZ)
        gradE_ana = self.M.project_edge_vector(Ec)

        err = np.linalg.norm((gradE - gradE_ana), np.inf)

        return err

    def test_order(self):
        self.orderTest()


class TestNodalGrad2D(discretize.tests.OrderTest):
    name = "Nodal Gradient 2D"
    meshTypes = MESHTYPES
    meshDimension = 2

    def getError(self):
        # Test function
        fun = lambda x, y: (np.cos(x) + np.cos(y))
        # i (sin(x)) + j (sin(y)) + k (sin(z))
        solX = lambda x, y: -np.sin(x)
        solY = lambda x, y: -np.sin(y)

        phi = call2(fun, self.M.gridN)
        gradE = self.M.nodal_gradient.dot(phi)

        Ec = cartE2(self.M, solX, solY)
        gradE_ana = self.M.project_edge_vector(Ec)

        err = np.linalg.norm((gradE - gradE_ana), np.inf)

        return err

    def test_order(self):
        self.orderTest()


class TestAveraging1D(discretize.tests.OrderTest):
    name = "Averaging 1D"
    meshTypes = ["uniformTensorMesh"]
    meshDimension = 1
    meshSizes = [16, 32, 64]

    def getError(self):
        num = self.getAve(self.M) * self.getHere(self.M)
        err = np.linalg.norm((self.getThere(self.M) - num), np.inf)
        return err

    def test_orderN2CC(self):
        self.name = "Averaging 1D: N2CC"
        fun = lambda x: np.cos(x)
        self.getHere = lambda M: fun(M.gridN)
        self.getThere = lambda M: fun(M.gridCC)
        self.getAve = lambda M: M.aveN2CC
        self.orderTest()

    def test_exactN2F(self):
        self.name = "Averaging 1D: N2F"
        fun = lambda x: np.cos(x)
        M, _ = discretize.tests.setup_mesh("uniformTensorMesh", 32, 1)
        v1 = M.aveN2F * fun(M.gridN)
        v2 = fun(M.faces)
        np.testing.assert_allclose(v1, v2)

    def test_orderN2E(self):
        self.name = "Averaging 1D: N2E"
        fun = lambda x: np.cos(x)
        self.getHere = lambda M: fun(M.gridN)
        self.getThere = lambda M: fun(M.edges)
        self.getAve = lambda M: M.aveN2E
        self.orderTest()

    def test_orderF2CC(self):
        self.name = "Averaging 1D: F2CC"
        fun = lambda x: np.cos(x)
        self.getHere = lambda M: fun(M.faces)
        self.getThere = lambda M: fun(M.gridCC)
        self.getAve = lambda M: M.aveF2CC
        self.orderTest()

    def test_orderF2CCV(self):
        self.name = "Averaging 1D: F2CCV"
        fun = lambda x: np.cos(x)
        self.getHere = lambda M: fun(M.faces)
        self.getThere = lambda M: fun(M.cell_centers)
        self.getAve = lambda M: M.aveF2CCV
        self.orderTest()

    def test_orderCC2F(self):
        self.name = "Averaging 1D: CC2F"
        fun = lambda x: np.cos(x)
        self.getHere = lambda M: fun(M.gridCC)
        self.getThere = lambda M: fun(M.faces)
        self.getAve = lambda M: M.aveCC2F
        self.expectedOrders = 1
        self.orderTest()
        self.expectedOrders = 2

    def test_exactE2CC(self):
        self.name = "Averaging 1D: E2CC"
        fun = lambda x: np.cos(x)
        M, _ = discretize.tests.setup_mesh("uniformTensorMesh", 32, 1)
        v1 = M.aveE2CC @ fun(M.edges)
        v2 = fun(M.gridCC)
        np.testing.assert_allclose(v1, v2)

    def test_exactE2CCV(self):
        self.name = "Averaging 1D: E2CCV"
        fun = lambda x: np.cos(x)
        M, _ = discretize.tests.setup_mesh("uniformTensorMesh", 32, 1)
        v1 = M.aveE2CCV @ fun(M.edges)
        v2 = fun(M.gridCC)
        np.testing.assert_allclose(v1, v2)

    def test_exactCC2E(self):
        self.name = "Averaging 1D: cell_centers_to_edges"
        fun = lambda x: np.cos(x)
        M, _ = discretize.tests.setup_mesh("uniformTensorMesh", 32, 1)
        v1 = M.average_cell_to_edge @ fun(M.edges)
        v2 = fun(M.gridCC)
        np.testing.assert_allclose(v1, v2)

    def test_orderCC2FV(self):
        self.name = "Averaging 1D: CC2FV"
        fun = lambda x: np.cos(x)
        self.getHere = lambda M: fun(M.gridCC)
        self.getThere = lambda M: fun(M.faces)
        self.getAve = lambda M: M.aveCCV2F
        self.expectedOrders = 1
        self.orderTest()
        self.expectedOrders = 2

    def test_orderE2FV(self):
        self.name = "Averaging 1D: E2FV"
        fun = lambda x: np.cos(x)
        self.getHere = lambda M: fun(M.edges)
        self.getThere = lambda M: fun(M.faces)
        self.getAve = lambda M: M.average_edge_to_face
        self.orderTest()


class TestAverating2DSimple(unittest.TestCase):
    def setUp(self):
        hx = gen.random(10)
        hy = gen.random(10)
        self.mesh = discretize.TensorMesh([hx, hy])

    def test_constantEdges(self):
        edge_vec = np.ones(self.mesh.nE)
        assert all(self.mesh.aveE2CC * edge_vec == 1.0)
        assert all(self.mesh.aveE2CCV * edge_vec == 1.0)

    def test_constantFaces(self):
        face_vec = np.ones(self.mesh.nF)
        assert all(self.mesh.aveF2CC * face_vec == 1.0)
        assert all(self.mesh.aveF2CCV * face_vec == 1.0)


class TestAveraging2D(discretize.tests.OrderTest):
    name = "Averaging 2D"
    meshTypes = MESHTYPES
    meshDimension = 2
    meshSizes = [16, 32, 64]

    def getError(self):
        num = self.getAve(self.M) * self.getHere(self.M)
        err = np.linalg.norm((self.getThere(self.M) - num), np.inf)
        return err

    def test_orderN2CC(self):
        self.name = "Averaging 2D: N2CC"
        fun = lambda x, y: (np.cos(x) + np.sin(y))
        self.getHere = lambda M: call2(fun, M.gridN)
        self.getThere = lambda M: call2(fun, M.gridCC)
        self.getAve = lambda M: M.aveN2CC
        self.orderTest()

    def test_orderN2F(self):
        self.name = "Averaging 2D: N2F"
        fun = lambda x, y: (np.cos(x) + np.sin(y))
        self.getHere = lambda M: call2(fun, M.gridN)
        self.getThere = lambda M: np.r_[call2(fun, M.gridFx), call2(fun, M.gridFy)]
        self.getAve = lambda M: M.aveN2F
        self.orderTest()

    def test_orderN2E(self):
        self.name = "Averaging 2D: N2E"
        fun = lambda x, y: (np.cos(x) + np.sin(y))
        self.getHere = lambda M: call2(fun, M.gridN)
        self.getThere = lambda M: np.r_[call2(fun, M.gridEx), call2(fun, M.gridEy)]
        self.getAve = lambda M: M.aveN2E
        self.orderTest()

    def test_orderF2CC(self):
        self.name = "Averaging 2D: F2CC"
        fun = lambda x, y: (np.cos(x) + np.sin(y))
        self.getHere = lambda M: np.r_[call2(fun, M.gridFx), call2(fun, M.gridFy)]
        self.getThere = lambda M: call2(fun, M.gridCC)
        self.getAve = lambda M: M.aveF2CC
        self.orderTest()

    def test_orderF2CCV(self):
        self.name = "Averaging 2D: F2CCV"
        funX = lambda x, y: (np.cos(x) + np.sin(y))
        funY = lambda x, y: (np.cos(y) * np.sin(x))
        self.getHere = lambda M: np.r_[call2(funX, M.gridFx), call2(funY, M.gridFy)]
        self.getThere = lambda M: np.r_[call2(funX, M.gridCC), call2(funY, M.gridCC)]
        self.getAve = lambda M: M.aveF2CCV
        self.orderTest()

    def test_orderCC2F(self):
        self.name = "Averaging 2D: CC2F"
        fun = lambda x, y: (np.cos(x) + np.sin(y))
        self.getHere = lambda M: call2(fun, M.gridCC)
        self.getThere = lambda M: np.r_[call2(fun, M.gridFx), call2(fun, M.gridFy)]
        self.getAve = lambda M: M.aveCC2F
        self.expectedOrders = ORDERS / 2.0
        self.orderTest()
        self.expectedOrders = ORDERS

    def test_orderE2CC(self):
        self.name = "Averaging 2D: E2CC"
        fun = lambda x, y: (np.cos(x) + np.sin(y))
        self.getHere = lambda M: np.r_[call2(fun, M.gridEx), call2(fun, M.gridEy)]
        self.getThere = lambda M: call2(fun, M.gridCC)
        self.getAve = lambda M: M.aveE2CC
        self.orderTest()

    def test_orderE2CCV(self):
        self.name = "Averaging 2D: E2CCV"
        funX = lambda x, y: (np.cos(x) + np.sin(y))
        funY = lambda x, y: (np.cos(y) * np.sin(x))
        self.getHere = lambda M: np.r_[call2(funX, M.gridEx), call2(funY, M.gridEy)]
        self.getThere = lambda M: np.r_[call2(funX, M.gridCC), call2(funY, M.gridCC)]
        self.getAve = lambda M: M.aveE2CCV
        self.orderTest()

    def test_orderCC2E(self):
        self.name = "Averaging 2D: cell_centers_to_edges"
        fun = lambda x, y: (np.cos(x) + np.sin(y))
        self.getHere = lambda M: call2(fun, M.gridCC)
        self.getThere = lambda M: call2(fun, M.edges)
        self.getAve = lambda M: M.average_cell_to_edge
        self.expectedOrders = ORDERS / 2.0
        self.orderTest()
        self.expectedOrders = ORDERS

    def test_orderCC2FV(self):
        self.name = "Averaging 2D: CC2FV"
        funX = lambda x, y: (np.cos(x) + np.sin(y))
        funY = lambda x, y: (np.cos(y) * np.sin(x))
        self.getHere = lambda M: np.r_[call2(funX, M.gridCC), call2(funY, M.gridCC)]
        self.getThere = lambda M: np.r_[call2(funX, M.gridFx), call2(funY, M.gridFy)]
        self.getAve = lambda M: M.aveCCV2F
        self.expectedOrders = ORDERS / 2.0
        self.orderTest()
        self.expectedOrders = ORDERS


class TestAverating3DSimple(unittest.TestCase):
    def setUp(self):
        hx = gen.random(10)
        hy = gen.random(10)
        hz = gen.random(10)
        self.mesh = discretize.TensorMesh([hx, hy, hz])

    def test_constantEdges(self):
        edge_vec = np.ones(self.mesh.nE)
        assert all(np.absolute(self.mesh.aveE2CC * edge_vec - 1.0) < TOL)
        assert all(np.absolute(self.mesh.aveE2CCV * edge_vec - 1.0) < TOL)

    def test_constantFaces(self):
        face_vec = np.ones(self.mesh.nF)
        assert all(np.absolute(self.mesh.aveF2CC * face_vec - 1.0) < TOL)
        assert all(np.absolute(self.mesh.aveF2CCV * face_vec - 1.0) < TOL)


class TestAveraging3D(discretize.tests.OrderTest):
    name = "Averaging 3D"
    meshTypes = MESHTYPES
    meshDimension = 3
    meshSizes = [16, 32, 64]

    def getError(self):
        num = self.getAve(self.M) * self.getHere(self.M)
        err = np.linalg.norm((self.getThere(self.M) - num), np.inf)
        return err

    def test_orderN2CC(self):
        self.name = "Averaging 3D: N2CC"
        fun = lambda x, y, z: (np.cos(x) + np.sin(y) + np.exp(z))
        self.getHere = lambda M: call3(fun, M.gridN)
        self.getThere = lambda M: call3(fun, M.gridCC)
        self.getAve = lambda M: M.aveN2CC
        self.orderTest()

    def test_orderN2F(self):
        self.name = "Averaging 3D: N2F"
        fun = lambda x, y, z: (np.cos(x) + np.sin(y) + np.exp(z))
        self.getHere = lambda M: call3(fun, M.gridN)
        self.getThere = lambda M: np.r_[
            call3(fun, M.gridFx), call3(fun, M.gridFy), call3(fun, M.gridFz)
        ]
        self.getAve = lambda M: M.aveN2F
        self.orderTest()

    def test_orderN2E(self):
        self.name = "Averaging 3D: N2E"
        fun = lambda x, y, z: (np.cos(x) + np.sin(y) + np.exp(z))
        self.getHere = lambda M: call3(fun, M.gridN)
        self.getThere = lambda M: np.r_[
            call3(fun, M.gridEx), call3(fun, M.gridEy), call3(fun, M.gridEz)
        ]
        self.getAve = lambda M: M.aveN2E
        self.orderTest()

    def test_orderF2CC(self):
        self.name = "Averaging 3D: F2CC"
        fun = lambda x, y, z: (np.cos(x) + np.sin(y) + np.exp(z))
        self.getHere = lambda M: np.r_[
            call3(fun, M.gridFx), call3(fun, M.gridFy), call3(fun, M.gridFz)
        ]
        self.getThere = lambda M: call3(fun, M.gridCC)
        self.getAve = lambda M: M.aveF2CC
        self.orderTest()

    def test_orderF2CCV(self):
        self.name = "Averaging 3D: F2CCV"
        funX = lambda x, y, z: (np.cos(x) + np.sin(y) + np.exp(z))
        funY = lambda x, y, z: (np.cos(x) + np.sin(y) * np.exp(z))
        funZ = lambda x, y, z: (np.cos(x) * np.sin(y) + np.exp(z))
        self.getHere = lambda M: np.r_[
            call3(funX, M.gridFx), call3(funY, M.gridFy), call3(funZ, M.gridFz)
        ]
        self.getThere = lambda M: np.r_[
            call3(funX, M.gridCC), call3(funY, M.gridCC), call3(funZ, M.gridCC)
        ]
        self.getAve = lambda M: M.aveF2CCV
        self.orderTest()

    def test_orderE2CC(self):
        self.name = "Averaging 3D: E2CC"
        fun = lambda x, y, z: (np.cos(x) + np.sin(y) + np.exp(z))
        self.getHere = lambda M: np.r_[
            call3(fun, M.gridEx), call3(fun, M.gridEy), call3(fun, M.gridEz)
        ]
        self.getThere = lambda M: call3(fun, M.gridCC)
        self.getAve = lambda M: M.aveE2CC
        self.orderTest()

    def test_orderE2CCV(self):
        self.name = "Averaging 3D: E2CCV"
        funX = lambda x, y, z: (np.cos(x) + np.sin(y) + np.exp(z))
        funY = lambda x, y, z: (np.cos(x) + np.sin(y) * np.exp(z))
        funZ = lambda x, y, z: (np.cos(x) * np.sin(y) + np.exp(z))
        self.getHere = lambda M: np.r_[
            call3(funX, M.gridEx), call3(funY, M.gridEy), call3(funZ, M.gridEz)
        ]
        self.getThere = lambda M: np.r_[
            call3(funX, M.gridCC), call3(funY, M.gridCC), call3(funZ, M.gridCC)
        ]
        self.getAve = lambda M: M.aveE2CCV
        self.expectedOrders = ORDERS
        self.orderTest()

    def test_orderCC2F(self):
        self.name = "Averaging 3D: CC2F"
        fun = lambda x, y, z: (np.cos(x) + np.sin(y) + np.exp(z))
        self.getHere = lambda M: call3(fun, M.gridCC)
        self.getThere = lambda M: np.r_[
            call3(fun, M.gridFx), call3(fun, M.gridFy), call3(fun, M.gridFz)
        ]
        self.getAve = lambda M: M.aveCC2F
        self.expectedOrders = ORDERS / 2.0
        self.orderTest()
        self.expectedOrders = ORDERS

    def test_orderCC2E(self):
        self.name = "Averaging 3D: CC2E"
        fun = lambda x, y, z: (np.cos(x) + np.sin(y) + np.exp(z))
        self.getHere = lambda M: call3(fun, M.gridCC)
        self.getThere = lambda M: call3(fun, M.edges)
        self.getAve = lambda M: M.average_cell_to_edge
        self.expectedOrders = ORDERS / 2.0
        self.orderTest()
        self.expectedOrders = ORDERS

    def test_orderCCV2F(self):
        self.name = "Averaging 3D: CC2FV"
        funX = lambda x, y, z: (np.cos(x) + np.sin(y) + np.exp(z))
        funY = lambda x, y, z: (np.cos(x) + np.sin(y) * np.exp(z))
        funZ = lambda x, y, z: (np.cos(x) * np.sin(y) + np.exp(z))
        self.getHere = lambda M: np.r_[
            call3(funX, M.gridCC), call3(funY, M.gridCC), call3(funZ, M.gridCC)
        ]
        self.getThere = lambda M: np.r_[
            call3(funX, M.gridFx), call3(funY, M.gridFy), call3(funZ, M.gridFz)
        ]
        self.getAve = lambda M: M.aveCCV2F
        self.expectedOrders = ORDERS / 2.0
        self.orderTest()
        self.expectedOrders = ORDERS


class MimeticProperties(unittest.TestCase):
    meshTypes = MESHTYPES
    meshDimension = 3
    meshSize = 64
    tol = 1e-11  # there is still some error due to rounding

    def test_DivCurl(self):
        for meshType in self.meshTypes:
            mesh, _ = discretize.tests.setup_mesh(
                meshType, self.meshSize, self.meshDimension
            )
            v = gen.random(mesh.nE)
            divcurlv = mesh.face_divergence * (mesh.edge_curl * v)
            rel_err = np.linalg.norm(divcurlv) / np.linalg.norm(v)
            passed = rel_err < self.tol
            print(
                "Testing Div * Curl on {} : |Div Curl v| / |v| = {} "
                "... {}".format(meshType, rel_err, "FAIL" if not passed else "ok")
            )

    def test_CurlGrad(self):
        for meshType in self.meshTypes:
            mesh, _ = discretize.tests.setup_mesh(
                meshType, self.meshSize, self.meshDimension
            )
            v = gen.random(mesh.nN)
            curlgradv = mesh.edge_curl * (mesh.nodal_gradient * v)
            rel_err = np.linalg.norm(curlgradv) / np.linalg.norm(v)
            passed = rel_err < self.tol
            print(
                "Testing Curl * Grad on {} : |Curl Grad v| / |v|= {} "
                "... {}".format(meshType, rel_err, "FAIL" if not passed else "ok")
            )


if __name__ == "__main__":
    unittest.main()
