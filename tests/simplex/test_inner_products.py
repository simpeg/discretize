import numpy as np
import unittest
import discretize
from discretize.utils import example_simplex_mesh


class TestInnerProducts2D(discretize.tests.OrderTest):
    meshSizes = [8, 16, 32]
    meshTypes = ['uniform simplex mesh']

    def setupMesh(self, n):
        points, simplices = example_simplex_mesh((n, n))
        self.M = discretize.SimplexMesh(points, simplices)
        return 1.0 / n

    def getError(self):

        z = 5  # Because 5 is just such a great number.

        ex = lambda x, y: x ** 2 + y * z
        ey = lambda x, y: (z ** 2) * x + y * z

        sigma1 = lambda x, y: x * y + 1
        sigma2 = lambda x, y: x * z + 2
        sigma3 = lambda x, y: 3 + z * y

        mesh = self.M

        cc = mesh.cell_centers
        if self.sigmaTest == 1:
            sigma = np.c_[sigma1(*cc.T)]
            analytic = 144877.0 / 360   # Found using sympy.
        elif self.sigmaTest == 2:
            sigma = np.r_[sigma1(*cc.T), sigma2(*cc.T)]
            analytic = 189959.0 / 120  # Found using sympy.
        elif self.sigmaTest == 3:
            sigma = np.c_[
                sigma1(*cc.T),
                sigma2(*cc.T),
                sigma3(*cc.T),
            ]
            analytic = 781427.0 / 360  # Found using sympy.

        if self.location == "edges":
            p = mesh.edges
            Ec = np.c_[ex(*p.T), ey(*p.T)]
            E = mesh.project_edge_vector(Ec)
            if self.invProp:
                sigma = discretize.utils.inverse_property_tensor(mesh, sigma)
            A = mesh.get_edge_inner_product(sigma, invert_model=self.invProp)
            numeric = E.T.dot(A.dot(E))

        elif self.location == "faces":
            p = mesh.faces
            Fc = np.c_[ex(*p.T), ey(*p.T)]
            F = mesh.project_face_vector(Fc)

            if self.invProp:
                sigma = discretize.utils.inverse_property_tensor(mesh, sigma)

            A = self.M.get_face_inner_product(sigma, invert_model=self.invProp)
            numeric = F.T.dot(A.dot(F))

        err = np.abs(numeric - analytic)
        return err

    def test_order1_edges(self):
        self.name = "Edge Inner Product - Isotropic"
        self.location = "edges"
        self.sigmaTest = 1
        self.invProp = False
        self.orderTest()

    def test_order1_edges_invProp(self):
        self.name = "Edge Inner Product - Isotropic - invProp"
        self.location = "edges"
        self.sigmaTest = 1
        self.invProp = True
        self.orderTest()

    def test_order2_edges(self):
        self.name = "Edge Inner Product - Anisotropic"
        self.location = "edges"
        self.sigmaTest = 2
        self.invProp = False
        self.orderTest()

    def test_order2_edges_invProp(self):
        self.name = "Edge Inner Product - Anisotropic - invProp"
        self.location = "edges"
        self.sigmaTest = 2
        self.invProp = True
        self.orderTest()

    def test_order3_edges(self):
        self.name = "Edge Inner Product - Full Tensor"
        self.location = "edges"
        self.sigmaTest = 3
        self.invProp = False
        self.orderTest()

    def test_order3_edges_invProp(self):
        self.name = "Edge Inner Product - Full Tensor - invProp"
        self.location = "edges"
        self.sigmaTest = 3
        self.invProp = True
        self.orderTest()

    def test_order1_faces(self):
        self.name = "Face Inner Product - Isotropic"
        self.location = "faces"
        self.sigmaTest = 1
        self.invProp = False
        self.orderTest()

    def test_order1_faces_invProp(self):
        self.name = "Face Inner Product - Isotropic - invProp"
        self.location = "faces"
        self.sigmaTest = 1
        self.invProp = True
        self.orderTest()

    def test_order2_faces(self):
        self.name = "Face Inner Product - Anisotropic"
        self.location = "faces"
        self.sigmaTest = 2
        self.invProp = False
        self.orderTest()

    def test_order2_faces_invProp(self):
        self.name = "Face Inner Product - Anisotropic - invProp"
        self.location = "faces"
        self.sigmaTest = 2
        self.invProp = True
        self.orderTest()

    def test_order3_faces(self):
        self.name = "Face Inner Product - Full Tensor"
        self.location = "faces"
        self.sigmaTest = 3
        self.invProp = False
        self.orderTest()

    def test_order3_faces_invProp(self):
        self.name = "Face Inner Product - Full Tensor - invProp"
        self.location = "faces"
        self.sigmaTest = 3
        self.invProp = True
        self.orderTest()


class TestInnerProducts3D(discretize.tests.OrderTest):
    meshSizes = [8, 16, 32]
    meshTypes = ['uniform simplex mesh']

    def setupMesh(self, n):
        points, simplices = example_simplex_mesh((n, n, n))
        self.M = discretize.SimplexMesh(points, simplices)
        return 1.0 / n

    def getError(self):
        ex = lambda x, y, z: x ** 2 + y * z
        ey = lambda x, y, z: (z ** 2) * x + y * z
        ez = lambda x, y, z: y ** 2 + x * z

        sigma1 = lambda x, y, z: x * y + 1
        sigma2 = lambda x, y, z: x * z + 2
        sigma3 = lambda x, y, z: 3 + z * y
        sigma4 = lambda x, y, z: 0.1 * x * y * z
        sigma5 = lambda x, y, z: 0.2 * x * y
        sigma6 = lambda x, y, z: 0.1 * z

        mesh = self.M

        cc = mesh.cell_centers
        if self.sigmaTest == 1:
            sigma = np.c_[sigma1(*cc.T)]
            analytic = 647.0 / 360  # Found using sympy.
        elif self.sigmaTest == 3:
            sigma = np.r_[sigma1(*cc.T), sigma2(*cc.T), sigma3(*cc.T)]
            analytic = 37.0 / 12  # Found using sympy.
        elif self.sigmaTest == 6:
            sigma = np.c_[
                sigma1(*cc.T),
                sigma2(*cc.T),
                sigma3(*cc.T),
                sigma4(*cc.T),
                sigma5(*cc.T),
                sigma6(*cc.T),
            ]
            analytic = 69881.0 / 21600  # Found using sympy.

        if self.location == "edges":
            cart = lambda g: np.c_[ex(*g.T), ey(*g.T), ez(*g.T)]
            Ec = cart(mesh.edges)
            E = mesh.project_edge_vector(Ec)
            if self.invProp:
                sigma = discretize.utils.inverse_property_tensor(mesh, sigma)
            A = mesh.get_edge_inner_product(sigma, invert_model=self.invProp)
            numeric = E.T.dot(A.dot(E))

        elif self.location == "faces":
            cart = lambda g: np.c_[ex(*g.T), ey(*g.T), ez(*g.T)]
            Fc = cart(mesh.faces)
            F = mesh.project_face_vector(Fc)

            if self.invProp:
                sigma = discretize.utils.inverse_property_tensor(mesh, sigma)

            A = self.M.get_face_inner_product(sigma, invert_model=self.invProp)
            numeric = F.T.dot(A.dot(F))

        err = np.abs(numeric - analytic)
        return err

    def test_order1_edges(self):
        self.name = "Edge Inner Product - Isotropic"
        self.location = "edges"
        self.sigmaTest = 1
        self.invProp = False
        self.orderTest()

    def test_order1_edges_invProp(self):
        self.name = "Edge Inner Product - Isotropic - invProp"
        self.location = "edges"
        self.sigmaTest = 1
        self.invProp = True
        self.orderTest()

    def test_order3_edges(self):
        self.name = "Edge Inner Product - Anisotropic"
        self.location = "edges"
        self.sigmaTest = 3
        self.invProp = False
        self.orderTest()

    def test_order3_edges_invProp(self):
        self.name = "Edge Inner Product - Anisotropic - invProp"
        self.location = "edges"
        self.sigmaTest = 3
        self.invProp = True
        self.orderTest()

    def test_order6_edges(self):
        self.name = "Edge Inner Product - Full Tensor"
        self.location = "edges"
        self.sigmaTest = 6
        self.invProp = False
        self.orderTest()

    def test_order6_edges_invProp(self):
        self.name = "Edge Inner Product - Full Tensor - invProp"
        self.location = "edges"
        self.sigmaTest = 6
        self.invProp = True
        self.orderTest()

    def test_order1_faces(self):
        self.name = "Face Inner Product - Isotropic"
        self.location = "faces"
        self.sigmaTest = 1
        self.invProp = False
        self.orderTest()

    def test_order1_faces_invProp(self):
        self.name = "Face Inner Product - Isotropic - invProp"
        self.location = "faces"
        self.sigmaTest = 1
        self.invProp = True
        self.orderTest()

    def test_order3_faces(self):
        self.name = "Face Inner Product - Anisotropic"
        self.location = "faces"
        self.sigmaTest = 3
        self.invProp = False
        self.orderTest()

    def test_order3_faces_invProp(self):
        self.name = "Face Inner Product - Anisotropic - invProp"
        self.location = "faces"
        self.sigmaTest = 3
        self.invProp = True
        self.orderTest()

    def test_order6_faces(self):
        self.name = "Face Inner Product - Full Tensor"
        self.location = "faces"
        self.sigmaTest = 6
        self.invProp = False
        self.orderTest()

    def test_order6_faces_invProp(self):
        self.name = "Face Inner Product - Full Tensor - invProp"
        self.location = "faces"
        self.sigmaTest = 6
        self.invProp = True
        self.orderTest()


class TestInnerProductsDerivs(unittest.TestCase):
    def doTestFace(self, h, rep):
        nodes, simplices = example_simplex_mesh(h)
        mesh = discretize.SimplexMesh(nodes, simplices)
        v = np.random.rand(mesh.n_faces)
        sig = np.random.rand(1) if rep == 0 else np.random.rand(mesh.nC * rep)

        def fun(sig):
            M = mesh.get_face_inner_product(sig)
            Md = mesh.get_face_inner_product_deriv(
                sig
            )
            return M * v, Md(v)

        print("Face", rep)
        return discretize.tests.check_derivative(fun, sig, num=5, plotIt=False)

    def doTestEdge(self, h, rep):
        nodes, simplices = example_simplex_mesh(h)
        mesh = discretize.SimplexMesh(nodes, simplices)
        v = np.random.rand(mesh.n_edges)
        sig = np.random.rand(1) if rep == 0 else np.random.rand(mesh.nC * rep)

        def fun(sig):
            M = mesh.get_edge_inner_product(sig)
            Md = mesh.get_edge_inner_product_deriv(
                sig
            )
            return M * v, Md(v)

        print("Edge", rep)
        return discretize.tests.check_derivative(fun, sig, num=5, plotIt=False)

    def test_FaceIP_2D_float(self):
        self.assertTrue(self.doTestFace([10, 4], 0))

    def test_FaceIP_3D_float(self):
        self.assertTrue(self.doTestFace([10, 4, 5], 0))

    def test_FaceIP_2D_isotropic(self):
        self.assertTrue(self.doTestFace([10, 4], 1))

    def test_FaceIP_3D_isotropic(self):
        self.assertTrue(self.doTestFace([10, 4, 5], 1))

    def test_FaceIP_2D_anisotropic(self):
        self.assertTrue(self.doTestFace([10, 4], 2))

    def test_FaceIP_3D_anisotropic(self):
        self.assertTrue(self.doTestFace([10, 4, 5], 3))

    def test_FaceIP_2D_tensor(self):
        self.assertTrue(self.doTestFace([10, 4], 3))

    def test_FaceIP_3D_tensor(self):
        self.assertTrue(self.doTestFace([10, 4, 5], 6))

    def test_EdgeIP_2D_float(self):
        self.assertTrue(self.doTestEdge([10, 4], 0))

    def test_EdgeIP_3D_float(self):
        self.assertTrue(self.doTestEdge([10, 4, 5], 0))

    def test_EdgeIP_2D_isotropic(self):
        self.assertTrue(self.doTestEdge([10, 4], 1))

    def test_EdgeIP_3D_isotropic(self):
        self.assertTrue(self.doTestEdge([10, 4, 5], 1))

    def test_EdgeIP_2D_anisotropic(self):
        self.assertTrue(self.doTestEdge([10, 4], 2))

    def test_EdgeIP_3D_anisotropic(self):
        self.assertTrue(self.doTestEdge([10, 4, 5], 3))

    def test_EdgeIP_2D_tensor(self):
        self.assertTrue(self.doTestEdge([10, 4], 3))

    def test_EdgeIP_3D_tensor(self):
        self.assertTrue(self.doTestEdge([10, 4, 5], 6))
