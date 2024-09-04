import numpy as np
import unittest
import discretize
import scipy.sparse as sp
from discretize.utils import example_simplex_mesh

rng = np.random.default_rng(4421)


def u(*args):
    if len(args) == 1:
        x = args[0]
        return x**3
    if len(args) == 2:
        x, y = args
        return x**3 + y**2
    x, y, z = args
    return x**3 + y**2 + z**4


def v(*args):
    if len(args) == 1:
        x = args[0]
        return 2 * x**2
    if len(args) == 2:
        x, y = args
        return np.c_[2 * x**2, 3 * y**3]
    x, y, z = args
    return np.c_[2 * x**2, 3 * y**3, -4 * z**2]


def w(*args):
    if len(args) == 2:
        x, y = args
        return np.c_[(y - 2) ** 2, (x + 2) ** 2]
    x, y, z = args
    return np.c_[(y - 2) ** 2 + z**2, (x + 2) ** 2 - (z - 4) ** 2, y**2 - x**2]


class TestInnerProducts2D(discretize.tests.OrderTest):
    meshSizes = [8, 16, 32]
    meshTypes = ["uniform simplex mesh"]

    def setupMesh(self, n):
        points, simplices = example_simplex_mesh((n, n))
        self.M = discretize.SimplexMesh(points, simplices)
        return 1.0 / n

    def getError(self):
        z = 5  # Because 5 is just such a great number.

        ex = lambda x, y: x**2 + y * z
        ey = lambda x, y: (z**2) * x + y * z

        sigma1 = lambda x, y: x * y + 1
        sigma2 = lambda x, y: x * z + 2
        sigma3 = lambda x, y: 3 + z * y

        mesh = self.M

        cc = mesh.cell_centers
        if self.sigmaTest == 1:
            sigma = np.c_[sigma1(*cc.T)]
            analytic = 144877.0 / 360  # Found using sympy.
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
            if self.invert_model:
                sigma = discretize.utils.inverse_property_tensor(mesh, sigma)
            A = mesh.get_edge_inner_product(sigma, invert_model=self.invert_model)
            numeric = E.T.dot(A.dot(E))

        elif self.location == "faces":
            p = mesh.faces
            Fc = np.c_[ex(*p.T), ey(*p.T)]
            F = mesh.project_face_vector(Fc)

            if self.invert_model:
                sigma = discretize.utils.inverse_property_tensor(mesh, sigma)

            A = self.M.get_face_inner_product(sigma, invert_model=self.invert_model)
            numeric = F.T.dot(A.dot(F))

        err = np.abs(numeric - analytic)
        return err

    def test_order1_edges(self):
        self.name = "Edge Inner Product - Isotropic"
        self.location = "edges"
        self.sigmaTest = 1
        self.invert_model = False
        self.orderTest()

    def test_order1_edges_invert_model(self):
        self.name = "Edge Inner Product - Isotropic - invert_model"
        self.location = "edges"
        self.sigmaTest = 1
        self.invert_model = True
        self.orderTest()

    def test_order2_edges(self):
        self.name = "Edge Inner Product - Anisotropic"
        self.location = "edges"
        self.sigmaTest = 2
        self.invert_model = False
        self.orderTest()

    def test_order2_edges_invert_model(self):
        self.name = "Edge Inner Product - Anisotropic - invert_model"
        self.location = "edges"
        self.sigmaTest = 2
        self.invert_model = True
        self.orderTest()

    def test_order3_edges(self):
        self.name = "Edge Inner Product - Full Tensor"
        self.location = "edges"
        self.sigmaTest = 3
        self.invert_model = False
        self.orderTest()

    def test_order3_edges_invert_model(self):
        self.name = "Edge Inner Product - Full Tensor - invert_model"
        self.location = "edges"
        self.sigmaTest = 3
        self.invert_model = True
        self.orderTest()

    def test_order1_faces(self):
        self.name = "Face Inner Product - Isotropic"
        self.location = "faces"
        self.sigmaTest = 1
        self.invert_model = False
        self.orderTest()

    def test_order1_faces_invert_model(self):
        self.name = "Face Inner Product - Isotropic - invert_model"
        self.location = "faces"
        self.sigmaTest = 1
        self.invert_model = True
        self.orderTest()

    def test_order2_faces(self):
        self.name = "Face Inner Product - Anisotropic"
        self.location = "faces"
        self.sigmaTest = 2
        self.invert_model = False
        self.orderTest()

    def test_order2_faces_invert_model(self):
        self.name = "Face Inner Product - Anisotropic - invert_model"
        self.location = "faces"
        self.sigmaTest = 2
        self.invert_model = True
        self.orderTest()

    def test_order3_faces(self):
        self.name = "Face Inner Product - Full Tensor"
        self.location = "faces"
        self.sigmaTest = 3
        self.invert_model = False
        self.orderTest()

    def test_order3_faces_invert_model(self):
        self.name = "Face Inner Product - Full Tensor - invert_model"
        self.location = "faces"
        self.sigmaTest = 3
        self.invert_model = True
        self.orderTest()


class TestInnerProducts3D(discretize.tests.OrderTest):
    meshSizes = [8, 16, 32]
    meshTypes = ["uniform simplex mesh"]

    def setupMesh(self, n):
        points, simplices = example_simplex_mesh((n, n, n))
        self.M = discretize.SimplexMesh(points, simplices)
        return 1.0 / n

    def getError(self):
        ex = lambda x, y, z: x**2 + y * z
        ey = lambda x, y, z: (z**2) * x + y * z
        ez = lambda x, y, z: y**2 + x * z

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
            if self.invert_model:
                sigma = discretize.utils.inverse_property_tensor(mesh, sigma)
            A = mesh.get_edge_inner_product(sigma, invert_model=self.invert_model)
            numeric = E.T.dot(A.dot(E))

        elif self.location == "faces":
            cart = lambda g: np.c_[ex(*g.T), ey(*g.T), ez(*g.T)]
            Fc = cart(mesh.faces)
            F = mesh.project_face_vector(Fc)

            if self.invert_model:
                sigma = discretize.utils.inverse_property_tensor(mesh, sigma)

            A = self.M.get_face_inner_product(sigma, invert_model=self.invert_model)
            numeric = F.T.dot(A.dot(F))

        err = np.abs(numeric - analytic)
        return err

    def test_order1_edges(self):
        self.name = "Edge Inner Product - Isotropic"
        self.location = "edges"
        self.sigmaTest = 1
        self.invert_model = False
        self.orderTest()

    def test_order1_edges_invert_model(self):
        self.name = "Edge Inner Product - Isotropic - invert_model"
        self.location = "edges"
        self.sigmaTest = 1
        self.invert_model = True
        self.orderTest()

    def test_order3_edges(self):
        self.name = "Edge Inner Product - Anisotropic"
        self.location = "edges"
        self.sigmaTest = 3
        self.invert_model = False
        self.orderTest()

    def test_order3_edges_invert_model(self):
        self.name = "Edge Inner Product - Anisotropic - invert_model"
        self.location = "edges"
        self.sigmaTest = 3
        self.invert_model = True
        self.orderTest()

    def test_order6_edges(self):
        self.name = "Edge Inner Product - Full Tensor"
        self.location = "edges"
        self.sigmaTest = 6
        self.invert_model = False
        self.orderTest()

    def test_order6_edges_invert_model(self):
        self.name = "Edge Inner Product - Full Tensor - invert_model"
        self.location = "edges"
        self.sigmaTest = 6
        self.invert_model = True
        self.orderTest()

    def test_order1_faces(self):
        self.name = "Face Inner Product - Isotropic"
        self.location = "faces"
        self.sigmaTest = 1
        self.invert_model = False
        self.orderTest()

    def test_order1_faces_invert_model(self):
        self.name = "Face Inner Product - Isotropic - invert_model"
        self.location = "faces"
        self.sigmaTest = 1
        self.invert_model = True
        self.orderTest()

    def test_order3_faces(self):
        self.name = "Face Inner Product - Anisotropic"
        self.location = "faces"
        self.sigmaTest = 3
        self.invert_model = False
        self.orderTest()

    def test_order3_faces_invert_model(self):
        self.name = "Face Inner Product - Anisotropic - invert_model"
        self.location = "faces"
        self.sigmaTest = 3
        self.invert_model = True
        self.orderTest()

    def test_order6_faces(self):
        self.name = "Face Inner Product - Full Tensor"
        self.location = "faces"
        self.sigmaTest = 6
        self.invert_model = False
        self.orderTest()

    def test_order6_faces_invert_model(self):
        self.name = "Face Inner Product - Full Tensor - invert_model"
        self.location = "faces"
        self.sigmaTest = 6
        self.invert_model = True
        self.orderTest()


class TestInnerProductsDerivs(unittest.TestCase):
    def doTestFace(self, h, rep):
        nodes, simplices = example_simplex_mesh(h)
        mesh = discretize.SimplexMesh(nodes, simplices)
        v = rng.random(mesh.n_faces)
        sig = rng.random(1) if rep == 0 else rng.random(mesh.nC * rep)

        def fun(sig):
            M = mesh.get_face_inner_product(sig)
            Md = mesh.get_face_inner_product_deriv(sig)
            return M * v, Md(v)

        print("Face", rep)
        return discretize.tests.check_derivative(
            fun, sig, num=5, plotIt=False, random_seed=5352
        )

    def doTestEdge(self, h, rep):
        nodes, simplices = example_simplex_mesh(h)
        mesh = discretize.SimplexMesh(nodes, simplices)
        v = rng.random(mesh.n_edges)
        sig = rng.random(1) if rep == 0 else rng.random(mesh.nC * rep)

        def fun(sig):
            M = mesh.get_edge_inner_product(sig)
            Md = mesh.get_edge_inner_product_deriv(sig)
            return M * v, Md(v)

        print("Edge", rep)
        return discretize.tests.check_derivative(
            fun, sig, num=5, plotIt=False, random_seed=532
        )

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


class Test2DBoundaryIntegral(discretize.tests.OrderTest):
    meshSizes = [8, 16, 32]
    meshTypes = ["uniform simplex mesh"]

    def setupMesh(self, n):
        points, simplices = example_simplex_mesh((n, n))
        self.M = discretize.SimplexMesh(points, simplices)
        return 1.0 / n

    def getError(self):
        mesh = self.M
        if self.myTest == "cell_grad":
            # Functions:
            u_cc = u(*mesh.cell_centers.T)
            v_f = mesh.project_face_vector(v(*mesh.faces.T))
            u_bf = u(*mesh.boundary_faces.T)

            D = mesh.face_divergence
            M_c = sp.diags(mesh.cell_volumes)
            M_bf = mesh.boundary_face_scalar_integral

            discrete_val = -(v_f.T @ D.T) @ M_c @ u_cc + v_f.T @ (M_bf @ u_bf)
            true_val = 12 / 5
        elif self.myTest == "edge_div":
            u_n = u(*mesh.nodes.T)
            v_e = mesh.project_edge_vector(v(*mesh.edges.T))
            v_bn = v(*mesh.boundary_nodes.T).reshape(-1, order="F")

            M_e = mesh.get_edge_inner_product()
            G = mesh.nodal_gradient
            M_bn = mesh.boundary_node_vector_integral

            discrete_val = -(u_n.T @ G.T) @ M_e @ v_e + u_n.T @ (M_bn @ v_bn)
            true_val = 241 / 60
        elif self.myTest == "face_curl":
            w_e = mesh.project_edge_vector(w(*mesh.edges.T))
            u_c = u(*mesh.cell_centers.T)
            u_be = u(*mesh.boundary_edges.T)

            M_c = sp.diags(mesh.cell_volumes)
            Curl = mesh.edge_curl
            M_be = mesh.boundary_edge_vector_integral

            discrete_val = (w_e.T @ Curl.T) @ M_c @ u_c - w_e.T @ (M_be @ u_be)
            true_val = -173 / 30

        return np.abs(discrete_val - true_val)

    def test_orderWeakCellGradIntegral(self):
        self.name = "2D - weak cell gradient integral w/boundary"
        self.myTest = "cell_grad"
        self.orderTest()

    def test_orderWeakEdgeDivIntegral(self):
        self.name = "2D - weak edge divergence integral w/boundary"
        self.myTest = "edge_div"
        self.orderTest()

    def test_orderWeakFaceCurlIntegral(self):
        self.name = "2D - weak face curl integral w/boundary"
        self.myTest = "face_curl"
        self.orderTest()


class Test3DBoundaryIntegral(discretize.tests.OrderTest):
    meshSizes = [8, 16, 32]
    meshTypes = ["uniform simplex mesh"]

    def setupMesh(self, n):
        points, simplices = example_simplex_mesh((n, n, n))
        self.M = discretize.SimplexMesh(points, simplices)
        return 1.0 / n

    def getError(self):
        mesh = self.M
        if self.myTest == "cell_grad":
            # Functions:
            u_cc = u(*mesh.cell_centers.T)
            v_f = mesh.project_face_vector(v(*mesh.faces.T))
            u_bf = u(*mesh.boundary_faces.T)

            D = mesh.face_divergence
            M_c = sp.diags(mesh.cell_volumes)
            M_bf = mesh.boundary_face_scalar_integral

            discrete_val = -(v_f.T @ D.T) @ M_c @ u_cc + v_f.T @ (M_bf @ u_bf)

            true_val = -4 / 15
        elif self.myTest == "edge_div":
            u_n = u(*mesh.nodes.T)
            v_e = mesh.project_edge_vector(v(*mesh.edges.T))
            v_bn = v(*mesh.boundary_nodes.T).reshape(-1, order="F")

            M_e = mesh.get_edge_inner_product()
            G = mesh.nodal_gradient
            M_bn = mesh.boundary_node_vector_integral

            discrete_val = -(u_n.T @ G.T) @ M_e @ v_e + u_n.T @ (M_bn @ v_bn)
            true_val = 27 / 20

        elif self.myTest == "face_curl":
            w_f = mesh.project_face_vector(w(*mesh.faces.T))
            v_e = mesh.project_edge_vector(v(*mesh.edges.T))
            w_be = w(*mesh.boundary_edges.T).reshape(-1, order="F")

            M_f = mesh.get_face_inner_product()
            Curl = mesh.edge_curl
            M_be = mesh.boundary_edge_vector_integral

            discrete_val = (v_e.T @ Curl.T) @ M_f @ w_f - v_e.T @ (M_be @ w_be)
            true_val = -79 / 6

        return np.abs(discrete_val - true_val)

    def test_orderWeakCellGradIntegral(self):
        self.name = "3D - weak cell gradient integral w/boundary"
        self.myTest = "cell_grad"
        self.orderTest()

    def test_orderWeakEdgeDivIntegral(self):
        self.name = "3D - weak edge divergence integral w/boundary"
        self.myTest = "edge_div"
        self.orderTest()

    def test_orderWeakFaceCurlIntegral(self):
        self.name = "3D - weak face curl integral w/boundary"
        self.myTest = "face_curl"
        self.orderTest()


class TestBadModels(unittest.TestCase):
    def setUp(self):
        n = 8
        points, simplices = example_simplex_mesh((n, n))
        self.mesh = discretize.SimplexMesh(points, simplices)

    def test_bad_model_size(self):
        mesh = self.mesh
        bad_model = rng.random((mesh.n_cells, 5))
        with self.assertRaises(ValueError):
            mesh.get_face_inner_product(bad_model)
        with self.assertRaises(ValueError):
            mesh.get_face_inner_product_deriv(bad_model)

    def test_cant_invert(self):
        mesh = self.mesh
        good_model = rng.random(mesh.n_cells)
        with self.assertRaises(NotImplementedError):
            mesh.get_face_inner_product(good_model, invert_matrix=True)
        with self.assertRaises(NotImplementedError):
            mesh.get_face_inner_product_deriv(good_model, invert_matrix=True)
        with self.assertRaises(NotImplementedError):
            mesh.get_face_inner_product_deriv(good_model, invert_model=True)
        with self.assertRaises(NotImplementedError):
            mesh.get_edge_inner_product(good_model, invert_matrix=True)
        with self.assertRaises(NotImplementedError):
            mesh.get_edge_inner_product_deriv(good_model, invert_matrix=True)
        with self.assertRaises(NotImplementedError):
            mesh.get_edge_inner_product_deriv(good_model, invert_model=True)
