import numpy as np
import unittest
import discretize
from discretize.utils import example_simplex_mesh

class TestOperators2D(discretize.tests.OrderTest):
    meshSizes = [8, 16, 32, 64]
    meshTypes = ['uniform simplex mesh']

    def setupMesh(self, n):
        points, simplices = example_simplex_mesh((n, n))
        self.M = discretize.SimplexMesh(points, simplices)
        return 1.0 / n

    def getError(self):
        mesh = self.M
        if self._test_type == "Curl":
            C = mesh.edge_curl

            ex = lambda x, y: np.cos(y)
            ey = lambda x, y: np.cos(x)
            sol = lambda x, y: -np.sin(x) + np.sin(y)

            Ev = np.c_[ex(*mesh.edges.T), ey(*mesh.edges.T)]
            Ep = mesh.project_edge_vector(Ev)
            ana = sol(*mesh.cell_centers.T)
            test = C @ Ep
            err = np.linalg.norm(mesh.cell_volumes * (test-ana))
        elif self._test_type == "Div":
            D = mesh.face_divergence

            fx = lambda x, y: np.sin(2 * np.pi * x)
            fy = lambda x, y: np.sin(2 * np.pi * y)
            sol = lambda x, y: 2 * np.pi * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))

            f = mesh.project_face_vector(np.c_[fx(*mesh.faces.T), fy(*mesh.faces.T)])

            ana = sol(*mesh.cell_centers.T)
            test = D @ f
            err = np.linalg.norm(mesh.cell_volumes * (test-ana))
        elif self._test_type == "Grad":
            G = mesh.nodal_gradient

            phi = lambda x, y: np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

            dphi_dx = lambda x, y: 2 * np.pi * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
            dphi_dy = lambda x, y: 2 * np.pi * np.cos(2 * np.pi * y) * np.sin(2 * np.pi * x)

            p = phi(*mesh.nodes.T)

            ana = mesh.project_edge_vector(np.c_[dphi_dx(*mesh.edges.T), dphi_dy(*mesh.edges.T)])
            test = G @ p
            err = np.linalg.norm(test-ana, np.inf)
        return err

    def test_curl_order(self):
        self.name = "SimplexMesh curl order test"
        self._test_type = "Curl"
        self.orderTest()

    def test_div_order(self):
        self.name = "SimplexMesh div order test"
        self._test_type = "Div"
        self.orderTest()

    def test_grad_order(self):
        self.name = "SimplexMesh grad order test"
        self._test_type = "Grad"
        self.orderTest()


class TestOperators3D(discretize.tests.OrderTest):
    meshSizes = [8, 16, 32]
    meshTypes = ['uniform simplex mesh']

    def setupMesh(self, n):
        points, simplices = example_simplex_mesh((n, n, n))
        self.M = discretize.SimplexMesh(points, simplices)
        return 1.0 / n

        self.M = discretize.SimplexMesh(points, simplices)
        return 1.0 / n

    def getError(self):
        mesh = self.M
        if self._test_type == "Curl":
            C = mesh.edge_curl

            ex = lambda x, y, z: np.cos(2 * np.pi * y)
            ey = lambda x, y, z: np.cos(2 * np.pi * z)
            ez = lambda x, y, z: np.cos(2 * np.pi * x)

            fx = lambda x, y, z: 2 * np.pi * np.sin(2 * np.pi * z)
            fy = lambda x, y, z: 2 * np.pi * np.sin(2 * np.pi * x)
            fz = lambda x, y, z: 2 * np.pi * np.sin(2 * np.pi * y)

            Ev = np.c_[ex(*mesh.edges.T), ey(*mesh.edges.T), ez(*mesh.edges.T)]
            Ep = mesh.project_edge_vector(Ev)

            Fv = np.c_[fx(*mesh.faces.T), fy(*mesh.faces.T), fz(*mesh.faces.T)]
            ana = mesh.project_face_vector(Fv)
            test = C @ Ep

            err = np.linalg.norm(test-ana)/mesh.n_faces
        elif self._test_type == "Div":
            D = mesh.face_divergence

            fx = lambda x, y, z: np.sin(2 * np.pi * x)
            fy = lambda x, y, z: np.sin(2 * np.pi * y)
            fz = lambda x, y, z: np.sin(2 * np.pi * z)
            sol = lambda x, y, z: (
                2 * np.pi * np.cos(2 * np.pi * x)
                + 2 * np.pi * np.cos(2 * np.pi * y)
                + 2 * np.pi * np.cos(2 * np.pi * z)
            )

            f = mesh.project_face_vector(np.c_[fx(*mesh.faces.T), fy(*mesh.faces.T), fz(*mesh.faces.T)])

            ana = sol(*mesh.cell_centers.T)
            test = D @ f

            err = np.linalg.norm(mesh.cell_volumes * (test-ana))
        elif self._test_type == "Grad":
            G = mesh.nodal_gradient

            phi = lambda x, y, z: (np.cos(x) + np.cos(y) + np.cos(z))
            # i (sin(x)) + j (sin(y)) + k (sin(z))
            ex = lambda x, y, z: -np.sin(x)
            ey = lambda x, y, z: -np.sin(y)
            ez = lambda x, y, z: -np.sin(z)

            p = phi(*mesh.nodes.T)

            ana = mesh.project_edge_vector(
                np.c_[ex(*mesh.edges.T), ey(*mesh.edges.T), ez(*mesh.edges.T)]
            )
            test = G @ p
            err = np.linalg.norm(test-ana, np.inf)
        return err

    def test_curl_order(self):
        self.name = "SimplexMesh curl order test"
        self._test_type = "Curl"
        self.orderTest()

    def test_div_order(self):
        self.name = "SimplexMesh div order test"
        self._test_type = "Div"
        self.orderTest()

    def test_grad_order(self):
        self.name = "SimplexMesh grad order test"
        self._test_type = "Grad"
        self.orderTest()


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
    meshSizes = [8, 16]
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
