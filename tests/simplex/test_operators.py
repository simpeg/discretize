import numpy as np
import pytest

import discretize
from discretize import SimplexMesh
from discretize.utils import example_simplex_mesh


class TestOperators2D(discretize.tests.OrderTest):
    meshSizes = [8, 16, 32, 64]
    meshTypes = ["uniform simplex mesh"]

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
            err = np.linalg.norm(mesh.cell_volumes * (test - ana))
        elif self._test_type == "Div":
            D = mesh.face_divergence

            fx = lambda x, y: np.sin(2 * np.pi * x)
            fy = lambda x, y: np.sin(2 * np.pi * y)
            sol = (
                lambda x, y: 2 * np.pi * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))
            )

            f = mesh.project_face_vector(np.c_[fx(*mesh.faces.T), fy(*mesh.faces.T)])

            ana = sol(*mesh.cell_centers.T)
            test = D @ f
            err = np.linalg.norm(mesh.cell_volumes * (test - ana))
        elif self._test_type == "Grad":
            G = mesh.nodal_gradient

            phi = lambda x, y: np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

            dphi_dx = (
                lambda x, y: 2 * np.pi * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
            )
            dphi_dy = (
                lambda x, y: 2 * np.pi * np.cos(2 * np.pi * y) * np.sin(2 * np.pi * x)
            )

            p = phi(*mesh.nodes.T)

            ana = mesh.project_edge_vector(
                np.c_[dphi_dx(*mesh.edges.T), dphi_dy(*mesh.edges.T)]
            )
            test = G @ p
            err = np.linalg.norm(test - ana, np.inf)
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

    def test_cell_gradient_stencil(self):
        n = 4
        points, simplices = example_simplex_mesh((n, n))
        mesh = discretize.SimplexMesh(points, simplices)

        G_sten = mesh.stencil_cell_gradient
        mod = np.zeros(mesh.n_cells)

        test_cell = 21
        mod[test_cell] = 1.0

        diff = G_sten @ mod
        np.testing.assert_equal(
            np.where(diff != 0.0)[0], np.sort(mesh._simplex_faces[21])
        )
        np.testing.assert_equal(diff[diff != 0.0], [-1, 1, -1])


class TestOperators3D(discretize.tests.OrderTest):
    meshSizes = [8, 16, 32]
    meshTypes = ["uniform simplex mesh"]

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

            err = np.linalg.norm(test - ana) / mesh.n_faces
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

            f = mesh.project_face_vector(
                np.c_[fx(*mesh.faces.T), fy(*mesh.faces.T), fz(*mesh.faces.T)]
            )

            ana = sol(*mesh.cell_centers.T)
            test = D @ f

            err = np.linalg.norm(mesh.cell_volumes * (test - ana))
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
            err = np.linalg.norm(test - ana, np.inf)
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


@pytest.mark.parametrize("i_type", ["E", "F"])
def test_simplex_projection_caching(i_type):
    n = 5
    mesh = SimplexMesh(*example_simplex_mesh((n, n)))
    P1 = mesh._SimplexMesh__get_inner_product_projection_matrices(
        i_type, with_volume=False, return_pointers=False
    )
    P2 = mesh._SimplexMesh__get_inner_product_projection_matrices(
        i_type, with_volume=True, return_pointers=False
    )
    assert P1 is not P2
