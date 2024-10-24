import unittest
import numpy as np
from discretize import utils

tol = 1e-15

rng = np.random.default_rng(523)


class coorutilsTest(unittest.TestCase):
    def test_rotation_matrix_from_normals(self):
        v0 = rng.random(3)
        v0 *= 1.0 / np.linalg.norm(v0)

        np.random.seed(5)
        v1 = rng.random(3)
        v1 *= 1.0 / np.linalg.norm(v1)

        Rf = utils.rotation_matrix_from_normals(v0, v1)
        Ri = utils.rotation_matrix_from_normals(v1, v0)

        self.assertTrue(np.linalg.norm(utils.mkvc(Rf.dot(v0) - v1)) < tol)
        self.assertTrue(np.linalg.norm(utils.mkvc(Ri.dot(v1) - v0)) < tol)

    def test_rotate_points_from_normals(self):
        v0 = rng.random(3)
        v0 *= 1.0 / np.linalg.norm(v0)

        np.random.seed(15)
        v1 = rng.random(3)
        v1 *= 1.0 / np.linalg.norm(v1)

        v2 = utils.mkvc(utils.rotate_points_from_normals(utils.mkvc(v0, 2).T, v0, v1))

        self.assertTrue(np.linalg.norm(v2 - v1) < tol)

    def test_rotateMatrixFromNormals(self):
        n0 = rng.random(3)
        n0 *= 1.0 / np.linalg.norm(n0)

        np.random.seed(25)
        n1 = rng.random(3)
        n1 *= 1.0 / np.linalg.norm(n1)

        np.random.seed(30)
        scale = rng.random((100, 1))
        XYZ0 = scale * n0
        XYZ1 = scale * n1

        XYZ2 = utils.rotate_points_from_normals(XYZ0, n0, n1)
        self.assertTrue(
            np.linalg.norm(utils.mkvc(XYZ1) - utils.mkvc(XYZ2))
            / np.linalg.norm(utils.mkvc(XYZ1))
            < tol
        )

    def test_rotate_vec_cyl2cart(self):
        vec = np.r_[1.0, 0, 0].reshape(1, 3)
        grid = np.r_[1.0, np.pi / 4, 0].reshape(1, 3)
        self.assertTrue(
            np.allclose(utils.cyl2cart(grid, vec), np.sqrt(2) / 2 * np.r_[1, 1, 0])
        )
        self.assertTrue(
            np.allclose(utils.cyl2cart(grid), np.sqrt(2) / 2 * np.r_[1, 1, 0])
        )

        vec = np.r_[0, 1, 2].reshape(1, 3)
        grid = np.r_[1, np.pi / 4, 0].reshape(1, 3)
        self.assertTrue(
            np.allclose(
                utils.cyl2cart(grid, vec), np.r_[-np.sqrt(2) / 2, np.sqrt(2) / 2, 2]
            )
        )

        vec = np.r_[1.0, 0]
        grid = np.r_[1.0, np.pi / 4].reshape(1, 2)
        self.assertTrue(
            np.allclose(utils.cyl2cart(grid, vec), np.sqrt(2) / 2 * np.r_[1, 1])
        )


if __name__ == "__main__":
    unittest.main()
