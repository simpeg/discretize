import numpy as np
import unittest
import pytest

from discretize import TensorMesh, CurvilinearMesh
from discretize.utils import ndgrid
from discretize.tests import setup_mesh
from discretize.tests import check_derivative


class BasicCurvTests(unittest.TestCase):
    def setUp(self):
        a = np.array([1, 1, 1])
        b = np.array([1, 2])
        c = np.array([1, 4])

        def gridIt(h):
            return [np.cumsum(np.r_[0, x]) for x in h]

        X, Y = ndgrid(gridIt([a, b]), vector=False)
        self.TM2 = TensorMesh([a, b])
        self.Curv2 = CurvilinearMesh([X, Y])
        X, Y, Z = ndgrid(gridIt([a, b, c]), vector=False)
        self.TM3 = TensorMesh([a, b, c])
        self.Curv3 = CurvilinearMesh([X, Y, Z])

    def test_area_3D(self):
        test_area = np.array(
            [
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                4,
                4,
                4,
                4,
                8,
                8,
                8,
                8,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                1,
                1,
                1,
                2,
                2,
                2,
                1,
                1,
                1,
                2,
                2,
                2,
                1,
                1,
                1,
                2,
                2,
                2,
            ]
        )
        self.assertTrue(np.all(self.Curv3.face_areas == test_area))

    def test_vol_3D(self):
        test_vol = np.array([1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8])
        np.testing.assert_almost_equal(self.Curv3.cell_volumes, test_vol)
        self.assertTrue(True)  # Pass if you get past the assertion.

    def test_vol_2D(self):
        test_vol = np.array([1, 1, 1, 2, 2, 2])
        t1 = np.all(self.Curv2.cell_volumes == test_vol)
        self.assertTrue(t1)

    def test_edge_3D(self):
        test_edge = np.array(
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
            ]
        )
        t1 = np.all(self.Curv3.edge_lengths == test_edge)
        self.assertTrue(t1)

    def test_edge_2D(self):
        test_edge = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2])
        t1 = np.all(self.Curv2.edge_lengths == test_edge)
        self.assertTrue(t1)

    def test_tangents(self):
        T = self.Curv2.edge_tangents
        self.assertTrue(
            np.all(self.Curv2.reshape(T, "E", "Ex", "V")[0] == np.ones(self.Curv2.nEx))
        )
        self.assertTrue(
            np.all(self.Curv2.reshape(T, "E", "Ex", "V")[1] == np.zeros(self.Curv2.nEx))
        )
        self.assertTrue(
            np.all(self.Curv2.reshape(T, "E", "Ey", "V")[0] == np.zeros(self.Curv2.nEy))
        )
        self.assertTrue(
            np.all(self.Curv2.reshape(T, "E", "Ey", "V")[1] == np.ones(self.Curv2.nEy))
        )

        T = self.Curv3.edge_tangents
        self.assertTrue(
            np.all(self.Curv3.reshape(T, "E", "Ex", "V")[0] == np.ones(self.Curv3.nEx))
        )
        self.assertTrue(
            np.all(self.Curv3.reshape(T, "E", "Ex", "V")[1] == np.zeros(self.Curv3.nEx))
        )
        self.assertTrue(
            np.all(self.Curv3.reshape(T, "E", "Ex", "V")[2] == np.zeros(self.Curv3.nEx))
        )

        self.assertTrue(
            np.all(self.Curv3.reshape(T, "E", "Ey", "V")[0] == np.zeros(self.Curv3.nEy))
        )
        self.assertTrue(
            np.all(self.Curv3.reshape(T, "E", "Ey", "V")[1] == np.ones(self.Curv3.nEy))
        )
        self.assertTrue(
            np.all(self.Curv3.reshape(T, "E", "Ey", "V")[2] == np.zeros(self.Curv3.nEy))
        )

        self.assertTrue(
            np.all(self.Curv3.reshape(T, "E", "Ez", "V")[0] == np.zeros(self.Curv3.nEz))
        )
        self.assertTrue(
            np.all(self.Curv3.reshape(T, "E", "Ez", "V")[1] == np.zeros(self.Curv3.nEz))
        )
        self.assertTrue(
            np.all(self.Curv3.reshape(T, "E", "Ez", "V")[2] == np.ones(self.Curv3.nEz))
        )

    def test_normals(self):
        N = self.Curv2.face_normals
        self.assertTrue(
            np.all(self.Curv2.reshape(N, "F", "Fx", "V")[0] == np.ones(self.Curv2.nFx))
        )
        self.assertTrue(
            np.all(self.Curv2.reshape(N, "F", "Fx", "V")[1] == np.zeros(self.Curv2.nFx))
        )
        self.assertTrue(
            np.all(self.Curv2.reshape(N, "F", "Fy", "V")[0] == np.zeros(self.Curv2.nFy))
        )
        self.assertTrue(
            np.all(self.Curv2.reshape(N, "F", "Fy", "V")[1] == np.ones(self.Curv2.nFy))
        )

        N = self.Curv3.face_normals
        self.assertTrue(
            np.all(self.Curv3.reshape(N, "F", "Fx", "V")[0] == np.ones(self.Curv3.nFx))
        )
        self.assertTrue(
            np.all(self.Curv3.reshape(N, "F", "Fx", "V")[1] == np.zeros(self.Curv3.nFx))
        )
        self.assertTrue(
            np.all(self.Curv3.reshape(N, "F", "Fx", "V")[2] == np.zeros(self.Curv3.nFx))
        )

        self.assertTrue(
            np.all(self.Curv3.reshape(N, "F", "Fy", "V")[0] == np.zeros(self.Curv3.nFy))
        )
        self.assertTrue(
            np.all(self.Curv3.reshape(N, "F", "Fy", "V")[1] == np.ones(self.Curv3.nFy))
        )
        self.assertTrue(
            np.all(self.Curv3.reshape(N, "F", "Fy", "V")[2] == np.zeros(self.Curv3.nFy))
        )

        self.assertTrue(
            np.all(self.Curv3.reshape(N, "F", "Fz", "V")[0] == np.zeros(self.Curv3.nFz))
        )
        self.assertTrue(
            np.all(self.Curv3.reshape(N, "F", "Fz", "V")[1] == np.zeros(self.Curv3.nFz))
        )
        self.assertTrue(
            np.all(self.Curv3.reshape(N, "F", "Fz", "V")[2] == np.ones(self.Curv3.nFz))
        )

    def test_grid(self):
        self.assertTrue(np.all(self.Curv2.gridCC == self.TM2.gridCC))
        self.assertTrue(np.all(self.Curv2.gridN == self.TM2.gridN))
        self.assertTrue(np.all(self.Curv2.gridFx == self.TM2.gridFx))
        self.assertTrue(np.all(self.Curv2.gridFy == self.TM2.gridFy))
        self.assertTrue(np.all(self.Curv2.gridEx == self.TM2.gridEx))
        self.assertTrue(np.all(self.Curv2.gridEy == self.TM2.gridEy))

        self.assertTrue(np.all(self.Curv3.gridCC == self.TM3.gridCC))
        self.assertTrue(np.all(self.Curv3.gridN == self.TM3.gridN))
        self.assertTrue(np.all(self.Curv3.gridFx == self.TM3.gridFx))
        self.assertTrue(np.all(self.Curv3.gridFy == self.TM3.gridFy))
        self.assertTrue(np.all(self.Curv3.gridFz == self.TM3.gridFz))
        self.assertTrue(np.all(self.Curv3.gridEx == self.TM3.gridEx))
        self.assertTrue(np.all(self.Curv3.gridEy == self.TM3.gridEy))
        self.assertTrue(np.all(self.Curv3.gridEz == self.TM3.gridEz))


@pytest.mark.parametrize("u_type", ["edge", "face"])
@pytest.mark.parametrize("dim", [2, 3], ids=["2D", "3D"])
@pytest.mark.parametrize("rep", [0, 1], ids=["uniform", "isotropic"])
def test_surface_inner_product_prop_deriv(u_type, dim, rep):
    rng = np.random.default_rng(6732)
    mesh, _ = setup_mesh("rotateCurv", 20, dim)
    tau = rng.uniform(1, 2, 1) if rep == 0 else rng.uniform(1, 2, mesh.n_faces * rep)

    match u_type:
        case "edge":
            v = rng.uniform(1, 2, mesh.n_edges)

            def fun(tau):
                M = mesh.get_edge_inner_product_surface(tau)
                Md = mesh.get_edge_inner_product_surface_deriv(tau)
                return M * v, Md(v)

        case "face":
            v = rng.uniform(1, 2, mesh.n_faces)

            def fun(tau):
                M = mesh.get_face_inner_product_surface(tau)
                Md = mesh.get_face_inner_product_surface_deriv(tau)
                return M * v, Md(v)

        case _:
            raise Exception("Invalid test parameter.")

    check_derivative(fun, tau, num=5, random_seed=rng)


@pytest.mark.parametrize("dim", [2, 3], ids=["2D", "3D"])
@pytest.mark.parametrize("rep", [0, 1], ids=["uniform", "isotropic"])
def test_line_inner_product_prop_deriv(dim, rep):
    rng = np.random.default_rng(6732)
    mesh, _ = setup_mesh("rotateCurv", 20, dim)
    v = rng.uniform(1, 2, mesh.n_edges)
    tau = rng.uniform(1, 2, 1) if rep == 0 else rng.uniform(1, 2, mesh.n_edges * rep)

    def fun(tau):
        M = mesh.get_edge_inner_product_line(tau)
        Md = mesh.get_edge_inner_product_line_deriv(tau)
        return M * v, Md(v)

    check_derivative(fun, tau, num=5, random_seed=rng)


if __name__ == "__main__":
    unittest.main()
