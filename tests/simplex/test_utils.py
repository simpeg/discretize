import unittest
import numpy as np
import discretize
from discretize.utils import example_simplex_mesh
import os
import pickle
import matplotlib.pyplot as plt

try:
    import vtk  # NOQA F401

    has_vtk = True
except ImportError:
    has_vtk = False

rng = np.random.default_rng(87916253)


class SimplexTests(unittest.TestCase):
    def test_init_errors(self):
        bad_nodes = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [2, 2, 0],
            ]
        )
        simplices = np.array([[0, 1, 2, 3]])
        with self.assertRaises(ValueError):
            discretize.SimplexMesh(bad_nodes, simplices)

        good_nodes = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ]
        )
        with self.assertRaises(ValueError):
            # pass incompatible shaped nodes and simplices
            discretize.SimplexMesh(good_nodes, simplices[:, :-1])

        with self.assertRaises(ValueError):
            # pass bad dimensionality
            discretize.SimplexMesh(np.ones((10, 4)), simplices[:, :-1])

    def test_find_containing(self):
        n = 4
        points, simplices = example_simplex_mesh((n, n))
        mesh = discretize.SimplexMesh(points, simplices)

        x = np.array([[0.1, 0.2], [0.3, 0.4]])

        inds = mesh.point2index(x)
        np.testing.assert_equal(inds, [16, 5])

    def test_pickle2D(self):
        n = 5
        points, simplices = discretize.utils.example_simplex_mesh((n, n))
        mesh0 = discretize.SimplexMesh(points, simplices)

        byte_string = pickle.dumps(mesh0)
        mesh1 = pickle.loads(byte_string)

        np.testing.assert_equal(mesh0.nodes, mesh1.nodes)
        np.testing.assert_equal(mesh0._simplices, mesh1._simplices)

    def test_pickle3D(self):
        n = 5
        points, simplices = discretize.utils.example_simplex_mesh((n, n, n))
        mesh0 = discretize.SimplexMesh(points, simplices)

        byte_string = pickle.dumps(mesh0)
        mesh1 = pickle.loads(byte_string)

        np.testing.assert_equal(mesh0.nodes, mesh1.nodes)
        np.testing.assert_equal(mesh0._simplices, mesh1._simplices)

    def test_image_plotting(self):
        n = 5
        points, simplices = discretize.utils.example_simplex_mesh((n, n))
        mesh = discretize.SimplexMesh(points, simplices)

        cc_dat = rng.random(mesh.n_cells)
        n_dat = rng.random(mesh.n_nodes)
        f_dat = rng.random(mesh.n_faces)
        e_dat = rng.random(mesh.n_edges)
        ccv_dat = rng.random((mesh.n_cells, 2))

        mesh.plot_image(cc_dat)
        mesh.plot_image(ccv_dat, v_type="CCv", view="vec")

        mesh.plot_image(n_dat)

        mesh.plot_image(f_dat, v_type="Fx")
        mesh.plot_image(f_dat, v_type="Fy")
        mesh.plot_image(f_dat, v_type="F")
        mesh.plot_image(f_dat, v_type="F", view="vec")

        mesh.plot_image(e_dat, v_type="Ex")
        mesh.plot_image(e_dat, v_type="Ey")
        mesh.plot_image(e_dat, v_type="E")
        mesh.plot_image(e_dat, v_type="E", view="vec")

        with self.assertRaises(NotImplementedError):
            points, simplices = discretize.utils.example_simplex_mesh((n, n, n))
            mesh = discretize.SimplexMesh(points, simplices)

            cc_dat = rng.random(mesh.n_cells)
            mesh.plot_image(cc_dat)

        plt.close("all")

    def test_plot_grid(self):
        n = 5
        points, simplices = discretize.utils.example_simplex_mesh((n, n))
        mesh = discretize.SimplexMesh(points, simplices)
        mesh.plot_grid(nodes=True, faces=True, edges=True, centers=True)

        points, simplices = discretize.utils.example_simplex_mesh((n, n, n))
        mesh = discretize.SimplexMesh(points, simplices)
        mesh.plot_grid(nodes=True, faces=True, edges=True, centers=True)
        plt.close("all")

    if has_vtk:

        def test_2D_vtk(self):
            n = 5
            points, simplices = discretize.utils.example_simplex_mesh((n, n))
            mesh = discretize.SimplexMesh(points, simplices)
            cc_dat = rng.random(mesh.n_cells)

            vtk_obj = mesh.to_vtk(models={"info": cc_dat})

            mesh2, models = discretize.SimplexMesh.vtk_to_simplex_mesh(vtk_obj)

            np.testing.assert_equal(mesh.nodes, mesh2.nodes)
            np.testing.assert_equal(mesh._simplices, mesh2._simplices)
            np.testing.assert_equal(cc_dat, models["info"])

            mesh.write_vtk("test.vtu", models={"info": cc_dat})
            mesh2, models = discretize.SimplexMesh.read_vtk("test.vtu")

            np.testing.assert_equal(mesh.nodes, mesh2.nodes)
            np.testing.assert_equal(mesh._simplices, mesh2._simplices)
            np.testing.assert_equal(cc_dat, models["info"])

        def test_3D_vtk(self):
            n = 5
            points, simplices = discretize.utils.example_simplex_mesh((n, n, n))
            mesh = discretize.SimplexMesh(points, simplices)
            cc_dat = rng.random(mesh.n_cells)

            vtk_obj = mesh.to_vtk(models={"info": cc_dat})

            mesh2, models = discretize.SimplexMesh.vtk_to_simplex_mesh(vtk_obj)

            np.testing.assert_equal(mesh.nodes, mesh2.nodes)
            np.testing.assert_equal(mesh._simplices, mesh2._simplices)
            np.testing.assert_equal(cc_dat, models["info"])

            mesh.write_vtk("test.vtu", models={"info": cc_dat})
            mesh2, models = discretize.SimplexMesh.read_vtk("test.vtu")

            np.testing.assert_equal(mesh.nodes, mesh2.nodes)
            np.testing.assert_equal(mesh._simplices, mesh2._simplices)
            np.testing.assert_equal(cc_dat, models["info"])

    def tearDown(self):
        try:
            os.remove("test.vtu")
        except FileNotFoundError:
            pass
