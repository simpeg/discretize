import unittest
import numpy as np
import discretize
import os
import pickle

try:
    import vtk
except ImportError:
    has_vtk = False
else:
    has_vtk = True


class SimplexTests(unittest.TestCase):

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

        cc_dat = np.random.rand(mesh.n_cells)
        n_dat = np.random.rand(mesh.n_nodes)
        f_dat = np.random.rand(mesh.n_faces)
        e_dat = np.random.rand(mesh.n_edges)
        ccv_dat = np.random.rand(mesh.n_cells, 2)

        mesh.plot_image(cc_dat)
        mesh.plot_image(ccv_dat, v_type='CCv', view='vec')

        mesh.plot_image(n_dat)

        mesh.plot_image(f_dat, v_type='Fx')
        mesh.plot_image(f_dat, v_type='Fy')
        mesh.plot_image(f_dat, v_type='F')
        mesh.plot_image(f_dat, v_type='F', view='vec')

        mesh.plot_image(e_dat, v_type='Ex')
        mesh.plot_image(e_dat, v_type='Ey')
        mesh.plot_image(e_dat, v_type='E')
        mesh.plot_image(e_dat, v_type='E', view='vec')

        with self.assertRaises(NotImplementedError):
            points, simplices = discretize.utils.example_simplex_mesh((n, n, n))
            mesh = discretize.SimplexMesh(points, simplices)

            cc_dat = np.random.rand(mesh.n_cells)
            mesh.plot_image(cc_dat)

    def test_plot_grid(self):
        n = 5
        points, simplices = discretize.utils.example_simplex_mesh((n, n))
        mesh = discretize.SimplexMesh(points, simplices)
        mesh.plot_grid(nodes=True, faces=True, edges=True, centers=True)

        points, simplices = discretize.utils.example_simplex_mesh((n, n, n))
        mesh = discretize.SimplexMesh(points, simplices)
        mesh.plot_grid(nodes=True, faces=True, edges=True, centers=True)

    if has_vtk:
        def test_2D_vtk(self):
            n = 5
            points, simplices = discretize.utils.example_simplex_mesh((n, n))
            mesh = discretize.SimplexMesh(points, simplices)
            cc_dat = np.random.rand(mesh.n_cells)

            vtk_obj = mesh.to_vtk(models={'info': cc_dat})

            mesh2, models = discretize.SimplexMesh.vtk_to_simplex_mesh(vtk_obj)

            np.testing.assert_equal(mesh.nodes, mesh2.nodes)
            np.testing.assert_equal(mesh._simplices, mesh2._simplices)
            np.testing.assert_equal(cc_dat, models['info'])

            mesh.write_vtk('test.vtu', models={'info': cc_dat})
            mesh2, models = discretize.SimplexMesh.read_vtk('test.vtu')

            np.testing.assert_equal(mesh.nodes, mesh2.nodes)
            np.testing.assert_equal(mesh._simplices, mesh2._simplices)
            np.testing.assert_equal(cc_dat, models['info'])

        def test_3D_vtk(self):
            n = 5
            points, simplices = discretize.utils.example_simplex_mesh((n, n, n))
            mesh = discretize.SimplexMesh(points, simplices)
            cc_dat = np.random.rand(mesh.n_cells)

            vtk_obj = mesh.to_vtk(models={'info': cc_dat})

            mesh2, models = discretize.SimplexMesh.vtk_to_simplex_mesh(vtk_obj)

            np.testing.assert_equal(mesh.nodes, mesh2.nodes)
            np.testing.assert_equal(mesh._simplices, mesh2._simplices)
            np.testing.assert_equal(cc_dat, models['info'])

            mesh.write_vtk('test.vtu', models={'info': cc_dat})
            mesh2, models = discretize.SimplexMesh.read_vtk('test.vtu')

            np.testing.assert_equal(mesh.nodes, mesh2.nodes)
            np.testing.assert_equal(mesh._simplices, mesh2._simplices)
            np.testing.assert_equal(cc_dat, models['info'])

    def tearDown(self):
        try:
            os.remove("test.vtu")
        except FileNotFoundError:
            pass
