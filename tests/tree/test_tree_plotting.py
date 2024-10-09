import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import unittest
from discretize import TreeMesh

matplotlib.use("Agg")

rng = np.random.default_rng(4213678)


class TestOcTreePlotting(unittest.TestCase):
    def setUp(self):
        mesh = TreeMesh([32, 32, 32])
        mesh.refine_box([0.2, 0.2, 0.2], [0.5, 0.7, 0.8], 5)
        self.mesh = mesh

    def test_plot_slice(self):
        mesh = self.mesh

        plt.figure()
        ax = plt.subplot(111)

        mesh.plot_grid(faces=True, edges=True, nodes=True)

        # CC plot
        mod_cc = rng.random(len(mesh)) + 1j * rng.random(len(mesh))
        mod_cc[rng.random(len(mesh)) < 0.2] = np.nan

        mesh.plot_slice(mod_cc, normal="X", grid=True)
        mesh.plot_slice(mod_cc, normal="Y", ax=ax)
        mesh.plot_slice(mod_cc, normal="Z", ax=ax)
        mesh.plot_slice(mod_cc, view="imag", ax=ax)
        mesh.plot_slice(mod_cc, view="abs", ax=ax)

        mod_ccv = rng.random((len(mesh), 3))
        mesh.plot_slice(mod_ccv, v_type="CCv", view="vec", ax=ax)

        # F plot tests
        mod_f = rng.random(mesh.n_faces)
        mesh.plot_slice(mod_f, v_type="Fx", ax=ax)
        mesh.plot_slice(mod_f, v_type="Fy", ax=ax)
        mesh.plot_slice(mod_f, v_type="Fz", ax=ax)
        mesh.plot_slice(mod_f, v_type="F", ax=ax)
        mesh.plot_slice(mod_f, v_type="F", view="vec", ax=ax)

        # E plot tests
        mod_e = rng.random(mesh.n_edges)
        mesh.plot_slice(mod_e, v_type="Ex", ax=ax)
        mesh.plot_slice(mod_e, v_type="Ey", ax=ax)
        mesh.plot_slice(mod_e, v_type="Ez", ax=ax)
        mesh.plot_slice(mod_e, v_type="E", ax=ax)
        mesh.plot_slice(mod_e, v_type="E", view="vec", ax=ax)

        # Nodes
        mod_n = rng.random(mesh.n_nodes)
        mesh.plot_slice(mod_n, v_type="N")
        plt.close("all")


class TestQuadTreePlotting(unittest.TestCase):
    def setUp(self):
        mesh = TreeMesh([32, 32])
        mesh.refine_box([0.2, 0.2], [0.5, 0.8], 5)
        self.mesh = mesh

    def test_plot_slice(self):
        mesh = self.mesh
        plt.figure()
        ax = plt.subplot(111)

        mesh.plot_grid(faces=True, edges=True, nodes=True)

        # CC plot
        mod_cc = rng.random(len(mesh)) + 1j * rng.random(len(mesh))
        mod_cc[rng.random(len(mesh)) < 0.2] = np.nan

        mesh.plot_image(mod_cc)
        mesh.plot_image(mod_cc, ax=ax)
        mesh.plot_image(mod_cc, view="imag", ax=ax)
        mesh.plot_image(mod_cc, view="abs", ax=ax)

        mod_ccv = rng.random((len(mesh), 2))
        mesh.plot_image(mod_ccv, v_type="CCv", view="vec", ax=ax)

        # F plot tests
        mod_f = rng.random(mesh.n_faces)
        mesh.plot_image(mod_f, v_type="Fx", ax=ax)
        mesh.plot_image(mod_f, v_type="Fy", ax=ax)
        mesh.plot_image(mod_f, v_type="F", ax=ax)
        mesh.plot_image(mod_f, v_type="F", view="vec", ax=ax)

        # E plot tests
        mod_e = rng.random(mesh.n_edges)
        mesh.plot_image(mod_e, v_type="Ex", ax=ax)
        mesh.plot_image(mod_e, v_type="Ey", ax=ax)
        mesh.plot_image(mod_e, v_type="E", ax=ax)
        mesh.plot_image(mod_e, v_type="E", view="vec", ax=ax)

        # Nodes
        mod_n = rng.random(mesh.n_nodes)
        mesh.plot_image(mod_n, v_type="N", ax=ax)
        plt.close("all")
