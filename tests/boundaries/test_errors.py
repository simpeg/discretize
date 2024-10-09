import unittest
import numpy as np
import discretize


rng = np.random.default_rng(53679)


class RobinOperatorTest(unittest.TestCase):
    def setUp(self):
        self.mesh = discretize.TensorMesh([18, 20, 32])

    def testCellGradBroadcasting(self):
        mesh = self.mesh
        boundary_faces = mesh.boundary_faces

        n_boundary_faces = boundary_faces.shape[0]

        alpha = np.full(n_boundary_faces, 0.5)
        beta = np.full(n_boundary_faces, 1.5)
        gamma = np.full(n_boundary_faces, 2.0)

        B1, b1 = mesh.cell_gradient_weak_form_robin(alpha=alpha, beta=beta, gamma=gamma)

        for a in [0.5, alpha]:
            for b in [1.5, beta]:
                for g in [2.0, gamma]:
                    B_t, bt = mesh.cell_gradient_weak_form_robin(
                        alpha=a, beta=b, gamma=g
                    )
                    np.testing.assert_equal(bt, b1)
                    self.assertEqual((B1 - B_t).nnz, 0)

        gamma = rng.random((n_boundary_faces, 2))
        B1, b1 = mesh.cell_gradient_weak_form_robin(
            alpha=0.5, beta=1.5, gamma=gamma[:, 0]
        )
        B2, b2 = mesh.cell_gradient_weak_form_robin(
            alpha=0.5, beta=1.5, gamma=gamma[:, 1]
        )
        B3, b3 = mesh.cell_gradient_weak_form_robin(alpha=0.5, beta=1.5, gamma=gamma)
        np.testing.assert_allclose(B1.data, B2.data)
        np.testing.assert_allclose(B1.data, B3.data)
        np.testing.assert_allclose(np.c_[b1, b2], b3)

    def testEdgeDivBroadcasting(self):
        mesh = self.mesh
        boundary_faces = mesh.boundary_faces
        boundary_nodes = mesh.boundary_nodes

        n_boundary_faces = boundary_faces.shape[0]
        n_boundary_nodes = boundary_nodes.shape[0]

        alpha = np.full(n_boundary_faces, 0.5)
        beta = np.full(n_boundary_faces, 1.5)
        gamma = np.full(n_boundary_faces, 2.0)

        B1, b1 = mesh.edge_divergence_weak_form_robin(
            alpha=alpha, beta=beta, gamma=gamma
        )

        for a in [0.5, alpha]:
            for b in [1.5, beta]:
                for g in [2.0, gamma]:
                    B_t, bt = mesh.edge_divergence_weak_form_robin(
                        alpha=a, beta=b, gamma=g
                    )
                    np.testing.assert_allclose(bt, b1)
                    np.testing.assert_allclose(B1.data, B_t.data)

        alpha = np.full(n_boundary_nodes, 0.5)
        beta = np.full(n_boundary_nodes, 1.5)
        gamma = np.full(n_boundary_nodes, 2.0)

        for a in [0.5, alpha]:
            for b in [1.5, beta]:
                for g in [2.0, gamma]:
                    B_t, bt = mesh.edge_divergence_weak_form_robin(
                        alpha=a, beta=b, gamma=g
                    )
                    np.testing.assert_allclose(bt, b1)
                    np.testing.assert_allclose(B1.data, B_t.data)

        gamma = rng.random((n_boundary_faces, 2))
        B1, b1 = mesh.edge_divergence_weak_form_robin(
            alpha=0.5, beta=1.5, gamma=gamma[:, 0]
        )
        B2, b2 = mesh.edge_divergence_weak_form_robin(
            alpha=0.5, beta=1.5, gamma=gamma[:, 1]
        )
        B3, b3 = mesh.edge_divergence_weak_form_robin(alpha=0.5, beta=1.5, gamma=gamma)
        np.testing.assert_allclose(B1.data, B2.data)
        np.testing.assert_allclose(B1.data, B3.data)
        np.testing.assert_allclose(np.c_[b1, b2], b3)

        gamma = rng.random((n_boundary_nodes, 2))
        B1, b1 = mesh.edge_divergence_weak_form_robin(
            alpha=0.5, beta=1.5, gamma=gamma[:, 0]
        )
        B2, b2 = mesh.edge_divergence_weak_form_robin(
            alpha=0.5, beta=1.5, gamma=gamma[:, 1]
        )
        B3, b3 = mesh.edge_divergence_weak_form_robin(alpha=0.5, beta=1.5, gamma=gamma)
        np.testing.assert_allclose(B1.data, B2.data)
        np.testing.assert_allclose(B1.data, B3.data)
        np.testing.assert_allclose(np.c_[b1, b2], b3)

    def testEdgeDivErrors(self):
        mesh = self.mesh
        boundary_faces = mesh.boundary_faces
        boundary_nodes = mesh.boundary_nodes

        n_boundary_faces = boundary_faces.shape[0]
        n_boundary_nodes = boundary_nodes.shape[0]

        alpha_f = np.full(n_boundary_faces, 0.5)
        beta_n = np.full(n_boundary_nodes, 1.5)
        gamma_n = np.full(n_boundary_nodes, 2.0)

        # throws error if a beta is 0
        with self.assertRaises(ValueError):
            mesh.edge_divergence_weak_form_robin(1.0, 0.0, 1.0)

        # incorrect length of an input
        with self.assertRaises(ValueError):
            mesh.edge_divergence_weak_form_robin(3, 2, [1.0, 0.0])

        with self.assertRaises(ValueError):
            # inconsistent input lengths
            mesh.edge_divergence_weak_form_robin(alpha_f, beta_n, gamma_n)


class mesh1DTests(unittest.TestCase):
    def setUp(self):
        self.mesh, _ = discretize.tests.setup_mesh("uniformTensorMesh", 32, 1)

    def testItems(self):
        mesh = self.mesh
        np.testing.assert_equal(mesh.boundary_faces, mesh.boundary_nodes)

        self.assertIs(mesh.boundary_edges, None)
        self.assertIs(mesh.project_edge_to_boundary_edge, None)
