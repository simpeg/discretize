import unittest
import numpy as np

import discretize
from discretize import utils

TOL = 1e-1


class TestCyl3DGeometries(unittest.TestCase):
    def setUp(self):
        hx = utils.unpack_widths([(1, 1)])
        htheta = utils.unpack_widths([(1.0, 4)])
        htheta = htheta * 2 * np.pi / htheta.sum()
        hz = hx

        self.mesh = discretize.CylindricalMesh([hx, htheta, hz])

    def test_areas(self):
        area = self.mesh.face_areas
        self.assertTrue(self.mesh.nF == len(area))
        self.assertTrue(
            area[: self.mesh.vnF[0]].sum()
            == 2 * np.pi * self.mesh.h[0] * self.mesh.h[2]
        )
        self.assertTrue(
            np.all(
                area[self.mesh.vnF[0] : self.mesh.vnF[1]]
                == self.mesh.h[0] * self.mesh.h[2]
            )
        )
        self.assertTrue(
            np.all(
                area[sum(self.mesh.vnF[:2]) :]
                == np.pi * self.mesh.h[0] ** 2 / self.mesh.shape_cells[1]
            )
        )

    def test_edges(self):
        edge = self.mesh.edge_lengths
        self.assertTrue(self.mesh.nE == len(edge))
        self.assertTrue(np.all(edge[: self.mesh.vnF[0]] == self.mesh.h[0]))
        self.assertTrue(
            np.all(
                self.mesh.edge_lengths[self.mesh.vnE[0] : sum(self.mesh.vnE[:2])]
                == np.kron(
                    np.ones(self.mesh.shape_cells[2] + 1),
                    self.mesh.h[0] * self.mesh.h[1],
                )
            )
        )
        self.assertTrue(
            np.all(
                self.mesh.edge_lengths[self.mesh.vnE[0] : sum(self.mesh.vnE[:2])]
                == np.kron(
                    np.ones(self.mesh.shape_cells[2] + 1),
                    self.mesh.h[0] * self.mesh.h[1],
                )
            )
        )
        self.assertTrue(
            np.all(
                self.mesh.edge_lengths[sum(self.mesh.vnE[:2]) :]
                == np.kron(self.mesh.h[2], np.ones(self.mesh.shape_cells[1] + 1))
            )
        )

    def test_vol(self):
        self.assertTrue(
            self.mesh.cell_volumes.sum() == np.pi * self.mesh.h[0] ** 2 * self.mesh.h[2]
        )
        self.assertTrue(
            np.all(
                self.mesh.cell_volumes
                == np.pi
                * self.mesh.h[0] ** 2
                * self.mesh.h[2]
                / self.mesh.shape_cells[1]
            )
        )


def test_boundary_items():
    mesh = discretize.CylindricalMesh([3, 4, 5])

    # Nodes
    is_bn = (mesh.nodes[:, 0] == 1) | (mesh.nodes[:, 2] == 0) | (mesh.nodes[:, 2] == 1)
    np.testing.assert_equal(mesh.boundary_nodes, mesh.nodes[is_bn])
    P_bn = mesh.project_node_to_boundary_node
    np.testing.assert_equal(mesh.boundary_nodes, P_bn @ mesh.nodes)

    # Edges
    is_be = (mesh.edges[:, 0] == 1) | (mesh.edges[:, 2] == 0) | (mesh.edges[:, 2] == 1)
    np.testing.assert_equal(mesh.boundary_edges, mesh.edges[is_be])
    P_be = mesh.project_edge_to_boundary_edge
    np.testing.assert_equal(mesh.boundary_edges, P_be @ mesh.edges)

    # Faces
    is_bf = (mesh.faces[:, 0] == 1) | (mesh.faces[:, 2] == 0) | (mesh.faces[:, 2] == 1)
    np.testing.assert_equal(mesh.boundary_faces, mesh.faces[is_bf])
    P_bf = mesh.project_face_to_boundary_face
    np.testing.assert_equal(mesh.boundary_faces, P_bf @ mesh.faces)


# ----------------------- Test Grids and Counting --------------------------- #


class Cyl3DGrid(unittest.TestCase):
    def setUp(self):
        self.mesh = discretize.CylindricalMesh([2, 4, 1])

    def test_counting(self):
        mesh = self.mesh

        # cell centers
        self.assertEqual(mesh.nC, 8)
        self.assertEqual(mesh.shape_cells[0], 2)
        self.assertEqual(mesh.shape_cells[1], 4)
        self.assertEqual(mesh.shape_cells[2], 1)
        self.assertEqual(mesh.vnC, (2, 4, 1))

        # faces
        self.assertEqual(mesh.nFx, 8)
        self.assertEqual(mesh.nFy, 8)
        self.assertEqual(mesh.nFz, 16)
        self.assertEqual(mesh.nF, 32)
        self.assertEqual(mesh.vnFx, (2, 4, 1))
        self.assertEqual(mesh.vnFy, (2, 4, 1))
        self.assertEqual(mesh.vnFz, (2, 4, 2))
        self.assertEqual(mesh.vnF, (8, 8, 16))

        # edges
        self.assertEqual(mesh.nEx, 16)
        self.assertEqual(mesh.nEy, 16)
        self.assertEqual(mesh.nEz, 9)  # there is an edge at the center
        self.assertEqual(mesh.nE, 41)
        self.assertEqual(mesh.vnEx, (2, 4, 2))
        self.assertEqual(mesh.vnEy, (2, 4, 2))
        self.assertEqual(mesh.vnEz, (3, 4, 1))
        self.assertEqual(mesh.vnE, (16, 16, 9))
        self.assertNotEqual(np.prod(mesh.vnEz), mesh.nEz)  # periodic boundary condition

        # nodes
        self.assertEqual(mesh.shape_nodes[0], 3)
        self.assertEqual(mesh.shape_nodes[1], 4)
        self.assertEqual(mesh.shape_nodes[2], 2)
        self.assertEqual(mesh.vnN, (3, 4, 2))
        self.assertEqual(mesh.nN, 18)
        self.assertNotEqual(mesh.nN, np.prod(mesh.vnN))  # periodic boundary condition

    def test_gridCC(self):
        mesh = self.mesh

        # Cell centers
        self.assertTrue((mesh.cell_centers_x == [0.25, 0.75]).all())
        self.assertTrue(
            (
                mesh.cell_centers_y
                == 2.0 * np.pi * np.r_[1.0 / 8.0, 3.0 / 8.0, 5.0 / 8.0, 7.0 / 8.0]
            ).all()
        )
        self.assertTrue(mesh.cell_centers_z == 0.5)

        self.assertTrue((mesh.gridCC[:, 0] == 4 * [0.25, 0.75]).all())
        self.assertTrue(
            (
                mesh.gridCC[:, 1]
                == 2.0
                * np.pi
                * np.r_[
                    1.0 / 8.0,
                    1.0 / 8.0,
                    3.0 / 8.0,
                    3.0 / 8.0,
                    5.0 / 8.0,
                    5.0 / 8.0,
                    7.0 / 8.0,
                    7.0 / 8.0,
                ]
            ).all()
        )
        self.assertTrue((mesh.gridCC[:, 2] == 8 * [0.5]).all())

    def test_gridN(self):
        mesh = self.mesh

        # Nodes
        self.assertTrue((mesh.nodes_x == [0.0, 0.5, 1.0]).all())
        self.assertTrue((mesh.nodes_y == 2 * np.pi * np.r_[0.0, 0.25, 0.5, 0.75]).all())
        self.assertTrue((mesh.nodes_z == np.r_[0.0, 1.0]).all())

        self.assertTrue(
            (
                mesh.gridN[:, 0] == 2 * [0.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0]
            ).all()
        )
        self.assertTrue(
            (
                mesh.gridN[:, 1]
                == 2
                * np.pi
                * np.hstack(
                    2 * [3 * [0.0], 2 * [1.0 / 4.0], 2 * [1.0 / 2.0], 2 * [3.0 / 4.0]]
                )
            ).all()
        )
        self.assertTrue((mesh.gridN[:, 2] == 9 * [0.0] + 9 * [1.0]).all())

    def test_gridFx(self):
        mesh = self.mesh

        # x-faces
        self.assertTrue((mesh.gridFx[:, 0] == 4 * [0.5, 1.0]).all())
        self.assertTrue(
            (
                mesh.gridFx[:, 1]
                == 2
                * np.pi
                * np.hstack(
                    [2 * [1.0 / 8.0], 2 * [3.0 / 8.0], 2 * [5.0 / 8.0], 2 * [7.0 / 8.0]]
                )
            ).all()
        )
        self.assertTrue((mesh.gridFx[:, 2] == 8 * [0.5]).all())

    def test_gridFy(self):
        mesh = self.mesh

        # y-faces
        self.assertTrue((mesh.gridFy[:, 0] == 4 * [0.25, 0.75]).all())
        self.assertTrue(
            (
                mesh.gridFy[:, 1]
                == 2
                * np.pi
                * np.hstack(
                    [2 * [0.0], 2 * [1.0 / 4.0], 2 * [1.0 / 2.0], 2 * [3.0 / 4.0]]
                )
            ).all()
        )
        self.assertTrue((mesh.gridFy[:, 2] == 8 * [0.5]).all())

    def test_gridFz(self):
        mesh = self.mesh

        # z-faces
        self.assertTrue((mesh.gridFz[:, 0] == 8 * [0.25, 0.75]).all())
        self.assertTrue(
            (
                mesh.gridFz[:, 1]
                == 2
                * np.pi
                * np.hstack(
                    2
                    * [
                        2 * [1.0 / 8.0],
                        2 * [3.0 / 8.0],
                        2 * [5.0 / 8.0],
                        2 * [7.0 / 8.0],
                    ]
                )
            ).all()
        )
        self.assertTrue((mesh.gridFz[:, 2] == np.hstack([8 * [0.0], 8 * [1.0]])).all())

    def test_gridEx(self):
        mesh = self.mesh

        # x-edges
        self.assertTrue((mesh.gridEx[:, 0] == 8 * [0.25, 0.75]).all())
        self.assertTrue(
            (
                mesh.gridEx[:, 1]
                == 2
                * np.pi
                * np.hstack(
                    2 * [2 * [0.0], 2 * [1.0 / 4.0], 2 * [1.0 / 2.0], 2 * [3.0 / 4.0]]
                )
            ).all()
        )
        self.assertTrue((mesh.gridEx[:, 2] == np.hstack([8 * [0.0], 8 * [1.0]])).all())

    def test_gridEy(self):
        mesh = self.mesh

        # y-edges
        self.assertTrue((mesh.gridEy[:, 0] == 8 * [0.5, 1.0]).all())
        self.assertTrue(
            (
                mesh.gridEy[:, 1]
                == 2
                * np.pi
                * np.hstack(
                    2
                    * [
                        2 * [1.0 / 8.0],
                        2 * [3.0 / 8.0],
                        2 * [5.0 / 8.0],
                        2 * [7.0 / 8.0],
                    ]
                )
            ).all()
        )
        self.assertTrue((mesh.gridEy[:, 2] == np.hstack([8 * [0.0], 8 * [1.0]])).all())

    def test_gridEz(self):
        mesh = self.mesh

        # z-edges
        self.assertTrue(
            (mesh.gridEz[:, 0] == np.hstack([[0.0, 0.5, 1.0] + 3 * [0.5, 1.0]])).all()
        )
        self.assertTrue(
            (
                mesh.gridEz[:, 1]
                == 2
                * np.pi
                * np.hstack(
                    [3 * [0.0], 2 * [1.0 / 4.0], 2 * [1.0 / 2.0], 2 * [3.0 / 4.0]]
                )
            ).all()
        )
        self.assertTrue((mesh.gridEz[:, 2] == 9 * [0.5]).all())


# ------------------- Test conversion to Cartesian ----------------------- #


class TestCartesianGrid(unittest.TestCase):
    def test_cartesianGrid(self):
        mesh = discretize.CylindricalMesh([1, 4, 1])

        root2over2 = np.sqrt(2.0) / 2.0

        # cell centers
        cartCC = mesh.cartesian_grid("CC")
        self.assertTrue(
            np.allclose(cartCC[:, 0], 0.5 * root2over2 * np.r_[1.0, -1.0, -1.0, 1.0])
        )
        self.assertTrue(
            np.allclose(cartCC[:, 1], 0.5 * root2over2 * np.r_[1.0, 1.0, -1.0, -1.0])
        )
        self.assertTrue(np.allclose(cartCC[:, 2], 0.5 * np.ones(4)))

        # nodes
        cartN = mesh.cartesian_grid("N")
        self.assertTrue(
            np.allclose(cartN[:, 0], np.hstack(2 * [0.0, 1.0, 0.0, -1.0, 0.0]))
        )
        self.assertTrue(
            np.allclose(cartN[:, 1], np.hstack(2 * [0.0, 0.0, 1.0, 0.0, -1.0]))
        )
        self.assertTrue(np.allclose(cartN[:, 2], np.hstack(5 * [0.0] + 5 * [1.0])))


class Deflation(unittest.TestCase):
    def test_areas(self):
        mesh = discretize.CylindricalMesh([1, 2, 1])

        areas = np.hstack([[np.pi] * 2, [1] * 2, [np.pi / 2] * 4])
        self.assertTrue(np.all(mesh.face_areas == areas))

        edges = np.hstack([[1] * 4, [np.pi] * 4, [1] * 3])
        self.assertTrue(np.all(mesh.edge_lengths == edges))

        mesh = discretize.CylindricalMesh([2, 5, 3])

        hangingF = np.hstack(
            [
                getattr(mesh, "_ishanging_faces_{}".format(dim))
                for dim in ["x", "y", "z"]
            ]
        )
        self.assertTrue(np.all(mesh._face_areas_full[~hangingF] == mesh.face_areas))
        hangingE = np.hstack(
            [
                getattr(mesh, "_ishanging_edges_{}".format(dim))
                for dim in ["x", "y", "z"]
            ]
        )
        self.assertTrue(np.all(mesh._edge_lengths_full[~hangingE] == mesh.edge_lengths))


if __name__ == "__main__":
    unittest.main()
