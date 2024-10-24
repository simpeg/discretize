import numpy as np
import unittest
from discretize.utils import mesh_builder_xyz, refine_tree_xyz

TOL = 1e-8


class TestRefineOcTree(unittest.TestCase):
    def test_radial(self):
        dx = 0.25
        rad = 10
        mesh = mesh_builder_xyz(
            np.c_[0.01, 0.01, 0.01],
            [dx, dx, dx],
            depth_core=0,
            padding_distance=[[0, 20], [0, 20], [0, 20]],
            mesh_type="TREE",
        )

        radCell = int(np.ceil(rad / dx))
        mesh = refine_tree_xyz(
            mesh,
            np.c_[0, 0, 0],
            octree_levels=[radCell],
            method="radial",
            finalize=True,
        )
        cell_levels = mesh.cell_levels_by_index(np.arange(mesh.n_cells))

        vol = 4.0 * np.pi / 3.0 * (rad + dx) ** 3.0

        vol_mesh = mesh.cell_volumes[cell_levels == mesh.max_level].sum()

        self.assertLess(np.abs(vol - vol_mesh) / vol, 0.05)

        levels, cells_per_level = np.unique(cell_levels, return_counts=True)

        self.assertEqual(mesh.n_cells, 311858)
        np.testing.assert_array_equal(levels, [3, 4, 5, 6, 7])
        np.testing.assert_array_equal(cells_per_level, [232, 1176, 2671, 9435, 298344])

    def test_box(self):
        dx = 0.25
        dl = 10

        # Create a box 2*dl in width
        X, Y, Z = np.meshgrid(np.c_[-dl, dl], np.c_[-dl, dl], np.c_[-dl, dl])
        xyz = np.c_[np.ravel(X), np.ravel(Y), np.ravel(Z)]
        mesh = mesh_builder_xyz(
            np.c_[0.01, 0.01, 0.01],
            [dx, dx, dx],
            depth_core=0,
            padding_distance=[[0, 20], [0, 20], [0, 20]],
            mesh_type="TREE",
        )

        mesh = refine_tree_xyz(
            mesh, xyz, octree_levels=[0, 1], method="box", finalize=True
        )
        cell_levels = mesh.cell_levels_by_index(np.arange(mesh.n_cells))

        vol = (2 * (dl + 2 * dx)) ** 3  # 2*dx is cell size at second to highest level
        vol_mesh = np.sum(mesh.cell_volumes[cell_levels == mesh.max_level - 1])
        self.assertLess((vol - vol_mesh) / vol, 0.05)

        levels, cells_per_level = np.unique(cell_levels, return_counts=True)

        self.assertEqual(mesh.n_cells, 80221)
        np.testing.assert_array_equal(levels, [3, 4, 5, 6])
        np.testing.assert_array_equal(cells_per_level, [80, 1762, 4291, 74088])

    def test_surface(self):
        dx = 0.1
        dl = 20

        # Define triangle
        xyz = np.r_[
            np.c_[-dl / 2, -dl / 2, 0],
            np.c_[dl / 2, -dl / 2, 0],
            np.c_[dl / 2, dl / 2, 0],
        ]
        mesh = mesh_builder_xyz(
            np.c_[0.01, 0.01, 0.01],
            [dx, dx, dx],
            depth_core=0,
            padding_distance=[[0, 20], [0, 20], [0, 20]],
            mesh_type="TREE",
        )

        mesh = refine_tree_xyz(
            mesh, xyz, octree_levels=[1], method="surface", finalize=True
        )

        # Volume of triangle
        vol = dl * dl * dx / 2

        residual = (
            np.abs(
                vol
                - mesh.cell_volumes[
                    mesh._cell_levels_by_indexes(range(mesh.nC)) == mesh.max_level
                ].sum()
                / 2
            )
            / vol
            * 100
        )

        self.assertLess(residual, 5)

    def test_errors(self):
        dx = 0.25
        rad = 10
        self.assertRaises(
            ValueError,
            mesh_builder_xyz,
            np.c_[0.01, 0.01, 0.01],
            [dx, dx, dx],
            depth_core=0,
            padding_distance=[[0, 20], [0, 20], [0, 20]],
            mesh_type="cyl",
        )

        mesh = mesh_builder_xyz(
            np.c_[0.01, 0.01, 0.01],
            [dx, dx, dx],
            depth_core=0,
            padding_distance=[[0, 20], [0, 20], [0, 20]],
            mesh_type="tree",
        )

        radCell = int(np.ceil(rad / dx))
        self.assertRaises(
            NotImplementedError,
            refine_tree_xyz,
            mesh,
            np.c_[0, 0, 0],
            octree_levels=[radCell],
            method="other",
            finalize=True,
        )

        self.assertRaises(
            ValueError,
            refine_tree_xyz,
            mesh,
            np.c_[0, 0, 0],
            octree_levels=[radCell],
            octree_levels_padding=[],
            method="surface",
            finalize=True,
        )


if __name__ == "__main__":
    unittest.main()
