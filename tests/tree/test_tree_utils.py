import numpy as np
import unittest
import pytest
import discretize
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


def _refined_tree():
    mesh = discretize.TreeMesh(
        [np.full(16, 10.0), np.full(8, 20.0), np.full(32, 5.0)],
        origin=[-80.0, -80.0, -80.0],
        diagonal_balance=True,
    )
    mesh.refine_ball([5.0, -3.0, 10.0], 30.0, levels=-1, finalize=False)
    mesh.refine_box([-80.0, -80.0, -20.0], [80.0, 80.0, 0.0], levels=-2)
    return mesh


@pytest.mark.parametrize("normal", ["X", "Y", "Z"])
@pytest.mark.parametrize("location", [3.0, -61.0, 45.5])
def test_slice_tree_mesh_cells_and_indices(normal, location):
    from discretize.utils import slice_tree_mesh

    mesh = _refined_tree()
    axis = "XYZ".index(normal)
    in_plane = [i for i in range(3) if i != axis]
    mesh_2d, indices = slice_tree_mesh(mesh, normal, location, return_indices=True)

    # one 2D cell for every 3D cell cut by the plane
    lo = mesh.cell_centers[:, axis] - mesh.h_gridded[:, axis] / 2
    hi = mesh.cell_centers[:, axis] + mesh.h_gridded[:, axis] / 2
    cut = np.where((lo <= location) & (location < hi))[0]
    assert mesh_2d.n_cells == len(cut)
    np.testing.assert_array_equal(np.sort(indices), np.sort(cut))

    # each 2D cell is its 3D cell, seen in the plane
    np.testing.assert_allclose(
        mesh_2d.cell_centers, mesh.cell_centers[indices][:, in_plane]
    )
    np.testing.assert_allclose(mesh_2d.h_gridded, mesh.h_gridded[indices][:, in_plane])
    np.testing.assert_allclose(mesh_2d.origin, mesh.origin[in_plane])


def test_slice_tree_mesh_on_faces():
    from discretize.utils import slice_tree_mesh

    mesh = _refined_tree()
    for location, expected_side in ((0.0, "high"), (-80.0, "high"), (80.0, "low")):
        mesh_2d, indices = slice_tree_mesh(mesh, "Z", location, return_indices=True)
        lo = mesh.cell_centers[indices, 2] - mesh.h_gridded[indices, 2] / 2
        hi = mesh.cell_centers[indices, 2] + mesh.h_gridded[indices, 2] / 2
        if expected_side == "high":
            np.testing.assert_allclose(lo[np.isclose(lo, location)], location)
            assert np.all(hi > location)
        else:
            np.testing.assert_allclose(hi, location)


def test_slice_tree_mesh_default_matches_plot_slice_location():
    from discretize.utils import slice_tree_mesh

    mesh = _refined_tree()
    middle = mesh.cell_centers_z[len(mesh.h[2]) // 2]
    a = slice_tree_mesh(mesh, "Z")
    b = slice_tree_mesh(mesh, "Z", middle)
    np.testing.assert_allclose(a.cell_centers, b.cell_centers)


def test_slice_tree_mesh_errors():
    from discretize.utils import slice_tree_mesh

    mesh = _refined_tree()
    with pytest.raises(ValueError):
        slice_tree_mesh(mesh, "W")
    with pytest.raises(ValueError):
        slice_tree_mesh(mesh, "Z", 1000.0)
    mesh_2d = discretize.TreeMesh([8, 8])
    mesh_2d.refine(2)
    with pytest.raises(ValueError):
        slice_tree_mesh(mesh_2d, "Z")
