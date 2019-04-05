from __future__ import print_function
import numpy as np
import unittest
import discretize
from discretize.utils import meshutils

TOL = 1e-8
np.random.seed(12)


class TestRefineOcTree(unittest.TestCase):

    def test_radial(self):
        dx = 0.25
        rad = 10
        mesh = meshutils.mesh_builder_xyz(
            np.c_[0.01, 0.01, 0.01], [dx, dx, dx],
            depth_core=0,
            padding_distance=[[0, 20], [0, 20], [0, 20]],
            mesh_type='TREE',
        )

        radCell = int(np.ceil(rad/dx))
        mesh = meshutils.refine_tree_xyz(
            mesh, np.c_[0, 0, 0],
            octree_levels=[radCell], method='radial',
            finalize=True
        )

        # Volume of sphere
        vol = 4.*np.pi/3.*rad**3.

        residual = np.abs(
            vol -
            mesh.vol[
                mesh._cell_levels_by_indexes(range(mesh.nC)) == mesh.max_level
            ].sum()
        )/vol * 100

        self.assertTrue(residual < 3)

    def test_box(self):
        dx = 0.25
        dl = 10

        # Create a box 2*dl in width
        X, Y, Z = np.meshgrid(np.c_[-dl, dl], np.c_[-dl, dl], np.c_[-dl, dl])
        xyz = np.c_[np.ravel(X), np.ravel(Y), np.ravel(Z)]
        mesh = meshutils.mesh_builder_xyz(
            np.c_[0.01, 0.01, 0.01], [dx, dx, dx],
            depth_core=0,
            padding_distance=[[0, 20], [0, 20], [0, 20]],
            mesh_type='TREE',
        )

        mesh = meshutils.refine_tree_xyz(
            mesh, xyz, octree_levels=[1], method='box', finalize=True
        )

        # Volume of box
        vol = (2*dl)**3

        residual = np.abs(
            vol -
            mesh.vol[
                mesh._cell_levels_by_indexes(range(mesh.nC)) == mesh.max_level
            ].sum()
        )/vol * 100

        self.assertTrue(residual < 0.5)

    def test_surface(self):
        dx = 0.1
        dl = 20

        # Define triangle
        xyz = np.r_[
            np.c_[-dl/2, -dl/2, 0],
            np.c_[dl/2, -dl/2, 0],
            np.c_[dl/2, dl/2, 0]
        ]
        mesh = meshutils.mesh_builder_xyz(
            np.c_[0.01, 0.01, 0.01], [dx, dx, dx],
            depth_core=0,
            padding_distance=[[0, 20], [0, 20], [0, 20]],
            mesh_type='TREE',
        )

        mesh = meshutils.refine_tree_xyz(
            mesh, xyz, octree_levels=[1], method='surface', finalize=True
        )

        # Volume of triangle
        vol = dl*dl*dx/2

        residual = np.abs(
            vol -
            mesh.vol[
                mesh._cell_levels_by_indexes(range(mesh.nC)) == mesh.max_level
            ].sum()/2
        )/vol * 100

        self.assertTrue(residual < 5)


if __name__ == '__main__':
    unittest.main()
