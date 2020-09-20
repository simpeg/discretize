from __future__ import print_function
import unittest
import numpy as np

import discretize
from discretize import utils

np.random.seed(16)

TOL = 1e-1


class TestCyl3DGeometries(unittest.TestCase):
    def setUp(self):
        hx = utils.meshTensor([(1, 1)])
        htheta = utils.meshTensor([(1., 4)])
        htheta = htheta * 2*np.pi / htheta.sum()
        hz = hx

        self.mesh = discretize.CylMesh([hx, htheta, hz])

    def test_areas(self):
        area = self.mesh.area
        self.assertTrue(self.mesh.nF == len(area))
        self.assertTrue(
            area[:self.mesh.vnF[0]].sum() == 2*np.pi*self.mesh.hx*self.mesh.hz
        )
        self.assertTrue(
            np.all(
                area[self.mesh.vnF[0]:self.mesh.vnF[1]] ==
                self.mesh.hx*self.mesh.hz
            )
        )
        self.assertTrue(
            np.all(
                area[sum(self.mesh.vnF[:2]):] ==
                np.pi * self.mesh.hx**2 / self.mesh.nCy
            )
        )

    def test_edges(self):
        edge = self.mesh.edge
        self.assertTrue(self.mesh.nE == len(edge))
        self.assertTrue(
            np.all(edge[:self.mesh.vnF[0]] == self.mesh.hx)
        )
        self.assertTrue(
            np.all(
                self.mesh.edge[self.mesh.vnE[0]:sum(self.mesh.vnE[:2])] ==
                np.kron(np.ones(self.mesh.nCz+1), self.mesh.hx*self.mesh.hy)
            )
        )
        self.assertTrue(
            np.all(
                self.mesh.edge[self.mesh.vnE[0]:sum(self.mesh.vnE[:2])] ==
                np.kron(np.ones(self.mesh.nCz+1), self.mesh.hx*self.mesh.hy)
            )
        )
        self.assertTrue(
            np.all(
                self.mesh.edge[sum(self.mesh.vnE[:2]):] ==
                np.kron(self.mesh.hz, np.ones(self.mesh.nCy+1))
            )
        )

    def test_vol(self):
        self.assertTrue(
            self.mesh.vol.sum() == np.pi * self.mesh.hx ** 2 * self.mesh.hz
        )
        self.assertTrue(
            np.all(
                self.mesh.vol ==
                np.pi * self.mesh.hx ** 2 * self.mesh.hz / self.mesh.nCy
            )
        )

# ----------------------- Test Grids and Counting --------------------------- #


class Cyl3DGrid(unittest.TestCase):

    def setUp(self):
        self.mesh = discretize.CylMesh([2, 4, 1])

    def test_counting(self):
        mesh = self.mesh

        # cell centers
        self.assertEqual(mesh.nC, 8)
        self.assertEqual(mesh.nCx, 2)
        self.assertEqual(mesh.nCy, 4)
        self.assertEqual(mesh.nCz, 1)
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
        self.assertEqual(mesh.nNx, 3)
        self.assertEqual(mesh.nNy, 4)
        self.assertEqual(mesh.nNz, 2)
        self.assertEqual(mesh.vnN, (3, 4, 2))
        self.assertEqual(mesh.nN, 18)
        self.assertNotEqual(mesh.nN, np.prod(mesh.vnN))  # periodic boundary condition

    def test_gridCC(self):
        mesh = self.mesh

        # Cell centers
        self.assertTrue((mesh.vectorCCx == [0.25, 0.75]).all())
        self.assertTrue((
            mesh.vectorCCy == 2.*np.pi*np.r_[1./8., 3./8., 5./8., 7./8.]
        ).all())
        self.assertTrue(mesh.vectorCCz == 0.5)

        self.assertTrue((mesh.gridCC[:, 0] == 4*[0.25, 0.75]).all())
        self.assertTrue((
            mesh.gridCC[:, 1] == 2.*np.pi*np.r_[
                1./8., 1./8., 3./8., 3./8., 5./8., 5./8., 7./8., 7./8.
            ]
        ).all())
        self.assertTrue((mesh.gridCC[:, 2] == 8*[0.5]).all())

    def test_gridN(self):
        mesh = self.mesh

        # Nodes
        self.assertTrue((mesh.vectorNx == [0., 0.5, 1.]).all())
        self.assertTrue((
            mesh.vectorNy == 2*np.pi*np.r_[0., 0.25, 0.5, 0.75]
        ).all())
        self.assertTrue((mesh.vectorNz == np.r_[0., 1.]).all())

        self.assertTrue((
            mesh.gridN[:, 0] == 2*[0., 0.5, 1., 0.5, 1., 0.5, 1., 0.5, 1.]
        ).all())
        self.assertTrue((
            mesh.gridN[:, 1] == 2*np.pi*np.hstack(
                2*[3*[0.], 2*[1./4.], 2*[1./2.], 2*[3./4.]]
            )
        ).all())
        self.assertTrue((mesh.gridN[:, 2] == 9*[0.] + 9*[1.]).all())

    def test_gridFx(self):
        mesh = self.mesh

        # x-faces
        self.assertTrue((mesh.gridFx[:, 0] == 4*[0.5, 1.]).all())
        self.assertTrue((
            mesh.gridFx[:, 1] == 2*np.pi*np.hstack(
                [2*[1./8.], 2*[3./8.], 2*[5./8.], 2*[7./8.]]
            )
        ).all())
        self.assertTrue((mesh.gridFx[:, 2] == 8*[0.5]).all())

    def test_gridFy(self):
        mesh = self.mesh

        # y-faces
        self.assertTrue((mesh.gridFy[:, 0] == 4*[0.25, 0.75]).all())
        self.assertTrue((
            mesh.gridFy[:, 1] == 2*np.pi*np.hstack(
                [2*[0.], 2*[1./4.], 2*[1./2.], 2*[3./4.]]
            )
        ).all())
        self.assertTrue((mesh.gridFy[:, 2] == 8*[0.5]).all())

    def test_gridFz(self):
        mesh = self.mesh

        # z-faces
        self.assertTrue((mesh.gridFz[:, 0] == 8*[0.25, 0.75]).all())
        self.assertTrue((
            mesh.gridFz[:, 1] == 2*np.pi*np.hstack(
                2*[2*[1./8.], 2*[3./8.], 2*[5./8.], 2*[7./8.]]
            )
        ).all())
        self.assertTrue((
            mesh.gridFz[:, 2] == np.hstack([8*[0.], 8*[1.]])
        ).all())

    def test_gridEx(self):
        mesh = self.mesh

        # x-edges
        self.assertTrue((mesh.gridEx[:, 0] == 8*[0.25, 0.75]).all())
        self.assertTrue((
            mesh.gridEx[:, 1] == 2*np.pi*np.hstack(
                2*[2*[0.], 2*[1./4.], 2*[1./2.], 2*[3./4.]]
            )
        ).all())
        self.assertTrue((
            mesh.gridEx[:, 2] == np.hstack([8*[0.], 8*[1.]])
        ).all())

    def test_gridEy(self):
        mesh = self.mesh

        # y-edges
        self.assertTrue((mesh.gridEy[:, 0] == 8*[0.5, 1.]).all())
        self.assertTrue((
            mesh.gridEy[:, 1] == 2*np.pi*np.hstack(
                2*[2*[1./8.], 2*[3./8.], 2*[5./8.], 2*[7./8.]]
            )
        ).all())
        self.assertTrue((
            mesh.gridEy[:, 2] == np.hstack([8*[0.], 8*[1.]])
        ).all())

    def test_gridEz(self):
        mesh = self.mesh

        # z-edges
        self.assertTrue((
            mesh.gridEz[:, 0] == np.hstack([[0., 0.5, 1.] + 3*[0.5, 1.]])
        ).all())
        self.assertTrue((
            mesh.gridEz[:, 1] == 2*np.pi*np.hstack(
                [3*[0.], 2*[1./4.], 2*[1./2.], 2*[3./4.]]
            )
        ).all())
        self.assertTrue((mesh.gridEz[:, 2] == 9*[0.5]).all())



# ------------------- Test conversion to Cartesian ----------------------- #

class TestCartesianGrid(unittest.TestCase):

    def test_cartesianGrid(self):
        mesh = discretize.CylMesh([1, 4, 1])

        root2over2 = np.sqrt(2.)/2.

        # cell centers
        cartCC = mesh.cartesianGrid('CC')
        self.assertTrue(np.allclose(
            cartCC[:, 0], 0.5*root2over2*np.r_[1., -1., -1., 1.]
        ))
        self.assertTrue(np.allclose(
            cartCC[:, 1], 0.5*root2over2*np.r_[1., 1., -1., -1.]
        ))
        self.assertTrue(np.allclose(
            cartCC[:, 2], 0.5*np.ones(4)
        ))

        # nodes
        cartN = mesh.cartesianGrid('N')
        self.assertTrue(np.allclose(
            cartN[:, 0], np.hstack(2*[0., 1., 0., -1., 0.])
        ))
        self.assertTrue(np.allclose(
            cartN[:, 1], np.hstack(2*[0., 0., 1., 0., -1.])
        ))
        self.assertTrue(np.allclose(
            cartN[:, 2], np.hstack(5*[0.] + 5*[1.])
        ))


class Deflation(unittest.TestCase):

    def test_areas(self):
        mesh = discretize.CylMesh([1, 2, 1])

        areas = np.hstack([[np.pi]*2, [1]*2, [np.pi/2]*4])
        self.assertTrue(np.all(mesh.area == areas))

        edges = np.hstack([[1]*4, [np.pi]*4, [1]*3])
        self.assertTrue(np.all(mesh.edge == edges))

        mesh = discretize.CylMesh([2, 5, 3])

        hangingF = np.hstack([
            getattr(mesh, '_ishangingF{}'.format(dim))
            for dim in ['x', 'y', 'z']
        ])
        self.assertTrue(np.all(mesh._areaFull[~hangingF] == mesh.area))
        hangingE = np.hstack([
            getattr(mesh, '_ishangingE{}'.format(dim))
            for dim in ['x', 'y', 'z']
        ])
        self.assertTrue(np.all(mesh._edgeFull[~hangingE] == mesh.edge))


if __name__ == '__main__':
    unittest.main()
