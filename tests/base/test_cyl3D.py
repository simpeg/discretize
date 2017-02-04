from __future__ import print_function
import unittest
import numpy as np
import discretize
from discretize import Tests, utils

np.random.seed(13)

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
                area[self.mesh.vnF[:2].sum():] ==
                np.pi * self.mesh.hx**2 / self.mesh.nCy
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

MESHTYPES = ['uniformCylMesh']
call2 = lambda fun, xyz: fun(xyz[:, 0], xyz[:, 2])
call3 = lambda fun, xyz: fun(xyz[:, 0], xyz[:, 1], xyz[:, 2])
cyl_row2 = lambda g, xfun, yfun: np.c_[call2(xfun, g), call2(yfun, g)]
cyl_row3 = lambda g, xfun, yfun, zfun: np.c_[call3(xfun, g), call3(yfun, g), call3(zfun, g)]
cylF2 = lambda M, fx, fy: np.vstack((
    cyl_row2(M.gridFx, fx, fy), cyl_row2(M.gridFz, fx, fy)
))
cylF3 = lambda M, fx, fy, fz: np.vstack((
    cyl_row3(M.gridFx, fx, fy, fz),
    cyl_row3(M.gridFy, fx, fy, fz),
    cyl_row3(M.gridFz, fx, fy, fz)
))


class TestFaceDiv3D(Tests.OrderTest):
    name = "FaceDiv"
    meshTypes = MESHTYPES
    meshDimension = 3
    meshSizes = [16, 32, 64]

    def getError(self):

        funR = lambda r, t, z: np.sin(2.*np.pi*r)
        funT = lambda r, t, z: r * np.sin(2*t)
        funZ = lambda r, t, z: np.sin(2.*np.pi*z)

        sol = lambda r, t, z: (
            (2*np.pi*r*np.cos(2*np.pi*r) + np.sin(2*np.pi*r))/r +
            2*np.cos(2*t) +
            2*np.pi*np.cos(2*np.pi*z)
        )

        Fc = cylF3(self.M, funR, funT, funZ)
        # Fc = np.c_[Fc[:, 0], np.zeros(self.M.nF), Fc[:, 1]]
        F = self.M.projectFaceVector(Fc)

        divF = self.M.faceDiv.dot(F)
        divF_ana = call3(sol, self.M.gridCC)

        err = np.linalg.norm((divF-divF_ana), np.inf)
        return err

    def test_order(self):
        self.orderTest()

if __name__ == '__main__':
    unittest.main()
