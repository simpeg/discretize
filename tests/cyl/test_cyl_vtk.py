from __future__ import print_function
import numpy as np
import unittest
import os
import discretize
from discretize import utils

try:
    import vtk
    import vtk.util.numpy_support as nps
except ImportError:
    has_vtk = False
else:
    has_vtk = True


if has_vtk:
    class TestCylMeshVTK(unittest.TestCase):

        def setUp(self):
            hx = utils.meshTensor([(1, 1)])
            htheta = utils.meshTensor([(1., 4)])
            htheta = htheta * 2*np.pi / htheta.sum()
            hz = hx
            self.mesh = discretize.CylMesh([hx, htheta, hz])

        def test_VTK_object_conversion(self):
            mesh = self.mesh
            vec = np.arange(mesh.nC)
            models = {'arange': vec}

            # TODO:
            # vtkObj = mesh.toVTK(models)
            #
            # self.assertEqual(mesh.nC, vtkObj.GetNumberOfCells())
            # self.assertEqual(mesh.nN, vtkObj.GetNumberOfPoints())
            # self.assertEqual(len(models.keys()), vtkObj.GetCellData().GetNumberOfArrays())
            # bnds = vtkObj.GetBounds()
            # self.assertEqual(mesh.x0[0], bnds[0])
            # self.assertEqual(mesh.x0[1], bnds[2])
            # self.assertEqual(mesh.x0[2], bnds[4])
            #
            # names = list(models.keys())
            # for i in range(vtkObj.GetCellData().GetNumberOfArrays()):
            #     name = names[i]
            #     self.assertEqual(name, vtkObj.GetCellData().GetArrayName(i))
            #     arr = nps.vtk_to_numpy(vtkObj.GetCellData().GetArray(i))
            #     arr = arr.flatten(order='F')
            #     self.assertTrue(np.allclose(models[name], arr))
            return



if __name__ == '__main__':
    unittest.main()
