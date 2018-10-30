from __future__ import print_function
import numpy as np
import unittest
import os
import discretize

try:
    import vtk
    import vtk.util.numpy_support as nps
except ImportError:
    has_vtk = False
else:
    has_vtk = True


if has_vtk:
    class TestCurvilinearMeshVTK(unittest.TestCase):

        def setUp(self):
            sz = [16, 16, 16]
            mesh = discretize.CurvilinearMesh(discretize.utils.exampleLrmGrid(sz, 'rotate'))
            self.mesh = mesh

        def test_VTK_object_conversion(self):
            mesh = self.mesh
            vec = np.arange(mesh.nC)
            models = {'arange': vec}

            vtkObj = mesh.toVTK(models)

            self.assertEqual(mesh.nC, vtkObj.GetNumberOfCells())
            self.assertEqual(mesh.nN, vtkObj.GetNumberOfPoints())
            self.assertEqual(len(models.keys()), vtkObj.GetCellData().GetNumberOfArrays())
            bnds = vtkObj.GetBounds()
            self.assertEqual(mesh.x0[0], bnds[0])
            self.assertEqual(mesh.x0[1], bnds[2])
            self.assertEqual(mesh.x0[2], bnds[4])

            names = list(models.keys())
            for i in range(vtkObj.GetCellData().GetNumberOfArrays()):
                name = names[i]
                self.assertEqual(name, vtkObj.GetCellData().GetArrayName(i))
                arr = nps.vtk_to_numpy(vtkObj.GetCellData().GetArray(i))
                arr = arr.flatten(order='F')
                self.assertTrue(np.allclose(models[name], arr))

        def test_VTK_file_IO(self):
            mesh = self.mesh
            vec = np.arange(mesh.nC)
            models = {'arange.txt': vec}
            mesh.writeVTK('temp.vts', models)
            print('Writing of VTK files is working')
            os.remove('temp.vts')




if __name__ == '__main__':
    unittest.main()
