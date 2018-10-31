from __future__ import print_function
import numpy as np
import unittest
import discretize

try:
    import vtk.util.numpy_support as nps
except ImportError:
    has_vtk = False
else:
    has_vtk = True


if has_vtk:
    class TestTreeMeshVTK(unittest.TestCase):

        def setUp(self):
            h = np.ones(16)
            mesh = discretize.TreeMesh([h, 2*h, 3*h])
            cell_points = np.array([[0.5, 0.5, 0.5],
                                    [0.5, 2.5, 0.5]])
            cell_levels = np.array([4, 4])
            mesh.insert_cells(cell_points, cell_levels)
            self.mesh = mesh

        def test_VTK_object_conversion(self):
            mesh = self.mesh
            vec = np.arange(mesh.nC)
            models = {'arange': vec}

            vtkObj = mesh.toVTK(models)

            self.assertEqual(mesh.nC, vtkObj.GetNumberOfCells())
            # TODO: this is actually different?: self.assertEqual(mesh.nN, vtkObj.GetNumberOfPoints())
            # Remember that the tree vtk conversion adds an array
            self.assertEqual(len(models.keys())+1, vtkObj.GetCellData().GetNumberOfArrays())
            bnds = vtkObj.GetBounds()
            self.assertEqual(mesh.x0[0], bnds[0])
            self.assertEqual(mesh.x0[1], bnds[2])
            self.assertEqual(mesh.x0[2], bnds[4])

            names = list(models.keys())
            for name in names:
                arr = nps.vtk_to_numpy(vtkObj.GetCellData().GetArray(name))
                arr = arr.flatten(order='F')
                self.assertTrue(np.allclose(models[name], arr))



if __name__ == '__main__':
    unittest.main()
