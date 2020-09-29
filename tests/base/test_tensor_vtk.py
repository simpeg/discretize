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

    class TestTensorMeshVTK(unittest.TestCase):
        def setUp(self):
            h = np.ones(16)
            mesh = discretize.TensorMesh([h, 2 * h, 3 * h])
            self.mesh = mesh

        def test_VTK_object_conversion(self):
            mesh = self.mesh
            vec = np.arange(mesh.nC)
            models = {"arange": vec}

            vtkObj = mesh.to_vtk(models)

            self.assertEqual(mesh.nC, vtkObj.GetNumberOfCells())
            self.assertEqual(mesh.nN, vtkObj.GetNumberOfPoints())
            self.assertEqual(
                len(models.keys()), vtkObj.GetCellData().GetNumberOfArrays()
            )
            bnds = vtkObj.GetBounds()
            self.assertEqual(mesh.x0[0], bnds[0])
            self.assertEqual(mesh.x0[1], bnds[2])
            self.assertEqual(mesh.x0[2], bnds[4])

            for i in range(vtkObj.GetCellData().GetNumberOfArrays()):
                name = list(models.keys())[i]
                self.assertEqual(name, vtkObj.GetCellData().GetArrayName(i))
                arr = nps.vtk_to_numpy(vtkObj.GetCellData().GetArray(i))
                arr = arr.flatten(order="F")
                self.assertTrue(np.allclose(models[name], arr))


if __name__ == "__main__":
    unittest.main()
