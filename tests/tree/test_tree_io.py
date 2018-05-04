from __future__ import print_function
import numpy as np
import unittest
import os
import discretize
import pickle

try:
    import vtk
except ImportError:
    has_vtk = False
else:
    has_vtk = True


class TestOcTreeMeshIO(unittest.TestCase):

    def setUp(self):
        h = np.ones(16)
        mesh = discretize.TreeMesh([h, 2*h, 3*h])
        cell_points = np.array([[0.5, 0.5, 0.5],
                                [0.5, 2.5, 0.5]])
        cell_levels = np.array([4, 4])
        mesh._insert_cells(cell_points, cell_levels)
        mesh.number()
        self.mesh = mesh

    def test_UBCfiles(self):

        mesh = self.mesh
        # Make a vector
        vec = np.arange(mesh.nC)
        # Write and read
        mesh.writeUBC('temp.msh', {'arange.txt': vec})
        meshUBC = discretize.TreeMesh.readUBC('temp.msh')
        vecUBC = meshUBC.readModelUBC('arange.txt')

        assert(mesh.nC == meshUBC.nC)
        assert mesh.__str__() == meshUBC.__str__()
        assert np.sum(mesh.gridCC - meshUBC.gridCC) == 0
        assert np.sum(vec - vecUBC) == 0
        assert np.all(np.array(mesh.h) - np.array(meshUBC.h) == 0)
        print('IO of UBC octree files is working')
        os.remove('temp.msh')
        os.remove('arange.txt')

    if has_vtk:
        def test_VTUfiles(self):
            mesh = self.mesh
            vec = np.arange(mesh.nC)
            mesh.writeVTK('temp.vtu', {'arange': vec})
            print('Writing of VTU files is working')
            os.remove('temp.vtu')


class TestPickle(unittest.TestCase):

    def test_pickle2D(self):
        mesh0 = discretize.TreeMesh([8, 8])

        def refine(cell):
            xyz = cell.center
            dist = ((xyz - 0.25)**2).sum()**0.5
            if dist < 0.25:
                return 3
            return 2

        mesh0.refine(refine)

        byte_string = pickle.dumps(mesh0)
        mesh1 = pickle.loads(byte_string)

        assert(mesh0.nC == mesh1.nC)
        assert mesh0.__str__() == mesh1.__str__()
        assert np.allclose(mesh0.gridCC, mesh1.gridCC)
        assert np.all(np.array(mesh0.h) - np.array(mesh1.h) == 0)
        print('Pickling of 2D TreeMesh is working')

    def test_pickle3D(self):
        mesh0 = discretize.TreeMesh([8, 8, 8])

        def refine(cell):
            xyz = cell.center
            dist = ((xyz - 0.25)**2).sum()**0.5
            if dist < 0.25:
                return 3
            return 2

        mesh0.refine(refine)

        byte_string = pickle.dumps(mesh0)
        mesh1 = pickle.loads(byte_string)

        assert(mesh0.nC == mesh1.nC)
        assert mesh0.__str__() == mesh1.__str__()
        assert np.allclose(mesh0.gridCC, mesh1.gridCC)
        assert np.all(np.array(mesh0.h) - np.array(mesh1.h) == 0)
        print('Pickling of 3D TreeMesh is working')


if __name__ == '__main__':
    unittest.main()
