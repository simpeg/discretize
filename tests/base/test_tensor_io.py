from __future__ import print_function
import numpy as np
import unittest
import os
import discretize

try:
    import vtk
except ImportError:
    has_vtk = False
else:
    has_vtk = True


class TestTensorMeshIO(unittest.TestCase):

    def setUp(self):
        h = np.ones(16)
        mesh = discretize.TensorMesh([h, 2*h, 3*h])
        self.mesh = mesh

    def test_UBCfiles(self):

        mesh = self.mesh
        # Make a vector
        vec = np.arange(mesh.nC)
        # Write and read
        mesh.writeUBC('temp.msh', {'arange.txt': vec})
        meshUBC = discretize.TensorMesh.readUBC('temp.msh')
        vecUBC = meshUBC.readModelUBC('arange.txt')

        # The mesh
        assert mesh.__str__() == meshUBC.__str__()
        assert np.sum(mesh.gridCC - meshUBC.gridCC) == 0
        assert np.sum(vec - vecUBC) == 0
        assert np.all(np.array(mesh.h) - np.array(meshUBC.h) == 0)

        vecUBC = mesh.readModelUBC('arange.txt')
        assert np.sum(vec - vecUBC) == 0

        mesh.writeModelUBC('arange2.txt', vec + 1)
        vec2UBC = mesh.readModelUBC('arange2.txt')
        assert np.sum(vec + 1 - vec2UBC) == 0

        print('IO of UBC tensor mesh files is working')
        os.remove('temp.msh')
        os.remove('arange.txt')
        os.remove('arange2.txt')

    def test_read_ubc_mesh(self):
        fname = os.path.join(os.path.split(__file__)[0], 'ubc_tensor_mesh.msh')
        mesh = discretize.TensorMesh.readUBC(fname)
        assert mesh.nCx == 78
        assert mesh.nCy == 50
        assert mesh.nCz == 51
        # spot check a few things in the file
        assert mesh.hx[0] == 55000
        assert mesh.hy[0] == 70000
        # The x0 is in a different place (-z)
        assert mesh.x0[-1] == 3000 - np.sum(mesh.hz)
        # the z axis is flipped
        assert mesh.hz[0] == 20000
        assert mesh.hz[-1] == 250


    if has_vtk:
        def test_VTKfiles(self):
            mesh = self.mesh
            vec = np.arange(mesh.nC)

            mesh.writeVTK('temp.vtr', {'arange.txt': vec})
            meshVTR, models = discretize.TensorMesh.readVTK('temp.vtr')

            assert mesh.__str__() == meshVTR.__str__()
            assert np.all(np.array(mesh.h) - np.array(meshVTR.h) == 0)

            assert 'arange.txt' in models
            vecVTK = models['arange.txt']
            assert np.sum(vec - vecVTK) == 0

            print('IO of VTR tensor mesh files is working')
            os.remove('temp.vtr')

    def test_read_ubc_DC2Dmesh(self):
        fname = os.path.join(os.path.split(__file__)[0], 'ubc_DC2D_tensor_mesh.msh')
        mesh = discretize.TensorMesh.readUBC_DC2DMesh(fname)
        assert mesh.nCx == 178
        assert mesh.nCy == 67
        # spot check a few things in the file
        assert mesh.hx[0] == 600.
        # The x0 is in a different place (-z)
        assert mesh.x0[-1] == - np.sum(mesh.hy)
        # the z axis is flipped
        assert mesh.hy[0] == 600.
        assert mesh.hy[-1] == 10.

if __name__ == '__main__':
    unittest.main()
