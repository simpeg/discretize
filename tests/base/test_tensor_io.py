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
        mesh = discretize.TensorMesh([h, 2 * h, 3 * h])
        self.mesh = mesh
        self.basePath = os.path.expanduser("~/TestIO")

    def test_write_read_ubc_mesh_model_3d(self):
        if not os.path.exists(self.basePath):
            os.mkdir(self.basePath)

        mesh = self.mesh
        # Make a vector
        vec = np.arange(mesh.nC)
        # Write and read
        mshfname = "temp.msh"
        modelfname = "arange.txt"
        modelfname1 = "arange2.txt"
        modeldict = {modelfname: vec, modelfname1: vec + 1}
        mesh.writeUBC("temp.msh", modeldict, directory=self.basePath)
        meshUBC = discretize.TensorMesh.readUBC(mshfname, directory=self.basePath)
        vecUBC = meshUBC.readModelUBC(modelfname, directory=self.basePath)
        vec2UBC = mesh.readModelUBC(modelfname1, directory=self.basePath)

        # The mesh
        self.assertTrue(mesh.__str__() == meshUBC.__str__())
        self.assertTrue(np.sum(mesh.gridCC - meshUBC.gridCC) == 0)
        self.assertTrue(np.sum(vec - vecUBC) == 0)
        self.assertTrue(np.all(np.array(mesh.h) - np.array(meshUBC.h) == 0))

        # The models
        self.assertTrue(np.sum(vec - vecUBC) == 0)
        self.assertTrue(np.sum(vec + 1 - vec2UBC) == 0)

        print("IO of UBC 3D tensor mesh files is working")
        os.remove(os.path.join(self.basePath, mshfname))
        os.remove(os.path.join(self.basePath, modelfname))
        os.remove(os.path.join(self.basePath, modelfname1))
        os.rmdir(self.basePath)

    if has_vtk:

        def test_VTKfiles(self):
            if not os.path.exists(self.basePath):
                os.mkdir(self.basePath)

            mesh = self.mesh
            vec = np.arange(mesh.nC)
            vtrfname = "temp.vtr"
            modelfname = "arange.txt"
            modeldict = {modelfname: vec}
            mesh.writeVTK(vtrfname, modeldict, directory=self.basePath)
            meshVTR, models = discretize.TensorMesh.read_vtk(
                vtrfname, directory=self.basePath
            )

            self.assertTrue(mesh.__str__() == meshVTR.__str__())
            self.assertTrue(np.all(np.array(mesh.h) - np.array(meshVTR.h) == 0))

            self.assertTrue(modelfname in models)
            vecVTK = models[modelfname]
            self.assertTrue(np.sum(vec - vecVTK) == 0)

            print("IO of VTR tensor mesh files is working")
            os.remove(os.path.join(self.basePath, vtrfname))
            os.rmdir(self.basePath)

    def test_write_read_ubc_mesh_model_2d(self):
        if not os.path.exists(self.basePath):
            os.mkdir(self.basePath)

        fname = "ubc_DC2D_tensor_mesh.msh"
        # Create 2D Mesh
        # Cells size
        csx, csz = 0.25, 0.25
        # Number of core cells in each direction
        ncx, ncz = 123, 41
        # Number of padding cells and expansion rate
        npad, expansion = 6, 1.5
        # Vectors of cell lengthts in each direction
        hx = [(csx, npad, -expansion), (csx, ncx), (csx, npad, expansion)]
        hz = [(csz, npad, -expansion), (csz, ncz)]
        mesh = discretize.TensorMesh([hx, hz], x0="CN")

        # Create 2-cylinders Model Creation
        # Spheres parameters
        x0, z0, r0 = -6.0, -5.0, 3.0
        x1, z1, r1 = 6.0, -5.0, 3.0
        background = -5.0
        sphere0 = -3.0
        sphere1 = -6.0

        # Create Model
        model = background * np.ones(mesh.nC)
        csph = (
            np.sqrt((mesh.gridCC[:, 1] - z0) ** 2.0 + (mesh.gridCC[:, 0] - x0) ** 2.0)
        ) < r0
        model[csph] = sphere0 * np.ones_like(model[csph])
        # Define the sphere limit
        rsph = (
            np.sqrt((mesh.gridCC[:, 1] - z1) ** 2.0 + (mesh.gridCC[:, 0] - x1) ** 2.0)
        ) < r1
        model[rsph] = sphere1 * np.ones_like(model[rsph])
        modeldict = {"2d_2cyl_model": model}

        # Write Mesh and model
        comment_lines = "!comment line\n" + "!again\n" + "!and again\n"
        mesh.writeUBC(
            fname,
            models=modeldict,
            directory=self.basePath,
            comment_lines=comment_lines,
        )

        # Read back mesh and model
        fname = os.path.sep.join([self.basePath, "ubc_DC2D_tensor_mesh.msh"])
        mesh = discretize.TensorMesh.readUBC(fname)
        modelfname = os.path.sep.join([self.basePath, "2d_2cyl_model"])
        readmodel = mesh.readModelUBC(modelfname)
        self.assertTrue(mesh.nCx == 135)
        self.assertTrue(mesh.nCy == 47)
        # spot check a few things in the file
        self.assertTrue(mesh.hx[0] == 2.84765625)
        # The x0 is in a different place (-z)
        self.assertTrue(mesh.x0[-1] == -np.sum(mesh.hy))
        # the z axis is flipped
        self.assertTrue(mesh.hy[0] == 2.84765625)
        self.assertTrue(mesh.hy[-1] == csz)
        self.assertTrue(mesh.dim == 2)
        self.assertTrue(np.all(model == readmodel))

        # Clean up the working directory
        print("IO of UBC 2D tensor mesh files is working")
        os.remove(os.path.join(self.basePath, fname))
        os.remove(os.path.join(self.basePath, modelfname))
        os.rmdir(self.basePath)


if __name__ == "__main__":
    unittest.main()
