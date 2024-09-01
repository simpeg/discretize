import numpy as np
import unittest
import discretize

try:
    import omf
except ImportError:
    has_omf = False
else:
    has_omf = True


if has_omf:
    from discretize.mixins.omf_mod import _unravel_data_array, _ravel_data_array

    class TestTensorMeshOMF(unittest.TestCase):
        def setUp(self):
            h = np.ones(16)
            mesh = discretize.TensorMesh([h, 2 * h, 3 * h])
            self.mesh = mesh

        def test_to_omf(self):
            mesh = self.mesh
            vec = np.arange(mesh.nC)
            models = {"arange": vec}

            omf_element = mesh.to_omf(models)
            geom = omf_element.geometry

            # Check geometry
            self.assertEqual(mesh.nC, geom.num_cells)
            self.assertEqual(mesh.nN, geom.num_nodes)
            self.assertTrue(np.allclose(mesh.h[0], geom.tensor_u))
            self.assertTrue(np.allclose(mesh.h[1], geom.tensor_v))
            self.assertTrue(np.allclose(mesh.h[2], geom.tensor_w))
            self.assertTrue(np.allclose(mesh.orientation[0], geom.axis_u))
            self.assertTrue(np.allclose(mesh.orientation[1], geom.axis_v))
            self.assertTrue(np.allclose(mesh.orientation[2], geom.axis_w))
            self.assertTrue(np.allclose(mesh.x0, geom.origin))

            # Check data arrays
            self.assertEqual(len(models.keys()), len(omf_element.data))
            for i in range(len(omf_element.data)):
                name = list(models.keys())[i]
                scalar_data = omf_element.data[i]
                self.assertEqual(name, scalar_data.name)
                arr = _unravel_data_array(
                    np.array(scalar_data.array), *mesh.shape_cells
                )
                self.assertTrue(np.allclose(models[name], arr))

        def test_from_omf(self):
            rng = np.random.default_rng(52134)
            omf_element = omf.VolumeElement(
                name="vol_ir",
                geometry=omf.VolumeGridGeometry(
                    axis_u=[1, 1, 0],
                    axis_v=[0, 0, 1],
                    axis_w=[1, -1, 0],
                    tensor_u=np.ones(10).astype(float),
                    tensor_v=np.ones(15).astype(float),
                    tensor_w=np.ones(20).astype(float),
                    origin=[10.0, 10.0, -10],
                ),
                data=[
                    omf.ScalarData(
                        name="Random Data",
                        location="cells",
                        array=rng.random((10, 15, 20)).flatten(),
                    )
                ],
            )

            # Make a discretize mesh
            mesh, models = discretize.TensorMesh.from_omf(omf_element)

            geom = omf_element.geometry
            # Check geometry
            self.assertEqual(mesh.nC, geom.num_cells)
            self.assertEqual(mesh.nN, geom.num_nodes)
            self.assertTrue(np.allclose(mesh.h[0], geom.tensor_u))
            self.assertTrue(np.allclose(mesh.h[1], geom.tensor_v))
            self.assertTrue(np.allclose(mesh.h[2], geom.tensor_w))
            self.assertTrue(np.allclose(mesh.orientation[0], geom.axis_u))
            self.assertTrue(np.allclose(mesh.orientation[1], geom.axis_v))
            self.assertTrue(np.allclose(mesh.orientation[2], geom.axis_w))
            self.assertTrue(np.allclose(mesh.x0, geom.origin))

            # Check data arrays
            self.assertEqual(len(models.keys()), len(omf_element.data))
            for i in range(len(omf_element.data)):
                name = list(models.keys())[i]
                scalar_data = omf_element.data[i]
                self.assertEqual(name, scalar_data.name)
                arr = _ravel_data_array(
                    models[name],
                    len(geom.tensor_u),
                    len(geom.tensor_v),
                    len(geom.tensor_w),
                )
                self.assertTrue(np.allclose(np.array(scalar_data.array), arr))


if __name__ == "__main__":
    unittest.main()
