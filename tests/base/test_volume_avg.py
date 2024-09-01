import numpy as np
import unittest
import discretize
from discretize.utils import volume_average
from numpy.testing import assert_array_equal, assert_allclose


class TestVolumeAverage(unittest.TestCase):
    def test_tensor_to_tensor(self):
        rng = np.random.default_rng(68723)
        h1 = rng.random(16)
        h1 /= h1.sum()
        h2 = rng.random(16)
        h2 /= h2.sum()

        h1s = []
        h2s = []
        for i in range(3):
            print(f"Tensor to Tensor {i+1}D: ", end="")
            h1s.append(h1)
            h2s.append(h2)
            mesh1 = discretize.TensorMesh(h1s)
            mesh2 = discretize.TensorMesh(h2s)

            in_put = rng.random(mesh1.nC)
            out_put = np.empty(mesh2.nC)
            # test the three ways of calling...
            out1 = volume_average(mesh1, mesh2, in_put, out_put)
            assert_array_equal(out1, out_put)

            out2 = volume_average(mesh1, mesh2, in_put)
            assert_allclose(out1, out2)

            Av = volume_average(mesh1, mesh2)
            out3 = Av @ in_put
            assert_allclose(out1, out3)

            vol1 = np.sum(mesh1.cell_volumes * in_put)
            vol2 = np.sum(mesh2.cell_volumes * out3)
            print(vol1, vol2)
            self.assertAlmostEqual(vol1, vol2)

    def test_tree_to_tree(self):
        rng = np.random.default_rng(68723)
        h1 = rng.random(16)
        h1 /= h1.sum()
        h2 = rng.random(16)
        h2 /= h2.sum()

        h1s = [h1]
        h2s = [h2]
        insert_1 = [0.25]
        insert_2 = [0.75]
        for i in range(1, 3):
            print(f"Tree to Tree {i+1}D: ", end="")
            h1s.append(h1)
            h2s.append(h2)
            insert_1.append(0.25)
            insert_2.append(0.75)
            mesh1 = discretize.TreeMesh(h1s)
            mesh1.insert_cells([insert_1], [4])
            mesh2 = discretize.TreeMesh(h2s)
            mesh2.insert_cells([insert_2], [4])

            in_put = rng.random(mesh1.nC)
            out_put = np.empty(mesh2.nC)
            # test the three ways of calling...
            out1 = volume_average(mesh1, mesh2, in_put, out_put)
            assert_array_equal(out1, out_put)

            out2 = volume_average(mesh1, mesh2, in_put)
            assert_allclose(out1, out2)

            Av = volume_average(mesh1, mesh2)
            out3 = Av @ in_put
            assert_allclose(out1, out3)

            vol1 = np.sum(mesh1.cell_volumes * in_put)
            vol2 = np.sum(mesh2.cell_volumes * out3)
            print(vol1, vol2)
            self.assertAlmostEqual(vol1, vol2)

    def test_tree_to_tensor(self):
        rng = np.random.default_rng(68723)
        h1 = rng.random(16)
        h1 /= h1.sum()
        h2 = rng.random(16)
        h2 /= h2.sum()

        h1s = [h1]
        h2s = [h2]
        insert_1 = [0.25]
        for i in range(1, 3):
            print(f"Tree to Tensor {i+1}D: ", end="")
            h1s.append(h1)
            h2s.append(h2)
            insert_1.append(0.25)
            mesh1 = discretize.TreeMesh(h1s)
            mesh1.insert_cells([insert_1], [4])
            mesh2 = discretize.TensorMesh(h2s)

            in_put = rng.random(mesh1.nC)
            out_put = np.empty(mesh2.nC)
            # test the three ways of calling...
            out1 = volume_average(mesh1, mesh2, in_put, out_put)
            assert_array_equal(out1, out_put)

            out2 = volume_average(mesh1, mesh2, in_put)
            assert_allclose(out1, out2)

            Av = volume_average(mesh1, mesh2)
            out3 = Av @ in_put
            assert_allclose(out1, out3)

            vol1 = np.sum(mesh1.cell_volumes * in_put)
            vol2 = np.sum(mesh2.cell_volumes * out3)
            print(vol1, vol2)
            self.assertAlmostEqual(vol1, vol2)

    def test_tensor_to_tree(self):
        rng = np.random.default_rng(68723)
        h1 = rng.random(16)
        h1 /= h1.sum()
        h2 = rng.random(16)
        h2 /= h2.sum()

        h1s = [h1]
        h2s = [h2]
        insert_2 = [0.75]
        for i in range(1, 3):
            print(f"Tensor to Tree {i+1}D: ", end="")
            h1s.append(h1)
            h2s.append(h2)
            insert_2.append(0.75)
            mesh1 = discretize.TensorMesh(h1s)
            mesh2 = discretize.TreeMesh(h2s)
            mesh2.insert_cells([insert_2], [4])

            in_put = rng.random(mesh1.nC)
            out_put = np.empty(mesh2.nC)
            # test the three ways of calling...
            out1 = volume_average(mesh1, mesh2, in_put, out_put)
            assert_array_equal(out1, out_put)

            out2 = volume_average(mesh1, mesh2, in_put)
            assert_allclose(out1, out2)

            Av = volume_average(mesh1, mesh2)
            out3 = Av @ in_put
            assert_allclose(out1, out3)

            vol1 = np.sum(mesh1.cell_volumes * in_put)
            vol2 = np.sum(mesh2.cell_volumes * out3)
            print(vol1, vol2)
            self.assertAlmostEqual(vol1, vol2)

    def test_errors(self):
        rng = np.random.default_rng(68723)
        h1 = rng.random(16)
        h1 /= h1.sum()
        h2 = rng.random(16)
        h2 /= h2.sum()
        mesh1D = discretize.TensorMesh([h1])
        mesh2D = discretize.TensorMesh([h1, h1])
        mesh3D = discretize.TensorMesh([h1, h1, h1])

        hr = np.r_[1, 1, 0.5]
        hz = np.r_[2, 1]
        meshCyl = discretize.CylindricalMesh([hr, 1, hz], np.r_[0.0, 0.0, 0.0])
        mesh2 = discretize.TreeMesh([h2, h2])
        mesh2.insert_cells([0.75, 0.75], [4])

        with self.assertRaises(TypeError):
            # Gives a wrong typed object to the function
            volume_average(mesh1D, h1)
        with self.assertRaises(NotImplementedError):
            # Gives a wrong typed mesh
            volume_average(meshCyl, mesh2)
        with self.assertRaises(ValueError):
            # Gives mismatching mesh dimensions
            volume_average(mesh2D, mesh3D)

        model1 = rng.standard_normal(mesh2D.nC)
        bad_model1 = rng.standard_normal(3)
        bad_model2 = rng.random(1)
        # gives input values with incorrect lengths
        with self.assertRaises(ValueError):
            volume_average(mesh2D, mesh2, bad_model1)
        with self.assertRaises(ValueError):
            volume_average(mesh2D, mesh2, model1, bad_model2)

    def test_tree_to_tree_same_base(self):
        rng = np.random.default_rng(68723)
        h1 = rng.random(16)
        h1 /= h1.sum()

        h1s = [h1]
        insert_1 = [0.25]
        insert_2 = [0.75]
        for i in range(1, 3):
            print(f"Tree to Tree {i+1}D: same base", end="")
            h1s.append(h1)
            insert_1.append(0.25)
            insert_2.append(0.75)
            mesh1 = discretize.TreeMesh(h1s)
            mesh1.insert_cells([insert_1], [4])
            mesh2 = discretize.TreeMesh(h1s)
            mesh2.insert_cells([insert_2], [4])

            in_put = rng.random(mesh1.nC)
            out_put = np.empty(mesh2.nC)
            # test the three ways of calling...
            out1 = volume_average(mesh1, mesh2, in_put, out_put)
            assert_array_equal(out1, out_put)

            out2 = volume_average(mesh1, mesh2, in_put)
            assert_allclose(out1, out2)

            Av = volume_average(mesh1, mesh2)
            out3 = Av @ in_put
            assert_allclose(out1, out3)

            vol1 = np.sum(mesh1.cell_volumes * in_put)
            vol2 = np.sum(mesh2.cell_volumes * out3)
            print(vol1, vol2)
            self.assertAlmostEqual(vol1, vol2)

    def test_tree_to_tensor_same_base(self):
        rng = np.random.default_rng(867532)
        h1 = rng.random(16)
        h1 /= h1.sum()

        h1s = [h1]
        insert_1 = [0.25]
        for i in range(1, 3):
            print(f"Tree to Tensor {i+1}D same base: ", end="")
            h1s.append(h1)
            insert_1.append(0.25)
            mesh1 = discretize.TreeMesh(h1s)
            mesh1.insert_cells([insert_1], [4])
            mesh2 = discretize.TensorMesh(h1s)

            in_put = rng.random(mesh1.nC)
            out_put = np.empty(mesh2.nC)
            # test the three ways of calling...
            out1 = volume_average(mesh1, mesh2, in_put, out_put)
            assert_array_equal(out1, out_put)

            out2 = volume_average(mesh1, mesh2, in_put)
            assert_allclose(out1, out2)

            Av = volume_average(mesh1, mesh2)
            out3 = Av @ in_put
            assert_allclose(out1, out3)

            vol1 = np.sum(mesh1.cell_volumes * in_put)
            vol2 = np.sum(mesh2.cell_volumes * out3)
            print(vol1, vol2)
            self.assertAlmostEqual(vol1, vol2)

    def test_tensor_to_tree_same_base(self):
        rng = np.random.default_rng(91)
        h1 = rng.random(16)
        h1 /= h1.sum()

        h1s = [h1]
        insert_2 = [0.75]
        for i in range(1, 3):
            print(f"Tensor to Tree {i+1}D same base: ", end="")
            h1s.append(h1)
            insert_2.append(0.75)
            mesh1 = discretize.TensorMesh(h1s)
            mesh2 = discretize.TreeMesh(h1s)
            mesh2.insert_cells([insert_2], [4])

            in_put = rng.random(mesh1.nC)
            out_put = np.empty(mesh2.nC)
            # test the three ways of calling...
            out1 = volume_average(mesh1, mesh2, in_put, out_put)
            assert_array_equal(out1, out_put)

            out2 = volume_average(mesh1, mesh2, in_put)
            assert_allclose(out1, out2)

            Av = volume_average(mesh1, mesh2)
            out3 = Av @ in_put
            assert_allclose(out1, out3)

            vol1 = np.sum(mesh1.cell_volumes * in_put)
            vol2 = np.sum(mesh2.cell_volumes * out3)
            print(vol1, vol2)
            self.assertAlmostEqual(vol1, vol2)

    def test_tensor_to_tensor_sub(self):
        rng = np.random.default_rng(9867153)
        h1 = np.ones(32)
        h2 = np.ones(16)

        h1s = []
        h2s = []
        for i in range(3):
            print(f"Tensor to smaller Tensor {i+1}D: ", end="")
            h1s.append(h1)
            h2s.append(h2)
            mesh1 = discretize.TensorMesh(h1s)
            mesh2 = discretize.TensorMesh(h2s)

            in_put = rng.random(mesh1.nC)
            out_put = np.empty(mesh2.nC)
            # test the three ways of calling...
            out1 = volume_average(mesh1, mesh2, in_put, out_put)
            assert_array_equal(out1, out_put)

            out2 = volume_average(mesh1, mesh2, in_put)
            assert_allclose(out1, out2)

            Av = volume_average(mesh1, mesh2)
            out3 = Av @ in_put
            assert_allclose(out1, out3)

            # get cells in extent of smaller mesh
            cells = mesh1.gridCC < [16] * (i + 1)
            if i > 0:
                cells = np.all(cells, axis=1)

            vol1 = np.sum(mesh1.cell_volumes[cells] * in_put[cells])
            vol2 = np.sum(mesh2.cell_volumes * out3)
            print(vol1, vol2)
            self.assertAlmostEqual(vol1, vol2)

    def test_tree_to_tree_sub(self):
        rng = np.random.default_rng(987263)
        h1 = np.ones(32)
        h2 = np.ones(16)

        h1s = [h1]
        h2s = [h2]
        insert_1 = [12]
        insert_2 = [4]
        for i in range(1, 3):
            print(f"Tree to smaller Tree {i+1}D: ", end="")
            h1s.append(h1)
            h2s.append(h2)
            insert_1.append(12)
            insert_2.append(4)
            mesh1 = discretize.TreeMesh(h1s)
            mesh1.insert_cells([insert_1], [4])
            mesh2 = discretize.TreeMesh(h2s)
            mesh2.insert_cells([insert_2], [4])

            in_put = rng.random(mesh1.nC)
            out_put = np.empty(mesh2.nC)
            # test the three ways of calling...
            out1 = volume_average(mesh1, mesh2, in_put, out_put)
            assert_array_equal(out1, out_put)

            out2 = volume_average(mesh1, mesh2, in_put)
            assert_allclose(out1, out2)

            Av = volume_average(mesh1, mesh2)
            out3 = Av @ in_put
            assert_allclose(out1, out3)

            # get cells in extent of smaller mesh
            cells = mesh1.gridCC < [16] * (i + 1)
            if i > 0:
                cells = np.all(cells, axis=1)

            vol1 = np.sum(mesh1.cell_volumes[cells] * in_put[cells])
            vol2 = np.sum(mesh2.cell_volumes * out3)
            print(vol1, vol2)
            self.assertAlmostEqual(vol1, vol2)

    def test_tree_to_tensor_sub(self):
        rng = np.random.default_rng(5)
        h1 = np.ones(32)
        h2 = np.ones(16)

        h1s = [h1]
        insert_1 = [12]
        h2s = [h2]
        for i in range(1, 3):
            print(f"Tree to smaller Tensor {i+1}D: ", end="")
            h1s.append(h1)
            h2s.append(h2)
            insert_1.append(12)
            mesh1 = discretize.TreeMesh(h1s)
            mesh1.insert_cells([insert_1], [4])
            mesh2 = discretize.TensorMesh(h2s)

            in_put = rng.random(mesh1.nC)
            out_put = np.empty(mesh2.nC)
            # test the three ways of calling...
            out1 = volume_average(mesh1, mesh2, in_put, out_put)
            assert_array_equal(out1, out_put)

            out2 = volume_average(mesh1, mesh2, in_put)
            assert_allclose(out1, out2)

            Av = volume_average(mesh1, mesh2)
            out3 = Av @ in_put
            assert_allclose(out1, out3)

            # get cells in extent of smaller mesh
            cells = mesh1.gridCC < [16] * (i + 1)
            if i > 0:
                cells = np.all(cells, axis=1)

            vol1 = np.sum(mesh1.cell_volumes[cells] * in_put[cells])
            vol2 = np.sum(mesh2.cell_volumes * out3)
            print(vol1, vol2)
            self.assertAlmostEqual(vol1, vol2)

    def test_tensor_to_tree_sub(self):
        rng = np.random.default_rng(1)
        h1 = np.ones(32)
        h2 = np.ones(16)

        h1s = [h1]
        h2s = [h2]
        insert_2 = [4]
        for i in range(1, 3):
            print(f"Tensor to smaller Tree {i+1}D: ", end="")
            h1s.append(h1)
            h2s.append(h2)
            insert_2.append(4)
            mesh1 = discretize.TensorMesh(h1s)
            mesh2 = discretize.TreeMesh(h2s)
            mesh2.insert_cells([insert_2], [4])

            in_put = rng.random(mesh1.nC)
            out_put = np.empty(mesh2.nC)
            # test the three ways of calling...
            out1 = volume_average(mesh1, mesh2, in_put, out_put)
            assert_array_equal(out1, out_put)

            out2 = volume_average(mesh1, mesh2, in_put)
            assert_allclose(out1, out2)

            Av = volume_average(mesh1, mesh2)
            out3 = Av @ in_put
            assert_allclose(out1, out3)

            # get cells in extent of smaller mesh
            cells = mesh1.gridCC < [16] * (i + 1)
            if i > 0:
                cells = np.all(cells, axis=1)

            vol1 = np.sum(mesh1.cell_volumes[cells] * in_put[cells])
            vol2 = np.sum(mesh2.cell_volumes * out3)
            print(vol1, vol2)
            self.assertAlmostEqual(vol1, vol2)


if __name__ == "__main__":
    unittest.main()
