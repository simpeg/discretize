"""Test TensorCell."""
import pytest
import numpy as np

from discretize import TensorCell, TensorMesh
from discretize.tensor_mesh import _slice_to_index


@pytest.mark.parametrize(
    "slice_indices, expected_result",
    [
        (slice(None, None, None), range(8)),
        (slice(0, None, None), range(8)),
        (slice(1, None, None), range(1, 8)),
        (slice(None, 4, None), range(4)),
        (slice(None, 8, None), range(8)),
        (slice(None, None, 1), range(8)),
        (slice(None, None, 2), range(0, 8, 2)),
        (slice(None, None, -1), reversed(range(0, 8, 1))),
        (slice(1, 7, -2), reversed(range(1, 7, 2))),
        (slice(1, -1, None), range(1, 7, 1)),
        (slice(1, -2, 2), range(1, 6, 2)),
    ],
)
def test_slice_to_index(slice_indices, expected_result):
    """Test private _slice_to_index function."""
    end = 8
    indices = tuple(i for i in _slice_to_index(slice_indices, end))
    expected_result = tuple(i for i in expected_result)
    assert indices == expected_result


class TestTensorCell:
    """Test attributes of TensorCell."""

    @pytest.fixture(params=("1D", "2D", "3D"))
    def cell(self, request):
        """Sample TensorCell."""
        dim = request.param
        if dim == "1D":
            h = np.array([4.0])
            origin = np.array([-2.0])
            index_unraveled = (1,)
            mesh_shape = (8,)
        elif dim == "2D":
            h = np.array([4.0, 2.0])
            origin = np.array([-2.0, 5.0])
            index_unraveled = (1, 2)
            mesh_shape = (8, 3)
        elif dim == "3D":
            h = np.array([4.0, 2.0, 10.0])
            origin = np.array([-2.0, 5.0, -12.0])
            index_unraveled = (1, 2, 3)
            mesh_shape = (8, 3, 10)
        return TensorCell(h, origin, index_unraveled, mesh_shape)

    def test_center(self, cell):
        """Test center property."""
        if cell.dim == 1:
            true_center = (0.0,)
        elif cell.dim == 2:
            true_center = (0.0, 6.0)
        elif cell.dim == 3:
            true_center = (0.0, 6.0, -7.0)
        assert all(cell.center == true_center)

    def test_index(self, cell):
        """Test index property."""
        if cell.dim == 1:
            true_index = 1
        elif cell.dim == 2:
            true_index = 17
        elif cell.dim == 3:
            true_index = 89
        assert cell.index == true_index

    def test_index_unraveled(self, cell):
        """Test index_unraveled property."""
        if cell.dim == 1:
            true_index_unraveled = (1,)
        elif cell.dim == 2:
            true_index_unraveled = (1, 2)
        elif cell.dim == 3:
            true_index_unraveled = (1, 2, 3)
        assert cell.index_unraveled == true_index_unraveled

    def test_bounds(self, cell):
        """Test bounds property."""
        if cell.dim == 1:
            true_bounds = (-2.0, 2.0)
        elif cell.dim == 2:
            true_bounds = (-2.0, 2.0, 5.0, 7.0)
        elif cell.dim == 3:
            true_bounds = (-2.0, 2.0, 5.0, 7.0, -12.0, -2.0)
        assert all(cell.bounds == true_bounds)

    @pytest.mark.parametrize("change_h", (True, False))
    @pytest.mark.parametrize("change_origin", (True, False))
    @pytest.mark.parametrize("change_index", (True, False))
    @pytest.mark.parametrize("change_mesh_shape", (True, False))
    def test_eq(self, cell, change_h, change_origin, change_index, change_mesh_shape):
        h, origin = cell.h, cell.origin
        index_unraveled, mesh_shape = cell.index_unraveled, cell.mesh_shape
        if change_h:
            h = np.array([h_i + 0.1 for h_i in h])
        if change_origin:
            origin = np.array([origin_i + 0.1 for origin_i in origin])
        if change_index:
            index_unraveled = tuple(i - 1 for i in index_unraveled)
        if change_mesh_shape:
            mesh_shape = tuple(i + 1 for i in mesh_shape)
        other_cell = TensorCell(h, origin, index_unraveled, mesh_shape)
        if any((change_origin, change_h, change_index, change_mesh_shape)):
            assert cell != other_cell
        else:
            assert cell == other_cell

    def test_eq_invalid_type(self, cell):
        """Test if error is raised when comparing other class to a TensorCell."""

        class Dummy:
            def __init__(self):
                pass

        other_object = Dummy()
        msg = "Cannot compare an object of type 'Dummy'"
        with pytest.raises(TypeError, match=msg):
            cell == other_object  # noqa: B015


class TestTensorMeshCells:
    """Test TensorMesh iterator and its resulting cells."""

    @pytest.fixture(params=("1D", "2D", "3D"))
    def mesh(self, request):
        """Sample TensorMesh."""
        dim = request.param
        if dim == "1D":
            h = [5]
            origin = [-2.0]
        elif dim == "2D":
            h = [5, 4]
            origin = [-2.0, 5.0]
        elif dim == "3D":
            h = [5, 4, 10]
            origin = [-2.0, 5.0, -12.0]
        return TensorMesh(h, origin)

    def test_cell_centers(self, mesh):
        """Test if cells in iterator are properly ordered by comparing cell centers."""
        cell_centers = np.array([cell.center for cell in mesh])
        if mesh.dim == 1:
            # Ravel cell_centers if mesh is 1D
            cell_centers = cell_centers.ravel()
        np.testing.assert_allclose(mesh.cell_centers, cell_centers)

    def test_cell_bounds(self, mesh):
        """Test if cells in iterator are properly ordered by comparing cell bounds."""
        if mesh.dim == 1:
            x1 = mesh.cell_centers - mesh.h_gridded.ravel() / 2
            x2 = mesh.cell_centers + mesh.h_gridded.ravel() / 2
            true_bounds = np.vstack([x1, x2]).T
        elif mesh.dim == 2:
            x1 = mesh.cell_centers[:, 0] - mesh.h_gridded[:, 0] / 2
            x2 = mesh.cell_centers[:, 0] + mesh.h_gridded[:, 0] / 2
            y1 = mesh.cell_centers[:, 1] - mesh.h_gridded[:, 1] / 2
            y2 = mesh.cell_centers[:, 1] + mesh.h_gridded[:, 1] / 2
            true_bounds = np.vstack([x1, x2, y1, y2]).T
        elif mesh.dim == 3:
            x1 = mesh.cell_centers[:, 0] - mesh.h_gridded[:, 0] / 2
            x2 = mesh.cell_centers[:, 0] + mesh.h_gridded[:, 0] / 2
            y1 = mesh.cell_centers[:, 1] - mesh.h_gridded[:, 1] / 2
            y2 = mesh.cell_centers[:, 1] + mesh.h_gridded[:, 1] / 2
            z1 = mesh.cell_centers[:, 2] - mesh.h_gridded[:, 2] / 2
            z2 = mesh.cell_centers[:, 2] + mesh.h_gridded[:, 2] / 2
            true_bounds = np.vstack([x1, x2, y1, y2, z1, z2]).T
        cell_bounds = np.array([cell.bounds for cell in mesh])
        np.testing.assert_allclose(true_bounds, cell_bounds)

    def test_cell_int_indices(self, mesh):
        """
        Test if integer indices return the expected cell.

        Test if an integer index is correctly converted to a tuple of indices,
        i.e. unravelling using FORTRAN order.
        """
        if mesh.dim == 1:
            size = len(mesh)
            indices_tuples = [(i,) for i in range(size)]
            for i in range(len(mesh)):
                cell, expected_cell = mesh[i], mesh[indices_tuples[i]]
                assert cell == expected_cell
        elif mesh.dim == 2:
            shape = mesh.shape_cells
            indices_tuples = [(i, j) for j in range(shape[1]) for i in range(shape[0])]
            for i in range(len(mesh)):
                cell, expected_cell = mesh[i], mesh[indices_tuples[i]]
                assert cell == expected_cell
        elif mesh.dim == 3:
            shape = mesh.shape_cells
            indices_tuples = [
                (i, j, k)
                for k in range(shape[2])
                for j in range(shape[1])
                for i in range(shape[0])
            ]
            cells = [mesh[i] for i in range(len(mesh))]
            expected_cells = [mesh[indices] for indices in indices_tuples]
            assert cells == expected_cells

    def test_cell_negative_int_indices(self, mesh):
        """Test if negative integer indices return the expected cell."""
        if mesh.dim == 1:
            assert mesh[-1] == mesh[5 - 1]
            assert mesh[-2] == mesh[5 - 2]
        elif mesh.dim == 2:
            assert mesh[-1] == mesh[5 * 4 - 1]
            assert mesh[-2] == mesh[5 * 4 - 2]
            assert mesh[-1, 0] == mesh[5 - 1, 0]
            assert mesh[0, -1] == mesh[0, 4 - 1]
            assert mesh[-2, -2] == mesh[5 - 2, 4 - 2]
        elif mesh.dim == 3:
            assert mesh[-1] == mesh[5 * 4 * 10 - 1]
            assert mesh[-2] == mesh[5 * 4 * 10 - 2]
            assert mesh[-1, 0, 0] == mesh[5 - 1, 0, 0]
            assert mesh[0, -1, 0] == mesh[0, 4 - 1, 0]
            assert mesh[0, 0, -1] == mesh[0, 0, 10 - 1]
            assert mesh[-2, -2, -2] == mesh[5 - 2, 4 - 2, 10 - 2]

    @pytest.mark.parametrize("start", [None, 0, 1, -2])
    @pytest.mark.parametrize("stop", [None, 4, -1, "end"])
    @pytest.mark.parametrize("step", [None, 1, 2, -1])
    def test_cells_single_slice(self, mesh, start, stop, step):
        """Test if a single slice return the expected cells."""
        if stop == "end":
            stop = len(mesh)
        cells = mesh[start:stop:step]
        indices = _slice_to_index(slice(start, stop, step), len(mesh))
        expected_cells = [mesh[i] for i in indices]
        assert cells == expected_cells

    @pytest.mark.parametrize("step", (None, 1, 2, -1))
    def test_cells_slices(self, mesh, step):
        """Test if passing slices return the expected cells."""
        start, stop = 1, 3
        if mesh.dim == 1:
            cells = mesh[start:stop:step]
            expected_cells = [
                mesh[i] for i in _slice_to_index(slice(start, stop, step), len(mesh))
            ]
            assert cells == expected_cells
        elif mesh.dim == 2:
            index_x = slice(start, stop, step)
            index_y = slice(start, stop, step)
            cells = mesh[index_x, index_y]
            expected_cells = self.generate_expected_cells(mesh, start, stop, step)
            assert cells == expected_cells
        elif mesh.dim == 3:
            index_x = slice(start, stop, step)
            index_y = slice(start, stop, step)
            index_z = slice(start, stop, step)
            cells = mesh[index_x, index_y, index_z]
            expected_cells = self.generate_expected_cells(mesh, start, stop, step)
            assert cells == expected_cells

    @pytest.mark.parametrize("step", (None, 1, 2, -1))
    def test_cells_slices_negative_bounds(self, mesh, step):
        """Test if passing slices with negative bounds return the expected cells."""
        if mesh.dim == 1:
            n = mesh.n_cells
            assert mesh[1:-1:step] == mesh[1 : n - 1 : step]
            assert mesh[-3:-1:step] == mesh[n - 3 : n - 1 : step]
        elif mesh.dim == 2:
            nx, ny = mesh.shape_cells
            assert (
                mesh[1:-1:step, 1:-1:step] == mesh[1 : nx - 1 : step, 1 : ny - 1 : step]
            )
            assert (
                mesh[-3:-1:step, -3:-1:step]
                == mesh[nx - 3 : nx - 1 : step, ny - 3 : ny - 1 : step]
            )
        elif mesh.dim == 3:
            nx, ny, nz = mesh.shape_cells
            assert (
                mesh[1:-1:step, 1:-1:step, 1:-1:step]
                == mesh[1 : nx - 1 : step, 1 : ny - 1 : step, 1 : nz - 1 : step]
            )
            assert (
                mesh[-3:-1:step, -3:-1:step, -3:-1:step]
                == mesh[
                    nx - 3 : nx - 1 : step,
                    ny - 3 : ny - 1 : step,
                    nz - 3 : nz - 1 : step,
                ]
            )

    def generate_expected_cells(self, mesh, start, stop, step):
        """Generate expected cells after slicing the mesh."""
        if step is None:
            step = 1
        if mesh.dim == 2:
            if step > 0:
                expected_cells = [
                    mesh[i, j]
                    for j in range(start, stop, step)
                    for i in range(start, stop, step)
                ]
            else:
                expected_cells = [
                    mesh[i, j]
                    for j in reversed(range(start, stop, -step))
                    for i in reversed(range(start, stop, -step))
                ]
        elif mesh.dim == 3:
            if step > 0:
                expected_cells = [
                    mesh[i, j, k]
                    for k in range(start, stop, step)
                    for j in range(start, stop, step)
                    for i in range(start, stop, step)
                ]
            else:
                expected_cells = [
                    mesh[i, j, k]
                    for k in reversed(range(start, stop, -step))
                    for j in reversed(range(start, stop, -step))
                    for i in reversed(range(start, stop, -step))
                ]
        return expected_cells


class TestNeighbors:
    """Test the neighbors property."""

    @pytest.fixture
    def sample_1D(self):
        """Cell attributes for building a 1D cell."""
        h = [3.1]
        origin = [-2.3]
        mesh_shape = [5]
        return h, origin, mesh_shape

    @pytest.fixture
    def sample_2D(self):
        """Cell attributes for building a 2D cell."""
        h = [3.1, 5.6]
        origin = [-2.3, 4.1]
        mesh_shape = [5, 4]
        return h, origin, mesh_shape

    @pytest.fixture
    def sample_3D(self):
        """Cell attributes for building a 3D cell."""
        h = [3.1, 5.6, 10.2]
        origin = [-2.3, 4.1, -3.4]
        mesh_shape = [5, 4, 10]
        return h, origin, mesh_shape

    @pytest.mark.parametrize("index", (0, 3, 4))
    def test_neighbors_1D(self, sample_1D, index):
        """Test the neighbors property on a 1D mesh."""
        h, origin, mesh_shape = sample_1D
        cell = TensorCell(
            h=h, origin=origin, index_unraveled=[index], mesh_shape=mesh_shape
        )
        expected_neighbors = []
        if index == 0:
            expected_neighbors = [[1]]
        elif index == 3:
            expected_neighbors = [[2], [4]]
        elif index == 4:
            expected_neighbors = [[3]]
        assert expected_neighbors == cell.neighbors

    @pytest.mark.parametrize("index_x", (0, 3, 4))
    @pytest.mark.parametrize("index_y", (0, 1, 3))
    def test_neighbors_2D(self, sample_2D, index_x, index_y):
        """Test the neighbors property on a 2D mesh."""
        h, origin, mesh_shape = sample_2D
        cell = TensorCell(
            h=h,
            origin=origin,
            index_unraveled=[index_x, index_y],
            mesh_shape=mesh_shape,
        )
        cell_index = cell.index_unraveled
        expected_neighbors = []
        if index_x == 0:
            expected_neighbors += [[1, cell_index[1]]]
        elif index_x == 3:
            expected_neighbors += [[i, cell_index[1]] for i in (2, 4)]
        elif index_x == 4:
            expected_neighbors += [[3, cell_index[1]]]
        if index_y == 0:
            expected_neighbors += [[cell_index[0], 1]]
        elif index_y == 1:
            expected_neighbors += [[cell_index[0], j] for j in (0, 2)]
        elif index_y == 3:
            expected_neighbors += [[cell_index[0], 2]]
        expected_neighbors = [
            np.ravel_multi_index(index, dims=mesh_shape, order="F")
            for index in expected_neighbors
        ]
        assert expected_neighbors == cell.neighbors

    @pytest.mark.parametrize("index_x", (0, 3, 4))
    @pytest.mark.parametrize("index_y", (0, 1, 3))
    @pytest.mark.parametrize("index_z", (0, 4, 9))
    def test_neighbors_3D(self, sample_3D, index_x, index_y, index_z):
        """Test the neighbors property on a 3D mesh."""
        h, origin, mesh_shape = sample_3D
        cell = TensorCell(
            h=h,
            origin=origin,
            index_unraveled=[index_x, index_y, index_z],
            mesh_shape=mesh_shape,
        )
        cell_index = cell.index_unraveled
        expected_neighbors = []
        if index_x == 0:
            expected_neighbors += [[1, cell_index[1], cell_index[2]]]
        elif index_x == 3:
            expected_neighbors += [[i, cell_index[1], cell_index[2]] for i in (2, 4)]
        elif index_x == 4:
            expected_neighbors += [[3, cell_index[1], cell_index[2]]]
        if index_y == 0:
            expected_neighbors += [[cell_index[0], 1, cell_index[2]]]
        elif index_y == 1:
            expected_neighbors += [[cell_index[0], j, cell_index[2]] for j in (0, 2)]
        elif index_y == 3:
            expected_neighbors += [[cell_index[0], 2, cell_index[2]]]
        if index_z == 0:
            expected_neighbors += [[cell_index[0], cell_index[1], 1]]
        elif index_z == 4:
            expected_neighbors += [[cell_index[0], cell_index[1], k] for k in (3, 5)]
        elif index_z == 9:
            expected_neighbors += [[cell_index[0], cell_index[1], 8]]
        expected_neighbors = [
            np.ravel_multi_index(index, dims=mesh_shape, order="F")
            for index in expected_neighbors
        ]
        assert expected_neighbors == cell.neighbors


class TestNodes:
    """Test the nodes property."""

    @pytest.fixture
    def cell_1D(self):
        """Sample 1D TensorCell."""
        return TensorCell(h=[3.4], origin=[-2.3], index_unraveled=[1], mesh_shape=[3])

    @pytest.fixture
    def cell_2D(self):
        """Sample 1D TensorCell."""
        cell = TensorCell(
            h=[3.4, 4.3], origin=[-2.3, 0.3], index_unraveled=[1, 2], mesh_shape=[3, 4]
        )
        return cell

    @pytest.fixture
    def cell_3D(self):
        """Sample 1D TensorCell."""
        cell = TensorCell(
            h=[3.4, 4.3, 5.6],
            origin=[-2.3, 0.3, 3.1],
            index_unraveled=[1, 2, 3],
            mesh_shape=[3, 4, 5],
        )
        return cell

    def test_nodes_indices_1D(self, cell_1D):
        """Test if nodes property return the expected indices."""
        assert cell_1D.nodes == [1, 2]

    def test_nodes_indices_2D(self, cell_2D):
        """Test if nodes property return the expected indices."""
        assert cell_2D.nodes == [9, 10, 13, 14]

    def test_nodes_indices_3D(self, cell_3D):
        """Test if nodes property return the expected indices."""
        assert cell_3D.nodes == [69, 70, 73, 74, 89, 90, 93, 94]


class TestEdges:
    """Test the edges property."""

    @pytest.fixture
    def mesh_1D(self):
        """Sample 1D TensorMesh."""
        h = [5]
        origin = [-2.0]
        return TensorMesh(h, origin)

    @pytest.fixture
    def mesh_2D(self):
        """Sample 2D TensorMesh."""
        h = [5, 4]
        origin = [-2.0, 5.0]
        return TensorMesh(h, origin)

    @pytest.fixture
    def mesh_3D(self):
        """Sample 3D TensorMesh."""
        h = [5, 4, 10]
        origin = [-2.0, 5.0, -12.0]
        return TensorMesh(h, origin)

    def test_edges_1D(self, mesh_1D):
        """Test the edges property on a 1D mesh."""
        index = (2,)
        cell = mesh_1D[index]
        xmin, xmax = cell.bounds
        true_edges = [(xmin + xmax) / 2]
        edges = [mesh_1D.edges[i] for i in cell.edges]
        assert true_edges == edges

    def test_edges_2D(self, mesh_2D):
        """Test the edges property on a 2D mesh."""
        index = (2, 3)
        cell = mesh_2D[index]
        xmin, xmax, ymin, ymax = cell.bounds
        true_edges = [
            np.array([(xmin + xmax) / 2, ymin]),
            np.array([(xmin + xmax) / 2, ymax]),
            np.array([xmin, (ymin + ymax) / 2]),
            np.array([xmax, (ymin + ymax) / 2]),
        ]
        edges = [mesh_2D.edges[i] for i in cell.edges]
        np.testing.assert_array_equal(true_edges, edges)

    def test_edges_3D(self, mesh_3D):
        """Test the edges property on a 3D mesh."""
        index = (2, 3, 4)
        cell = mesh_3D[index]
        xmin, xmax, ymin, ymax, zmin, zmax = cell.bounds
        true_edges = [
            np.array([(xmin + xmax) / 2, ymin, zmin]),
            np.array([(xmin + xmax) / 2, ymax, zmin]),
            np.array([(xmin + xmax) / 2, ymin, zmax]),
            np.array([(xmin + xmax) / 2, ymax, zmax]),
            np.array([xmin, (ymin + ymax) / 2, zmin]),
            np.array([xmax, (ymin + ymax) / 2, zmin]),
            np.array([xmin, (ymin + ymax) / 2, zmax]),
            np.array([xmax, (ymin + ymax) / 2, zmax]),
            np.array([xmin, ymin, (zmin + zmax) / 2]),
            np.array([xmax, ymin, (zmin + zmax) / 2]),
            np.array([xmin, ymax, (zmin + zmax) / 2]),
            np.array([xmax, ymax, (zmin + zmax) / 2]),
        ]
        edges = [mesh_3D.edges[i] for i in cell.edges]
        np.testing.assert_array_equal(true_edges, edges)


class TestFaces:
    """Test the faces property."""

    @pytest.fixture
    def mesh_1D(self):
        """Sample 1D TensorMesh."""
        h = [5]
        origin = [-2.0]
        return TensorMesh(h, origin)

    @pytest.fixture
    def mesh_2D(self):
        """Sample 2D TensorMesh."""
        h = [5, 4]
        origin = [-2.0, 5.0]
        return TensorMesh(h, origin)

    @pytest.fixture
    def mesh_3D(self):
        """Sample 3D TensorMesh."""
        h = [5, 4, 10]
        origin = [-2.0, 5.0, -12.0]
        return TensorMesh(h, origin)

    def test_faces_1D(self, mesh_1D):
        """Test the faces property on a 1D mesh."""
        index = (2,)
        cell = mesh_1D[index]
        xmin, xmax = cell.bounds
        true_faces = [xmin, xmax]
        faces = [mesh_1D.faces[i] for i in cell.faces]
        assert true_faces == faces

    def test_faces_2D(self, mesh_2D):
        """Test the faces property on a 2D mesh."""
        index = (2, 3)
        cell = mesh_2D[index]
        xmin, xmax, ymin, ymax = cell.bounds
        true_faces = [
            np.array([xmin, (ymin + ymax) / 2]),
            np.array([xmax, (ymin + ymax) / 2]),
            np.array([(xmin + xmax) / 2, ymin]),
            np.array([(xmin + xmax) / 2, ymax]),
        ]
        faces = [mesh_2D.faces[i] for i in cell.faces]
        np.testing.assert_array_equal(true_faces, faces)

    def test_faces_3D(self, mesh_3D):
        """Test the faces property on a 3D mesh."""
        index = (2, 3, 4)
        cell = mesh_3D[index]
        xmin, xmax, ymin, ymax, zmin, zmax = cell.bounds
        true_faces = [
            np.array([xmin, (ymin + ymax) / 2, (zmin + zmax) / 2]),
            np.array([xmax, (ymin + ymax) / 2, (zmin + zmax) / 2]),
            np.array([(xmin + xmax) / 2, ymin, (zmin + zmax) / 2]),
            np.array([(xmin + xmax) / 2, ymax, (zmin + zmax) / 2]),
            np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, zmin]),
            np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, zmax]),
        ]
        faces = [mesh_3D.faces[i] for i in cell.faces]
        np.testing.assert_array_equal(true_faces, faces)
