"""
Test TensorCell
"""
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
    ],
)
def test_slice_to_index(slice_indices, expected_result):
    """Test private _slice_to_index function"""
    end = 8
    indices = tuple(i for i in _slice_to_index(slice_indices, end))
    expected_result = tuple(i for i in expected_result)
    assert indices == expected_result


@pytest.mark.parametrize(
    "h, origin",
    [
        ((1.0,), (2.0, 3.0)),
        ((1.0, 2.0), (3.0,)),
        ((1.0, 2.0, 4.0), (3.0, 5.0)),
    ],
)
def test_invalid_h_origin(h, origin):
    """Test if error is raised after invalid h and origin arguments"""
    with pytest.raises(ValueError, match="Invalid h and origin"):
        TensorCell(h, origin)


class TestTensorCell:
    @pytest.fixture(params=("1D", "2D", "3D"))
    def cell(self, request):
        """
        Sample TensorCell
        """
        dim = request.param
        if dim == "1D":
            h = (4.0,)
            origin = (-2.0,)
        elif dim == "2D":
            h = (4.0, 2.0)
            origin = (-2.0, 5.0)
        elif dim == "3D":
            h = (4.0, 2.0, 10.0)
            origin = (-2.0, 5.0, -12.0)
        return TensorCell(h, origin)

    def test_center(self, cell):
        """Test center property"""
        if cell.dim == 1:
            true_center = (0.0,)
        elif cell.dim == 2:
            true_center = (0.0, 6.0)
        elif cell.dim == 3:
            true_center = (0.0, 6.0, -7.0)
        assert cell.center == true_center

    def test_bounds(self, cell):
        """Test bounds property"""
        if cell.dim == 1:
            true_bounds = (-2.0, 2.0)
        elif cell.dim == 2:
            true_bounds = (-2.0, 2.0, 5.0, 7.0)
        elif cell.dim == 3:
            true_bounds = (-2.0, 2.0, 5.0, 7.0, -12.0, -2.0)
        assert cell.bounds == true_bounds

    @pytest.mark.parametrize(
        "change_h, change_origin",
        [(True, True), (False, False), (True, False), (False, True)],
    )
    def test_eq(self, cell, change_h, change_origin):
        h, origin = cell.h, cell.origin
        if change_h:
            h = [h_i + 0.1 for h_i in h]
        if change_origin:
            origin = [o + 0.1 for o in origin]
        other_cell = TensorCell(h, origin)
        if change_origin or change_h:
            assert cell != other_cell
        else:
            assert cell == other_cell

    def test_eq_invalid_type(self, cell):
        class Dummy:
            def __init__(self):
                pass

        other_object = Dummy()
        msg = "Cannot compare an object of type 'Dummy'"
        with pytest.raises(TypeError, match=msg):
            cell == other_object


class TestTensorMeshCells:
    @pytest.fixture(params=("1D", "2D", "3D"))
    def mesh(self, request):
        """
        Sample TensorMesh
        """
        dim = request.param
        if dim == "1D":
            h = [5]
            origin = [-2.0]
        elif dim == "2D":
            h = [5, 3]
            origin = [-2.0, 5.0]
        elif dim == "3D":
            h = [5, 4, 10]
            origin = [-2.0, 5.0, -12.0]
        return TensorMesh(h, origin)

    def test_cell_centers(self, mesh):
        """Test if cells in iterator are properly ordered by comparing cell centers"""
        cell_centers = np.array([cell.center for cell in mesh])
        if mesh.dim == 1:
            # Ravel cell_centers if mesh is 1D
            cell_centers = cell_centers.ravel()
        np.testing.assert_allclose(mesh.cell_centers, cell_centers)

    def test_cell_bounds(self, mesh):
        """Test if cells in iterator are properly ordered by comparing cell bounds"""
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
        Test if integer indices return the expected cell

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
            for i in range(len(mesh)):
                cell, expected_cell = mesh[i], mesh[indices_tuples[i]]
                assert cell == expected_cell

    @pytest.mark.parametrize("start", [None, 0, 1])
    @pytest.mark.parametrize("stop", [None, 4, "end"])
    @pytest.mark.parametrize("step", [None, 1, 2, -1])
    def test_cells_single_slice(self, mesh, start, stop, step):
        """
        Test if a single slice return the expected cells
        """
        if stop == "end":
            stop = len(mesh)
        cells = mesh[start:stop:step]
        indices = _slice_to_index(slice(start, stop, step), len(mesh))
        expected_cells = [mesh[i] for i in indices]
        assert len(cells) == len(expected_cells)
        for cell, expected_cell in zip(cells, expected_cells):
            assert cell == expected_cell

    @pytest.mark.parametrize("step", (None, 1, 2, -1))
    def test_cells_slices(self, mesh, step):
        """
        Test passing slices return the expected cells
        """
        start, stop = 1, 3
        if mesh.dim == 1:
            cells = mesh[start:stop:step]
            expected_cells = [
                mesh[i] for i in _slice_to_index(slice(start, stop, step), len(mesh))
            ]
            assert len(cells) == len(expected_cells)
            for cell, expected_cell in zip(cells, expected_cells):
                assert cell == expected_cell
        elif mesh.dim == 2:
            index_x = slice(start, stop, step)
            index_y = slice(start, stop, step)
            cells = mesh[index_x, index_y]
            expected_cells = self.generate_expected_cells(mesh, start, stop, step)
            assert len(cells) == len(expected_cells)
            for cell, expected_cell in zip(cells, expected_cells):
                assert cell == expected_cell
        elif mesh.dim == 3:
            index_x = slice(start, stop, step)
            index_y = slice(start, stop, step)
            index_z = slice(start, stop, step)
            cells = mesh[index_x, index_y, index_z]
            expected_cells = self.generate_expected_cells(mesh, start, stop, step)
            assert len(cells) == len(expected_cells)
            for cell, expected_cell in zip(cells, expected_cells):
                assert cell == expected_cell

    def generate_expected_cells(self, mesh, start, stop, step):
        """
        Generate expected cells after slicing the mesh
        """
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
