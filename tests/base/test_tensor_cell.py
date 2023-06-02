"""
Test TensorCell
"""
import pytest
import numpy as np

from discretize import TensorCell, TensorMesh


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
            h = [5, 3, 10]
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
