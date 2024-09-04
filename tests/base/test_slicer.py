import numpy as np
import pytest
from matplotlib.colors import Normalize
import discretize
from discretize.mixins.mpl_mod import Slicer


@pytest.fixture()
def mesh():
    return discretize.TensorMesh([9, 10, 11])


def test_slicer_errors(mesh):
    model = np.ones(mesh.shape_cells)
    with pytest.raises(
        ValueError,
        match="(Passing a Normalize instance simultaneously with clim is not supported).*",
    ):
        Slicer(mesh, model, clim=[0, 1], pcolor_opts={"norm": Normalize()})


def test_slicer_default_clim(mesh):
    model = np.ones(mesh.shape_cells)
    model[0, 0, 0] = 0.5
    slc = Slicer(mesh, model)
    norm = slc.pc_props["norm"]
    assert (norm.vmin, norm.vmax) == (0.5, 1.0)


def test_slicer_set_clim(mesh):
    model = np.ones(mesh.shape_cells)
    slc = Slicer(mesh, model, clim=(0.5, 1.5))
    norm = slc.pc_props["norm"]
    assert (norm.vmin, norm.vmax) == (0.5, 1.5)


def test_slicer_set_norm(mesh):
    model = np.ones(mesh.shape_cells)
    norm = Normalize(0.5, 1.5)
    slc = Slicer(mesh, model, pcolor_opts={"norm": norm})
    norm = slc.pc_props["norm"]
    assert (norm.vmin, norm.vmax) == (0.5, 1.5)


def test_slicer_ones_clim(mesh):
    model = np.ones(mesh.shape_cells)
    slc = Slicer(mesh, model)
    norm = slc.pc_props["norm"]
    assert (norm.vmin, norm.vmax) == (0.9, 1.1)


def test_slicer_zeros_clim(mesh):
    model = np.zeros(mesh.shape_cells)
    slc = Slicer(mesh, model)
    norm = slc.pc_props["norm"]
    assert (norm.vmin, norm.vmax) == (-0.1, 0.1)
