import discretize
import pytest
import os
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def test_convert_zero_wrapped_to_vtk():
    mesh = discretize.CylindricalMesh([20, 15, 10])
    mesh.to_vtk()


def test_convert_zero_nonwrapped_to_vtk():
    mesh = discretize.CylindricalMesh([20, np.ones(3), 10])
    mesh.to_vtk()


def test_convert_nonzero_wrapped_to_vtk():
    mesh = discretize.CylindricalMesh([20, 15, 10], origin=[1, 0, 0])
    mesh.to_vtk()


def test_convert_nonzero_nonwrapped_to_vtk():
    mesh = discretize.CylindricalMesh([20, np.ones(3), 10], origin=[1, 0, 0])
    mesh.to_vtk()


@pytest.mark.skipif(plt is None, reason="Requires matplotlib")
def test_convert_zero_wrapped_plot_grid():
    mesh = discretize.CylindricalMesh([20, 15, 10])
    mesh.plot_grid()


@pytest.mark.skipif(plt is None, reason="Requires matplotlib")
def test_convert_zero_nonwrapped_plot_grid():
    mesh = discretize.CylindricalMesh([20, np.ones(3), 10])
    mesh.plot_grid()


@pytest.mark.skipif(plt is None, reason="Requires matplotlib")
def test_convert_nonzero_wrapped_plot_grid():
    mesh = discretize.CylindricalMesh([20, 15, 10], origin=[1, 0, 0])
    mesh.plot_grid()


@pytest.mark.skipif(plt is None, reason="Requires matplotlib")
def test_convert_nonzero_nonwrapped_plot_grid():
    mesh = discretize.CylindricalMesh([20, np.ones(3), 10], origin=[1, 0, 0])
    mesh.plot_grid()


@pytest.fixture(autouse=True)
def cleanup_files(monkeypatch):
    files = ["test.cyl"]
    yield
    for file in files:
        try:
            os.remove(file)
        except FileNotFoundError:
            pass
    if plt is not None:
        plt.close("all")
