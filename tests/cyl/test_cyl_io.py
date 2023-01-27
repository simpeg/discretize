import discretize
import pytest
import os


def test_convert_to_vtk():
    mesh = discretize.CylindricalMesh([20, 15, 10])
    mesh.to_vtk()


def test_bad_convert():
    mesh = discretize.CylindricalMesh([20, 2, 10])
    with pytest.raises(NotImplementedError):
        mesh.to_vtk()

    mesh = discretize.CylindricalMesh([20, 1, 10])
    with pytest.raises(NotImplementedError):
        mesh.to_vtk()


def test_serialize():
    mesh = discretize.CylindricalMesh([20, 10, 3])
    mesh.save("test.cyl")
    mesh2 = discretize.load_mesh("test.cyl")
    assert mesh.equals(mesh2)


@pytest.fixture(autouse=True)
def cleanup_files(monkeypatch):
    files = ["test.cyl"]
    yield
    for file in files:
        try:
            os.remove(file)
        except FileNotFoundError:
            pass
