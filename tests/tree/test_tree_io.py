import warnings
import numpy as np
import discretize
import pickle
import json
import pytest

try:
    import vtk  # NOQA F401

    has_vtk = True
except ImportError:
    has_vtk = False


@pytest.fixture(params=[2, 3])
def mesh(request):
    dim = request.param
    if dim == 2:
        mesh = discretize.TreeMesh([8, 8])
        mesh.refine(2, finalize=False)
        mesh.refine_ball([0.25, 0.25], 0.25, 3)
    else:
        h = np.ones(16)
        mesh = discretize.TreeMesh([h, 2 * h, 3 * h])
        cell_points = np.array([[0.5, 0.5, 0.5], [0.5, 2.5, 0.5]])
        cell_levels = np.array([4, 4])
        mesh.insert_cells(cell_points, cell_levels)
    return mesh


def test_UBCfiles(mesh, tmp_path):
    # Make a vector
    vec = np.arange(mesh.n_cells)
    # Write and read
    mesh_file = tmp_path / "temp.msh"
    model_file = tmp_path / "arange.txt"

    mesh.write_UBC(mesh_file, {model_file: vec})
    meshUBC = discretize.TreeMesh.read_UBC(mesh_file)
    vecUBC = meshUBC.read_model_UBC(model_file)

    assert mesh is not meshUBC
    assert mesh.equals(meshUBC)
    np.testing.assert_array_equal(vec, vecUBC)

    # Write it again with another IO function
    mesh.write_model_UBC([model_file], [vec])
    vecUBC2 = mesh.read_model_UBC(model_file)
    np.testing.assert_array_equal(vec, vecUBC2)


def test_ubc_files_no_warning_diagonal_balance(mesh, tmp_path):
    """
    Test that reading UBC files don't trigger the diagonal balance warning.
    """
    # Save the sample mesh into a UBC file
    fname = tmp_path / "temp.msh"
    mesh.write_UBC(fname)
    # Make sure that no warning is raised when reading the mesh
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        discretize.TreeMesh.read_UBC(fname)


if has_vtk:

    def test_write_VTU_files(mesh, tmp_path):
        vec = np.arange(mesh.nC)
        mesh_file = tmp_path / "temp.vtu"
        mesh.write_vtk(mesh_file, {"arange": vec})


def test_pickle(mesh):
    byte_string = pickle.dumps(mesh)
    mesh_pickle = pickle.loads(byte_string)

    assert mesh is not mesh_pickle
    assert mesh.equals(mesh_pickle)


def test_dic_serialize(mesh):
    mesh_dict = mesh.serialize()
    mesh2 = discretize.TreeMesh.deserialize(mesh_dict)
    assert mesh is not mesh2
    assert mesh.equals(mesh2)


def test_json_serialize(mesh, tmp_path):
    json_file = tmp_path / "tree.json"

    mesh.save(json_file)
    with open(json_file, "r") as outfile:
        jsondict = json.load(outfile)
    mesh2 = discretize.TreeMesh.deserialize(jsondict)
    assert mesh is not mesh2
    assert mesh.equals(mesh2)
