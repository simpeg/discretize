import numpy as np
import discretize
import pytest

try:
    import vtk  # NOQA F401

    has_vtk = True
except ImportError:
    has_vtk = False


@pytest.mark.parametrize("dim", [2, 3])
def test_write_read_ubc_mesh_model(dim, tmp_path):
    h = np.ones(16)
    mesh = discretize.TensorMesh([h, 2 * h, 3 * h][:dim])
    # Make a vector
    vec = np.arange(mesh.nC)
    # Write and read
    mshfname = "temp.msh"
    modelfname = "arange.txt"
    modelfname1 = "arange2.txt"
    modeldict = {modelfname: vec, modelfname1: vec + 1}
    comment_lines = "!comment line\n" + "!again\n" + "!and again\n"

    mesh.write_UBC(
        "temp.msh", modeldict, directory=tmp_path, comment_lines=comment_lines
    )
    meshUBC = discretize.TensorMesh.read_UBC(mshfname, directory=tmp_path)

    assert mesh is not meshUBC
    assert mesh.equals(meshUBC)

    vecUBC = meshUBC.read_model_UBC(modelfname, directory=tmp_path)
    vec2UBC = mesh.read_model_UBC(modelfname1, directory=tmp_path)

    np.testing.assert_equal(vec, vecUBC)
    np.testing.assert_equal(vec + 1, vec2UBC)


if has_vtk:

    def test_VTKfiles(tmp_path):
        h = np.ones(16)
        mesh = discretize.TensorMesh([h, 2 * h, 3 * h])

        vec = np.arange(mesh.nC)
        vtrfname = "temp.vtr"
        modelfname = "arange.txt"
        modeldict = {modelfname: vec}
        mesh.write_vtk(vtrfname, modeldict, directory=tmp_path)
        meshVTR, models = discretize.TensorMesh.read_vtk(vtrfname, directory=tmp_path)

        assert mesh is not meshVTR
        assert mesh.equals(meshVTR)

        np.testing.assert_equal(vec, models[modelfname])
