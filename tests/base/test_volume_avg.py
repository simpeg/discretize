import numpy as np
import pytest
import discretize
from discretize.utils import volume_average
from numpy.testing import assert_allclose


def generate_mesh(dim, mesh_type, tree_point=None, sub_mesh=False, seed=0):
    if seed is None:
        h = np.ones(16)
    else:
        rng = np.random.default_rng(seed)
        h = rng.random(16)
        h /= h.sum()
    if sub_mesh:
        h = h[:8]
        if tree_point is not None:
            tree_point = tree_point * 0.5
    hs = [h] * dim
    mesh = mesh_type(hs)
    if isinstance(mesh, discretize.TreeMesh):
        mesh.insert_cells(tree_point, -1)
    return mesh


@pytest.mark.parametrize("same_base", [True, False])
@pytest.mark.parametrize("sub_mesh", [True, False])
@pytest.mark.parametrize(
    "dim, mesh1_type, mesh2_type",
    [
        (1, discretize.TensorMesh, discretize.TensorMesh),
        (2, discretize.TensorMesh, discretize.TensorMesh),
        (3, discretize.TensorMesh, discretize.TensorMesh),
        (2, discretize.TreeMesh, discretize.TensorMesh),
        (3, discretize.TreeMesh, discretize.TensorMesh),
        (2, discretize.TensorMesh, discretize.TreeMesh),
        (3, discretize.TensorMesh, discretize.TreeMesh),
        (2, discretize.TreeMesh, discretize.TreeMesh),
        (3, discretize.TreeMesh, discretize.TreeMesh),
    ],
)
def test_volume_average(dim, mesh1_type, mesh2_type, same_base, sub_mesh, seed=102):
    if dim == 1:
        if mesh1_type is discretize.TreeMesh or mesh2_type is discretize.TreeMesh:
            pytest.skip("TreeMesh only in 2D or higher")

    p1 = p2 = None
    if mesh1_type is discretize.TreeMesh:
        p1 = np.asarray([0.25] * dim)
    if mesh2_type is discretize.TreeMesh:
        p2 = np.asarray([0.75] * dim)

    rng = np.random.default_rng(seed)

    if not sub_mesh:
        seed1, seed2 = rng.integers(554, size=(2,))
        if same_base:
            seed2 = seed1
    else:
        seed1 = seed2 = None

    mesh1 = generate_mesh(dim, mesh1_type, tree_point=p1, seed=seed1)
    mesh2 = generate_mesh(dim, mesh2_type, tree_point=p2, sub_mesh=sub_mesh, seed=seed2)

    model_in = rng.random(mesh1.nC)
    model_out1 = np.empty(mesh2.nC)

    # test the three ways of calling...

    # providing an output array
    out1 = volume_average(mesh1, mesh2, model_in, model_out1)
    # assert_array_equal(out1, model_out1)
    assert out1 is model_out1

    # only providing input array
    out2 = volume_average(mesh1, mesh2, model_in)
    assert_allclose(out1, out2)

    # not providing either (which constructs a sparse matrix representing the operation)
    Av = volume_average(mesh1, mesh2)
    out3 = Av @ model_in
    assert_allclose(out1, out3)

    # test for mass conserving properties:
    if sub_mesh:
        # get cells in extent of smaller mesh
        cells = mesh1.cell_centers < 8
        if dim > 1:
            cells = np.all(cells, axis=1)

        mass1 = np.sum(mesh1.cell_volumes[cells] * model_in[cells])
    else:
        mass1 = np.sum(mesh1.cell_volumes * model_in)
    mass2 = np.sum(mesh2.cell_volumes * out3)
    assert_allclose(mass1, mass2)


def test_errors():
    h1 = np.random.rand(16)
    h1 /= h1.sum()
    h2 = np.random.rand(16)
    h2 /= h2.sum()
    mesh1D = discretize.TensorMesh([h1])
    mesh2D = discretize.TensorMesh([h1, h1])
    mesh3D = discretize.TensorMesh([h1, h1, h1])

    hr = np.r_[1, 1, 0.5]
    hz = np.r_[2, 1]
    meshCyl = discretize.CylindricalMesh([hr, 1, hz], np.r_[0.0, 0.0, 0.0])
    mesh2 = discretize.TreeMesh([h2, h2])
    mesh2.insert_cells([0.75, 0.75], [4])

    with pytest.raises(TypeError):
        # Gives a wrong typed object to the function
        volume_average(mesh1D, h1)
    with pytest.raises(NotImplementedError):
        # Gives a wrong typed mesh
        volume_average(meshCyl, mesh2)
    with pytest.raises(ValueError):
        # Gives mismatching mesh dimensions
        volume_average(mesh2D, mesh3D)

    model1 = np.random.randn(mesh2D.nC)
    bad_model1 = np.random.randn(3)
    bad_model2 = np.random.rand(1)
    # gives input values with incorrect lengths
    with pytest.raises(ValueError):
        volume_average(mesh2D, mesh2, bad_model1)
    with pytest.raises(ValueError):
        volume_average(mesh2D, mesh2, model1, bad_model2)
