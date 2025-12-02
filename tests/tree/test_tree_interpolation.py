import numpy as np
import discretize

import pytest

MESHTYPES = ["uniformTree"]  # ['randomTree', 'uniformTree']
call2 = lambda fun, xyz: fun(xyz[:, 0], xyz[:, 1])
call3 = lambda fun, xyz: fun(xyz[:, 0], xyz[:, 1], xyz[:, 2])
cart_row2 = lambda g, xfun, yfun: np.c_[call2(xfun, g), call2(yfun, g)]
cart_row3 = lambda g, xfun, yfun, zfun: np.c_[
    call3(xfun, g), call3(yfun, g), call3(zfun, g)
]
cartF2 = lambda M, fx, fy: np.vstack(
    (cart_row2(M.gridFx, fx, fy), cart_row2(M.gridFy, fx, fy))
)
cartE2 = lambda M, ex, ey: np.vstack(
    (cart_row2(M.gridEx, ex, ey), cart_row2(M.gridEy, ex, ey))
)
cartF3 = lambda M, fx, fy, fz: np.vstack(
    (
        cart_row3(M.gridFx, fx, fy, fz),
        cart_row3(M.gridFy, fx, fy, fz),
        cart_row3(M.gridFz, fx, fy, fz),
    )
)
cartE3 = lambda M, ex, ey, ez: np.vstack(
    (
        cart_row3(M.gridEx, ex, ey, ez),
        cart_row3(M.gridEy, ex, ey, ez),
        cart_row3(M.gridEz, ex, ey, ez),
    )
)


plotIt = False

MESHTYPES = ["uniformTree", "notatreeTree"]


@pytest.mark.parametrize("tree_type", ["uniformTree", "notatreeTree"])
@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("zeros_outside", [True, False])
@pytest.mark.parametrize(
    "mesh_locs",
    [
        "cell_centers",
        "nodes",
        "edges_x",
        "edges_y",
        "edges_z",
        "faces_x",
        "faces_y",
        "faces_z",
    ],
)
def test_order(tree_type, dim, mesh_locs, zeros_outside):
    if dim == 2 and "z" in mesh_locs:
        pytest.skip()

    locs = (
        np.mgrid[
            *[
                slice(0.25, 0.75, 50j),
            ]
            * dim
        ]
        .reshape(dim, -1)
        .transpose()
    )

    if "notatree" in tree_type:
        expected_order = 2
    elif mesh_locs == "nodes":
        expected_order = 2
    else:
        expected_order = 1

    def ana_func(locs):
        return locs**2 * [2, -3, 4][:dim] + locs * [-4, 3, 2][:dim] + [2, 3, 4][:dim]

    ana_vals = ana_func(locs)

    if "faces" in mesh_locs:
        source_attr = "faces"
    elif "edges" in mesh_locs:
        source_attr = "edges"
    else:
        source_attr = mesh_locs

    def order_func(n):
        mesh, h = discretize.tests.setup_mesh(tree_type, n, dim)
        interp_mat = mesh.get_interpolation_matrix(
            locs, mesh_locs, zeros_outside=zeros_outside
        )
        grid_vals = ana_func(getattr(mesh, source_attr))
        interp_vals = interp_mat @ grid_vals

        return np.linalg.norm(interp_vals - ana_vals), h

    discretize.tests.assert_expected_order(
        order_func,
        [8, 16, 32],
        expected_order=expected_order,
        test_type="mean_at_least",
    )


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize(
    "mesh_locs",
    [
        "cell_centers",
        "nodes",
        "edges_x",
        "edges_y",
        "edges_z",
        "faces_x",
        "faces_y",
        "faces_z",
    ],
)
def test_zeros_outside(dim, mesh_locs, zeros_outside):
    if dim == 2 and "z" in mesh_locs:
        pytest.skip()

    locs = (
        np.mgrid[
            *[
                slice(-1, 2, 3j),
            ]
            * dim
        ]
        .reshape(dim, -1)
        .transpose()
    )
    mesh = discretize.TreeMesh([16, 16, 16][:dim])
    mesh.refine(-1)

    is_outside = np.any((locs < 0) | (locs > 1), axis=1)
    locs = locs[is_outside]

    interp_mat = mesh.get_interpolation_matrix(locs, mesh_locs, zeros_outside=True)

    if "faces" in mesh_locs:
        n = mesh.n_faces
    elif "edges" in mesh_locs:
        n = mesh.n_edges
    elif "nodes" in mesh_locs:
        n = mesh.n_nodes
    else:
        n = mesh.n_cells

    vs = interp_mat @ np.ones(n)

    np.testing.assert_equal(vs, 0)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize(
    "mesh_locs",
    [
        "cell_centers",
        "nodes",
        "edges_x",
        "edges_y",
        "edges_z",
        "faces_x",
        "faces_y",
        "faces_z",
    ],
)
def test_project_outside(dim, mesh_locs):
    if dim == 2 and "z" in mesh_locs:
        pytest.skip()

    locs = (
        np.mgrid[
            *[
                slice(-1, 2, 3j),
            ]
            * dim
        ]
        .reshape(dim, -1)
        .transpose()
    )
    mesh = discretize.TreeMesh([16, 16, 16][:dim], diagonal_balance=True)
    mesh.refine(-1)

    is_outside = np.any((locs < 0) | (locs > 1), axis=1)
    locs = locs[is_outside]

    grid_locs = getattr(mesh, mesh_locs)
    source_bounds = [
        grid_locs.min(axis=0),
        grid_locs.max(axis=0),
    ]
    interp_mat = mesh.get_interpolation_matrix(locs, mesh_locs, zeros_outside=False)

    def ana_func(locs):
        locs = np.clip(locs, a_min=source_bounds[0], a_max=source_bounds[1])
        return locs * [-4, 3, 2][:dim] + [2, 3, 4][:dim]

    ana_vals = ana_func(locs)

    # get the full list of locations associate with mesh_locs
    if "faces" in mesh_locs:
        source_attr = "faces"
    elif "edges" in mesh_locs:
        source_attr = "edges"
    else:
        source_attr = mesh_locs

    source_locs = getattr(mesh, source_attr)
    grid_vals = ana_func(source_locs)

    vs = interp_mat @ grid_vals

    np.testing.assert_equal(vs, ana_vals)
