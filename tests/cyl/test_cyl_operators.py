import numpy as np
import discretize
import discretize.tests as tests
import sympy as sp
from sympy.vector import CoordSys3D, gradient, divergence, curl
import pytest

C = CoordSys3D(
    "C",
    transformation="cylindrical",
    vector_names=list("rtz"),
    variable_names=list("RTZ"),
)


def lambdify_vector(vars, u_vecs, func):
    funcs = [sp.lambdify(vars, func.coeff(u_hat), "numpy") for u_hat in u_vecs]
    return lambda *args: np.stack([g_func(*args) for g_func in funcs], axis=-1)


u = C.R * sp.sin(sp.pi * C.Z) * sp.sin(sp.pi * C.R) * sp.cos(C.T)
su = C.R * sp.sin(sp.pi * C.Z) * sp.sin(sp.pi * C.R)
grad_u = gradient(u)
grad_su = gradient(su)

w = (
    sp.sin(2 * sp.pi * C.Z) * sp.sin(sp.pi * C.R) * sp.sin(C.T) * C.r
    + sp.cos(sp.pi * C.Z) * sp.sin(sp.pi * C.R) * sp.sin(C.T) * C.t
    + sp.sin(sp.pi * C.R) * sp.sin(C.T) * C.z
)
sw = (
    sp.sin(2 * sp.pi * C.Z) * sp.sin(sp.pi * C.R) * C.r
    + sp.cos(sp.pi * C.Z) * sp.sin(sp.pi * C.R) * C.t
    + sp.sin(sp.pi * C.R) * C.z
)
div_w = divergence(w)
curl_w = curl(w)

div_sw = divergence(sw)
curl_sw = curl(sw)

u_func = sp.lambdify((C.R, C.T, C.Z), u, "numpy")
gu_func = lambdify_vector((C.R, C.T, C.Z), (C.r, C.t, C.z), grad_u)

w_func = lambdify_vector((C.R, C.T, C.Z), (C.r, C.t, C.z), w)
dw_func = sp.lambdify((C.R, C.T, C.Z), div_w, "numpy")
cw_func = lambdify_vector((C.R, C.T, C.Z), (C.r, C.t, C.z), curl_w)

su_func = sp.lambdify((C.R, C.T, C.Z), su, "numpy")
gsu_func = lambdify_vector((C.R, C.T, C.Z), (C.r, C.t, C.z), grad_su)

sw_func = lambdify_vector((C.R, C.T, C.Z), (C.r, C.t, C.z), sw)
dsw_func = sp.lambdify((C.R, C.T, C.Z), div_sw, "numpy")
csw_func = lambdify_vector((C.R, C.T, C.Z), (C.r, C.t, C.z), curl_sw)


def setup_mesh(mesh_type, n):
    if mesh_type == "sym_full":
        mesh = discretize.CylindricalMesh([n, 1, n])
        max_h = max(mesh.h[0].max(), mesh.h[2].max())
    elif mesh_type == "sym_ring":
        mesh = discretize.CylindricalMesh([n, 1, n], origin=[1, 0, 0])
        max_h = max(mesh.h[0].max(), mesh.h[2].max())
    elif mesh_type == "quarter_pizza":
        ht = n * [np.pi / (4 * n)]
        mesh = discretize.CylindricalMesh([n, ht, n])
        max_h = max([np.max(hi) for hi in mesh.h])
    elif mesh_type == "full":
        mesh = discretize.CylindricalMesh([n, n, n])
        max_h = max([np.max(hi) for hi in mesh.h])
    elif mesh_type == "ring":
        mesh = discretize.CylindricalMesh([n, n, n], origin=[1, 0, 0])
        max_h = max([np.max(hi) for hi in mesh.h])
    else:
        # tensor styled (1/4 ring)
        ht = n * [np.pi / (4 * n)]
        mesh = discretize.CylindricalMesh([n, ht, ht], origin=[1, 0, 0])
        max_h = max([np.max(hi) for hi in mesh.h])
    return mesh, max_h


SYMMETRIC = [
    "sym_full",
    "sym_ring",
]

NONSYMMETRIC = [
    "quarter_pizza",
    "full",
    "ring",
    "cyl_tensor",
]

MESHTYPES = SYMMETRIC + NONSYMMETRIC


@pytest.mark.parametrize("mesh_type", MESHTYPES)
def test_edge_curl(mesh_type):
    def get_error(n_cells):
        mesh, h = setup_mesh(mesh_type, n_cells)

        if "sym" in mesh_type:
            w = sw_func
            cw = csw_func
        else:
            w = w_func
            cw = cw_func

        e_vec = mesh.project_edge_vector(w(*mesh.edges.T))
        C = mesh.edge_curl

        num = C @ e_vec
        ana = mesh.project_face_vector(cw(*mesh.faces.T))

        err = np.linalg.norm((num - ana), np.inf)
        return err, h

    tests.assert_expected_order(get_error, [10, 20, 30])


@pytest.mark.parametrize("mesh_type", NONSYMMETRIC)
def test_nodal_gradient(mesh_type):
    def get_error(n_cells):
        mesh, h = setup_mesh(mesh_type, n_cells)

        u = u_func
        gu = gu_func

        n_vec = u(*mesh.nodes.T)
        G = mesh.nodal_gradient

        num = G @ n_vec
        ana = mesh.project_edge_vector(gu(*mesh.edges.T))

        err = np.linalg.norm((num - ana), np.inf)
        return err, h

    tests.assert_expected_order(get_error, [10, 20, 30])


@pytest.mark.parametrize("mesh_type", NONSYMMETRIC)
def test_face_divergence(mesh_type):
    def get_error(n_cells):
        mesh, h = setup_mesh(mesh_type, n_cells)

        if "sym" in mesh_type:
            w = sw_func
            dw = dsw_func
        else:
            w = w_func
            dw = dw_func

        f_vec = mesh.project_face_vector(w(*mesh.faces.T))
        D = mesh.face_divergence

        num = D @ f_vec
        ana = dw(*mesh.cell_centers.T)

        err = np.linalg.norm((num - ana), np.inf)
        return err, h

    tests.assert_expected_order(get_error, [10, 20, 30])


@pytest.mark.parametrize("mesh_type", MESHTYPES)
def test_ave_edge_to_face(mesh_type):
    def get_error(n_cells):
        mesh, h = setup_mesh(mesh_type, n_cells)
        if "sym" in mesh_type:
            u = su_func
        else:
            u = u_func
        Ee = u(*mesh.edges.T)

        ave = mesh.average_edge_to_face @ Ee
        ana = u(*mesh.faces.T)
        err = np.linalg.norm((ave - ana), np.inf)
        return err, h

    tests.assert_expected_order(get_error, [10, 20, 30])


@pytest.mark.parametrize("mesh_type", MESHTYPES)
def test_ave_edge_to_cell(mesh_type):
    def get_error(n_cells):
        mesh, h = setup_mesh(mesh_type, n_cells)
        if "sym" in mesh_type:
            u = su_func
        else:
            u = u_func
        Ee = u(*mesh.edges.T)

        ave = mesh.average_edge_to_cell @ Ee
        ana = u(*mesh.cell_centers.T)
        err = np.linalg.norm((ave - ana), np.inf)
        return err, h

    tests.assert_expected_order(get_error, [10, 20, 30])


@pytest.mark.parametrize("mesh_type", MESHTYPES)
def test_ave_face_to_cell(mesh_type):
    def get_error(n_cells):
        mesh, h = setup_mesh(mesh_type, n_cells)
        if "sym" in mesh_type:
            u = su_func
        else:
            u = u_func
        Ee = u(*mesh.faces.T)

        ave = mesh.average_face_to_cell @ Ee
        ana = u(*mesh.cell_centers.T)
        err = np.linalg.norm((ave - ana), np.inf)
        return err, h

    tests.assert_expected_order(get_error, [10, 20, 30])


@pytest.mark.parametrize("mesh_type", MESHTYPES)
def test_ave_cell_to_face(mesh_type):
    def get_error(n_cells):
        mesh, h = setup_mesh(mesh_type, n_cells)
        if "sym" in mesh_type:
            u = su_func
        else:
            u = u_func
        Ee = u(*mesh.cell_centers.T)

        ave = (mesh.average_cell_to_face @ Ee)[~mesh._is_boundary_face]
        ana = u(*mesh.faces.T)[~mesh._is_boundary_face]
        err = np.linalg.norm((ave - ana), np.inf)
        return err, h

    tests.assert_expected_order(get_error, [10, 20, 30])


@pytest.mark.parametrize("mesh_type", NONSYMMETRIC)
def test_ave_node_to_cell(mesh_type):
    def get_error(n_cells):
        mesh, h = setup_mesh(mesh_type, n_cells)
        if "sym" in mesh_type:
            u = su_func
        else:
            u = u_func
        Ee = u(*mesh.nodes.T)

        ave = mesh.average_node_to_cell @ Ee
        ana = u(*mesh.cell_centers.T)
        err = np.linalg.norm((ave - ana), np.inf)
        return err, h

    tests.assert_expected_order(get_error, [10, 20, 30])


@pytest.mark.parametrize("mesh_type", NONSYMMETRIC)
def test_ave_node_to_face(mesh_type):
    def get_error(n_cells):
        mesh, h = setup_mesh(mesh_type, n_cells)
        if "sym" in mesh_type:
            u = su_func
        else:
            u = u_func
        Ee = u(*mesh.nodes.T)

        ave = mesh.average_node_to_face @ Ee
        ana = u(*mesh.faces.T)
        err = np.linalg.norm((ave - ana), np.inf)
        return err, h

    tests.assert_expected_order(get_error, [10, 20, 30])
