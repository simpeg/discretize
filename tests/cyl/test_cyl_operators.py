import numpy as np
import discretize
import discretize.tests as tests
import sympy as sp
from sympy.vector import CoordSys3D, gradient, divergence, curl
import pytest
import scipy.sparse as spr

C = CoordSys3D(
    "C",
    transformation="cylindrical",
    vector_names=list("rtz"),
    variable_names=list("RTZ"),
)

rng = np.random.default_rng(563279)


def lambdify_vector(variabls, u_vecs, func):
    funcs = [sp.lambdify(variabls, func.coeff(u_hat), "numpy") for u_hat in u_vecs]
    return lambda *args: np.stack(
        [g_func(*args) * np.ones_like(args[0]) for g_func in funcs], axis=-1
    )


u = C.R * (sp.sin(sp.pi * C.Z) * sp.sin(sp.pi * C.R) * sp.cos(C.T) + 1)
su = C.R * (sp.sin(sp.pi * C.Z) * sp.sin(sp.pi * C.R) + 1)
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
        mesh = discretize.CylindricalMesh([n, ht, n], origin=[1, 0, 0])
        max_h = max([np.max(hi) for hi in mesh.h])
    return mesh, max_h


SYMMETRIC = {
    "sym_full",
    "sym_ring",
}

NONSYMMETRIC = {
    "quarter_pizza",
    "full",
    "ring",
    "cyl_tensor",
}

ZEROSTART = {
    "sym_full",
    "quarter_pizza",
    "full",
}

PARTAZIMUTH = {
    "cyl_tensor",
    "quarter_pizza",
}

MESHTYPES = SYMMETRIC | NONSYMMETRIC


def get_integration_limits(mesh_type):
    if mesh_type in ZEROSTART:
        r_lims = [0, 1]
    else:
        r_lims = [1, 2]

    if mesh_type in PARTAZIMUTH:
        t_lims = [0, sp.pi / 4]
    else:
        t_lims = [0, 2 * sp.pi]
    z_lims = [0, 1]
    return r_lims, t_lims, z_lims


@pytest.mark.parametrize("mesh_type", MESHTYPES)
def test_edge_curl(mesh_type):
    def get_error(n_cells):
        mesh, h = setup_mesh(mesh_type, n_cells)

        if mesh_type in SYMMETRIC:
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


@pytest.mark.parametrize("mesh_type", MESHTYPES)
def test_face_divergence(mesh_type):
    def get_error(n_cells):
        mesh, h = setup_mesh(mesh_type, n_cells)

        if mesh_type in SYMMETRIC:
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
        if mesh_type in SYMMETRIC:
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
        if mesh_type in SYMMETRIC:
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
        if mesh_type in SYMMETRIC:
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
        if mesh_type in SYMMETRIC:
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
        if mesh_type in SYMMETRIC:
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
        if mesh_type in SYMMETRIC:
            u = su_func
        else:
            u = u_func
        Ee = u(*mesh.nodes.T)

        ave = mesh.average_node_to_face @ Ee
        ana = u(*mesh.faces.T)
        err = np.linalg.norm((ave - ana), np.inf)
        return err, h

    tests.assert_expected_order(get_error, [10, 20, 30])


@pytest.mark.parametrize("mesh_type", MESHTYPES)
def test_ave_face_to_cell_vector(mesh_type):
    def get_error(n_cells):
        mesh, h = setup_mesh(mesh_type, n_cells)
        if mesh_type in SYMMETRIC:
            w = sw_func
        else:
            w = w_func

        Ee = mesh.project_face_vector(w(*mesh.faces.T))

        ave = mesh.average_face_to_cell_vector @ Ee
        ana = w(*mesh.cell_centers.T)
        if mesh_type in SYMMETRIC:
            ana = ana[:, [0, 2]]
        ana = ana.reshape(-1, order="F")
        err = np.linalg.norm((ave - ana), np.inf)
        return err, h

    tests.assert_expected_order(get_error, [10, 20, 30])


@pytest.mark.parametrize("mesh_type", MESHTYPES)
def test_ave_edge_to_cell_vector(mesh_type):
    def get_error(n_cells):
        mesh, h = setup_mesh(mesh_type, n_cells)
        if mesh_type in SYMMETRIC:
            w = sw_func
        else:
            w = w_func

        Ee = mesh.project_edge_vector(w(*mesh.edges.T))

        ave = mesh.average_edge_to_cell_vector @ Ee
        ana = w(*mesh.cell_centers.T)
        if mesh_type in SYMMETRIC:
            ana = ana[:, [1]]
        ana = ana.reshape(-1, order="F")
        err = np.linalg.norm((ave - ana), np.inf)
        return err, h

    tests.assert_expected_order(get_error, [10, 20, 30])


@pytest.mark.parametrize("mesh_type", MESHTYPES)
def test_mimetic_div_curl(mesh_type):
    mesh, _ = setup_mesh(mesh_type, 10)

    v = rng.random(mesh.n_edges)
    divcurlv = mesh.face_divergence @ (mesh.edge_curl @ v)
    np.testing.assert_allclose(divcurlv, 0, atol=1e-11)


@pytest.mark.parametrize("mesh_type", NONSYMMETRIC)
def test_mimetic_curl_grad(mesh_type):
    mesh, _ = setup_mesh(mesh_type, 10)

    v = rng.random(mesh.n_nodes)
    divcurlv = mesh.edge_curl @ (mesh.nodal_gradient @ v)
    np.testing.assert_allclose(divcurlv, 0, atol=1e-11)


@pytest.mark.parametrize("mesh_type", MESHTYPES)
def test_simple_edge_inner_product(mesh_type):
    r_lims, t_lims, z_lims = get_integration_limits(mesh_type)

    if mesh_type in SYMMETRIC:
        # only theta edges
        e_ana = C.R * C.t
    else:
        e_ana = 1 * C.r + C.R * C.t + 1 * C.z
    e_func = lambdify_vector((C.R, C.T, C.Z), (C.r, C.t, C.z), e_ana)

    ana = float(
        sp.integrate(
            e_ana.dot(e_ana) * C.R, (C.R, *r_lims), (C.T, *t_lims), (C.Z, *z_lims)
        )
    )

    mesh, _ = setup_mesh(mesh_type, 10)

    e = mesh.project_edge_vector(e_func(*mesh.edges.T))
    Me = mesh.get_edge_inner_product()
    num = e.T @ Me @ e
    np.testing.assert_allclose(num, ana)


@pytest.mark.parametrize("mesh_type", MESHTYPES)
def test_simple_face_inner_product(mesh_type):
    r_lims, t_lims, z_lims = get_integration_limits(mesh_type)

    if mesh_type in SYMMETRIC:
        # no theta faces
        f_ana = C.R * C.r + 1 * C.z
    else:
        f_ana = C.R * C.r + 1 * C.t + 1 * C.z
    f_func = lambdify_vector((C.R, C.T, C.Z), (C.r, C.t, C.z), f_ana)

    ana = float(
        sp.integrate(
            f_ana.dot(f_ana) * C.R, (C.R, *r_lims), (C.T, *t_lims), (C.Z, *z_lims)
        )
    )

    mesh, _ = setup_mesh(mesh_type, 10)

    f = mesh.project_face_vector(f_func(*mesh.faces.T))
    Mf = mesh.get_face_inner_product()
    num = f.T @ Mf @ f
    np.testing.assert_allclose(num, ana)


@pytest.mark.parametrize("mesh_type", NONSYMMETRIC)
def test_simple_edge_ave(mesh_type):
    func = lambda r, t, z: np.c_[r, r, z]
    mesh, _ = setup_mesh(mesh_type, 10)
    e_ana = mesh.project_edge_vector(func(*mesh.edges.T))
    ave_e = mesh.aveE2CCV @ e_ana

    ana_cc = func(*mesh.cell_centers.T).reshape(-1, order="F")

    np.testing.assert_allclose(ave_e, ana_cc)


@pytest.mark.parametrize("mesh_type", NONSYMMETRIC)
def test_simple_face_ave(mesh_type):
    func = lambda r, t, z: np.c_[r, r, z]
    mesh, _ = setup_mesh(mesh_type, 10)
    f_ana = mesh.project_face_vector(func(*mesh.faces.T))
    ave_f = mesh.aveF2CCV @ f_ana

    ana_cc = func(*mesh.cell_centers.T).reshape(-1, order="F")

    np.testing.assert_allclose(ave_f, ana_cc)


@pytest.mark.parametrize("mesh_type", NONSYMMETRIC)
def test_gradient_boundary_integral(mesh_type):
    u_simp = C.R**2 + sp.cos(C.T) + C.Z**2
    gu_simp = gradient(u_simp)
    v_simp = C.R**2 * C.r + C.R * C.t + C.R * C.Z * C.z

    u_sfunc = sp.lambdify((C.R, C.T, C.Z), u_simp, "numpy")
    v_sfunc = lambdify_vector((C.R, C.T, C.Z), (C.r, C.t, C.z), v_simp)

    # int_V grad_u dot w dV
    r_lims, t_lims, z_lims = get_integration_limits(mesh_type)

    ana = float(
        sp.integrate(
            gu_simp.dot(v_simp) * C.R, (C.R, *r_lims), (C.T, *t_lims), (C.Z, *z_lims)
        )
    )

    def get_error(n_cells):
        mesh, h = setup_mesh(mesh_type, n_cells)

        u_cc = u_sfunc(*mesh.cell_centers.T)
        w_f = mesh.project_face_vector(v_sfunc(*mesh.faces.T))
        u_bf = u_sfunc(*mesh.boundary_faces.T)

        D = mesh.face_divergence
        M_c = spr.diags(mesh.cell_volumes)
        M_bf = mesh.boundary_face_scalar_integral

        d1 = (w_f.T @ D.T) @ M_c @ u_cc
        d2 = w_f.T @ M_bf @ u_bf

        discrete_val = -d1 + d2
        err = np.abs(ana - discrete_val)
        return err, h

    tests.assert_expected_order(get_error, [10, 20, 30])


@pytest.mark.parametrize("mesh_type", NONSYMMETRIC)
def test_div_boundary_integral(mesh_type):
    u_simp = C.R**2 + sp.cos(C.T) + C.Z**2
    v_simp = C.R**2 * C.r + C.R * C.t + C.R * C.Z * C.z
    dv_simp = divergence(v_simp)

    u_sfunc = sp.lambdify((C.R, C.T, C.Z), u_simp, "numpy")
    v_sfunc = lambdify_vector((C.R, C.T, C.Z), (C.r, C.t, C.z), v_simp)

    r_lims, t_lims, z_lims = get_integration_limits(mesh_type)

    ana = float(
        sp.integrate(
            u_simp * dv_simp * C.R, (C.R, *r_lims), (C.T, *t_lims), (C.Z, *z_lims)
        )
    )

    def get_error(n_cells):
        mesh, h = setup_mesh(mesh_type, n_cells)

        u_n = u_sfunc(*mesh.nodes.T)
        v_e = mesh.project_edge_vector(v_sfunc(*mesh.edges.T))
        v_bn = v_sfunc(*mesh.boundary_nodes.T).reshape(-1, order="F")

        M_e = mesh.get_edge_inner_product()
        G = mesh.nodal_gradient
        M_bn = mesh.boundary_node_vector_integral

        d1 = (u_n.T @ G.T) @ M_e @ v_e
        d2 = u_n.T @ (M_bn @ v_bn)

        discrete_val = -d1 + d2
        err = np.abs(ana - discrete_val)
        return err, h

    tests.assert_expected_order(get_error, [10, 20, 30])


@pytest.mark.parametrize("mesh_type", NONSYMMETRIC)
def test_curl_boundary_integral(mesh_type):
    w_simp = C.R**2 * C.Z * C.r + sp.sin(C.R) * C.Z * C.t + C.R**3 * C.z
    v_simp = C.R**2 * C.r + C.R * C.t + C.R * C.Z * C.z
    cw_simp = curl(w_simp)

    w_sfunc = lambdify_vector((C.R, C.T, C.Z), (C.r, C.t, C.z), w_simp)
    v_sfunc = lambdify_vector((C.R, C.T, C.Z), (C.r, C.t, C.z), v_simp)

    r_lims, t_lims, z_lims = get_integration_limits(mesh_type)

    ana = float(
        sp.integrate(
            cw_simp.dot(v_simp) * C.R, (C.R, *r_lims), (C.T, *t_lims), (C.Z, *z_lims)
        )
    )

    def get_error(n_cells):
        mesh, h = setup_mesh(mesh_type, n_cells)

        w_f = mesh.project_face_vector(w_sfunc(*mesh.faces.T))
        v_e = mesh.project_edge_vector(v_sfunc(*mesh.edges.T))
        w_be = w_sfunc(*mesh.boundary_edges.T).reshape(-1, order="F")

        M_f = mesh.get_face_inner_product()
        Curl = mesh.edge_curl
        M_be = mesh.boundary_edge_vector_integral

        d1 = (v_e.T @ Curl.T) @ M_f @ w_f
        d2 = v_e.T @ (M_be @ w_be)

        discrete_val = d1 - d2
        err = np.abs(ana - discrete_val)
        return err, h

    tests.assert_expected_order(get_error, [10, 20, 30])


@pytest.mark.parametrize(
    "location_type",
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
@pytest.mark.parametrize("mesh_type", NONSYMMETRIC)
def test_interpolation(mesh_type, location_type):
    u_func = lambda x, y, z: x**2 + y**2 + z**2

    interp_points = (
        np.mgrid[0.3:0.8:5j, np.pi / 10 : np.pi / 5 : 5j, 0.3:0.8:5j].reshape(3, -1).T
    )
    interp_points.shape

    if mesh_type not in ZEROSTART:
        interp_points[:, 0] += 1
    ana = u_func(*interp_points.T)

    def get_error(n_cells):
        mesh, h = setup_mesh(mesh_type, n_cells)
        if "edges" in location_type:
            grid = mesh.edges
        elif "faces" in location_type:
            grid = mesh.faces
        else:
            grid = getattr(mesh, location_type)
        A = mesh.get_interpolation_matrix(interp_points, location_type=location_type)

        num = A @ u_func(*grid.T)
        err = np.linalg.norm((num - ana), np.inf)
        return err, h

    tests.assert_expected_order(get_error, [10, 20, 30])
