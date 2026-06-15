import numpy as np
import unittest
import pytest

from discretize import TensorMesh, CurvilinearMesh
from discretize.utils import ndgrid
from discretize.tests import setup_mesh, check_derivative, assert_expected_order


class BasicCurvTests(unittest.TestCase):
    def setUp(self):
        a = np.array([1, 1, 1])
        b = np.array([1, 2])
        c = np.array([1, 4])

        def gridIt(h):
            return [np.cumsum(np.r_[0, x]) for x in h]

        X, Y = ndgrid(gridIt([a, b]), vector=False)
        self.TM2 = TensorMesh([a, b])
        self.Curv2 = CurvilinearMesh([X, Y])
        X, Y, Z = ndgrid(gridIt([a, b, c]), vector=False)
        self.TM3 = TensorMesh([a, b, c])
        self.Curv3 = CurvilinearMesh([X, Y, Z])

    def test_area_3D(self):
        test_area = np.array(
            [
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                4,
                4,
                4,
                4,
                8,
                8,
                8,
                8,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                1,
                1,
                1,
                2,
                2,
                2,
                1,
                1,
                1,
                2,
                2,
                2,
                1,
                1,
                1,
                2,
                2,
                2,
            ]
        )
        self.assertTrue(np.all(self.Curv3.face_areas == test_area))

    def test_vol_3D(self):
        test_vol = np.array([1, 1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8])
        np.testing.assert_almost_equal(self.Curv3.cell_volumes, test_vol)
        self.assertTrue(True)  # Pass if you get past the assertion.

    def test_vol_2D(self):
        test_vol = np.array([1, 1, 1, 2, 2, 2])
        t1 = np.all(self.Curv2.cell_volumes == test_vol)
        self.assertTrue(t1)

    def test_edge_3D(self):
        test_edge = np.array(
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
            ]
        )
        t1 = np.all(self.Curv3.edge_lengths == test_edge)
        self.assertTrue(t1)

    def test_edge_2D(self):
        test_edge = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2])
        t1 = np.all(self.Curv2.edge_lengths == test_edge)
        self.assertTrue(t1)

    def test_tangents(self):
        T = self.Curv2.edge_tangents
        self.assertTrue(
            np.all(self.Curv2.reshape(T, "E", "Ex", "V")[0] == np.ones(self.Curv2.nEx))
        )
        self.assertTrue(
            np.all(self.Curv2.reshape(T, "E", "Ex", "V")[1] == np.zeros(self.Curv2.nEx))
        )
        self.assertTrue(
            np.all(self.Curv2.reshape(T, "E", "Ey", "V")[0] == np.zeros(self.Curv2.nEy))
        )
        self.assertTrue(
            np.all(self.Curv2.reshape(T, "E", "Ey", "V")[1] == np.ones(self.Curv2.nEy))
        )

        T = self.Curv3.edge_tangents
        self.assertTrue(
            np.all(self.Curv3.reshape(T, "E", "Ex", "V")[0] == np.ones(self.Curv3.nEx))
        )
        self.assertTrue(
            np.all(self.Curv3.reshape(T, "E", "Ex", "V")[1] == np.zeros(self.Curv3.nEx))
        )
        self.assertTrue(
            np.all(self.Curv3.reshape(T, "E", "Ex", "V")[2] == np.zeros(self.Curv3.nEx))
        )

        self.assertTrue(
            np.all(self.Curv3.reshape(T, "E", "Ey", "V")[0] == np.zeros(self.Curv3.nEy))
        )
        self.assertTrue(
            np.all(self.Curv3.reshape(T, "E", "Ey", "V")[1] == np.ones(self.Curv3.nEy))
        )
        self.assertTrue(
            np.all(self.Curv3.reshape(T, "E", "Ey", "V")[2] == np.zeros(self.Curv3.nEy))
        )

        self.assertTrue(
            np.all(self.Curv3.reshape(T, "E", "Ez", "V")[0] == np.zeros(self.Curv3.nEz))
        )
        self.assertTrue(
            np.all(self.Curv3.reshape(T, "E", "Ez", "V")[1] == np.zeros(self.Curv3.nEz))
        )
        self.assertTrue(
            np.all(self.Curv3.reshape(T, "E", "Ez", "V")[2] == np.ones(self.Curv3.nEz))
        )

    def test_normals(self):
        N = self.Curv2.face_normals
        self.assertTrue(
            np.all(self.Curv2.reshape(N, "F", "Fx", "V")[0] == np.ones(self.Curv2.nFx))
        )
        self.assertTrue(
            np.all(self.Curv2.reshape(N, "F", "Fx", "V")[1] == np.zeros(self.Curv2.nFx))
        )
        self.assertTrue(
            np.all(self.Curv2.reshape(N, "F", "Fy", "V")[0] == np.zeros(self.Curv2.nFy))
        )
        self.assertTrue(
            np.all(self.Curv2.reshape(N, "F", "Fy", "V")[1] == np.ones(self.Curv2.nFy))
        )

        N = self.Curv3.face_normals
        self.assertTrue(
            np.all(self.Curv3.reshape(N, "F", "Fx", "V")[0] == np.ones(self.Curv3.nFx))
        )
        self.assertTrue(
            np.all(self.Curv3.reshape(N, "F", "Fx", "V")[1] == np.zeros(self.Curv3.nFx))
        )
        self.assertTrue(
            np.all(self.Curv3.reshape(N, "F", "Fx", "V")[2] == np.zeros(self.Curv3.nFx))
        )

        self.assertTrue(
            np.all(self.Curv3.reshape(N, "F", "Fy", "V")[0] == np.zeros(self.Curv3.nFy))
        )
        self.assertTrue(
            np.all(self.Curv3.reshape(N, "F", "Fy", "V")[1] == np.ones(self.Curv3.nFy))
        )
        self.assertTrue(
            np.all(self.Curv3.reshape(N, "F", "Fy", "V")[2] == np.zeros(self.Curv3.nFy))
        )

        self.assertTrue(
            np.all(self.Curv3.reshape(N, "F", "Fz", "V")[0] == np.zeros(self.Curv3.nFz))
        )
        self.assertTrue(
            np.all(self.Curv3.reshape(N, "F", "Fz", "V")[1] == np.zeros(self.Curv3.nFz))
        )
        self.assertTrue(
            np.all(self.Curv3.reshape(N, "F", "Fz", "V")[2] == np.ones(self.Curv3.nFz))
        )

    def test_grid(self):
        self.assertTrue(np.all(self.Curv2.gridCC == self.TM2.gridCC))
        self.assertTrue(np.all(self.Curv2.gridN == self.TM2.gridN))
        self.assertTrue(np.all(self.Curv2.gridFx == self.TM2.gridFx))
        self.assertTrue(np.all(self.Curv2.gridFy == self.TM2.gridFy))
        self.assertTrue(np.all(self.Curv2.gridEx == self.TM2.gridEx))
        self.assertTrue(np.all(self.Curv2.gridEy == self.TM2.gridEy))

        self.assertTrue(np.all(self.Curv3.gridCC == self.TM3.gridCC))
        self.assertTrue(np.all(self.Curv3.gridN == self.TM3.gridN))
        self.assertTrue(np.all(self.Curv3.gridFx == self.TM3.gridFx))
        self.assertTrue(np.all(self.Curv3.gridFy == self.TM3.gridFy))
        self.assertTrue(np.all(self.Curv3.gridFz == self.TM3.gridFz))
        self.assertTrue(np.all(self.Curv3.gridEx == self.TM3.gridEx))
        self.assertTrue(np.all(self.Curv3.gridEy == self.TM3.gridEy))
        self.assertTrue(np.all(self.Curv3.gridEz == self.TM3.gridEz))


@pytest.mark.parametrize("u_type", ["edge", "face"])
@pytest.mark.parametrize("dim", [2, 3], ids=["2D", "3D"])
@pytest.mark.parametrize("rep", [0, 1], ids=["uniform", "isotropic"])
def test_surface_inner_product_prop_deriv(u_type, dim, rep):
    rng = np.random.default_rng(6732)
    mesh, _ = setup_mesh("rotateCurv", 20, dim)
    tau = rng.uniform(1, 2, 1) if rep == 0 else rng.uniform(1, 2, mesh.n_faces * rep)

    match u_type:
        case "edge":
            v = rng.uniform(1, 2, mesh.n_edges)

            def fun(tau):
                M = mesh.get_edge_inner_product_surface(tau)
                Md = mesh.get_edge_inner_product_surface_deriv(tau)
                return M * v, Md(v)

        case "face":
            v = rng.uniform(1, 2, mesh.n_faces)

            def fun(tau):
                M = mesh.get_face_inner_product_surface(tau)
                Md = mesh.get_face_inner_product_surface_deriv(tau)
                return M * v, Md(v)

        case _:
            raise Exception("Invalid test parameter.")

    check_derivative(fun, tau, num=5, random_seed=rng)


@pytest.mark.parametrize("dim", [2, 3], ids=["2D", "3D"])
@pytest.mark.parametrize("rep", [0, 1], ids=["uniform", "isotropic"])
def test_line_inner_product_prop_deriv(dim, rep):
    rng = np.random.default_rng(6732)
    mesh, _ = setup_mesh("rotateCurv", 20, dim)
    v = rng.uniform(1, 2, mesh.n_edges)
    tau = rng.uniform(1, 2, 1) if rep == 0 else rng.uniform(1, 2, mesh.n_edges * rep)

    def fun(tau):
        M = mesh.get_edge_inner_product_line(tau)
        Md = mesh.get_edge_inner_product_line_deriv(tau)
        return M * v, Md(v)

    check_derivative(fun, tau, num=5, random_seed=rng)


def test_edge_surface_integral_2d():
    """
    testing the line integral here of:
    vector field w: [y**2, x**2]
    physical property u: (1 - x) * (1 + x)
    over the path x=t, y=(t - 1) * (t + 1)
    from t=-1 to 1
    """

    def error_eval(nx):
        ny = 4  # not really important to this test, as only one "surface" is non-zeros
        xlocs = np.linspace(-1, 1, nx + 1)
        nodes_x = xlocs[:, None] * np.ones(ny + 1)[None, :]

        ylocs = (xlocs - 1) * (xlocs + 1)
        nodes_y = ylocs[:, None] + np.linspace(-1, 1, ny + 1)

        mesh = CurvilinearMesh((nodes_x, nodes_y))

        faces = (np.arange(mesh.n_faces_y) + mesh.n_faces_x).reshape(ny + 1, nx)
        face_inds = faces[2, :]

        xcs = (xlocs[1:] + xlocs[:-1]) * 0.5
        prop = (1 - xcs) * (xcs + 1) + 1
        face_props = np.zeros(mesh.n_faces)
        face_props[face_inds] = prop

        edge_vectors = np.c_[mesh.edges[:, 1] ** 2, mesh.edges[:, 0] ** 2]
        edge_vals = mesh.project_edge_vector(edge_vectors)

        M = mesh.get_edge_inner_product_surface(model=face_props)

        discrete_val = np.sum(M @ edge_vals)
        reference_value = 208 / 105
        return np.abs(discrete_val - reference_value), xlocs[1] - xlocs[0]

    assert_expected_order(error_eval, [20.0, 30.0, 40.0, 50.0])


def test_edge_surface_integral_3d():
    """
    For this test, we will use a single surface within the curvilinear mesh
    parameterized as:
    x = u
    y = v
    z = u**2 + v**2
    (a simple parabola)

    we want to test the integral of:
    mu * w.dot(w) ds
    over this surface, for some a vector field that is always tangent to the surface
    the two vectors that are perpendicular and tangent to this surface are:
    r_u = [1, 0, 2*u]
    r_v = [0, 1, 2*v]

    so any linear combination of these vectors will be tangent to the surface, let's use:
    w = x * y * z * (r_u + r_v)

    the ds integrand is sqrt(|| r_u.cross(r_v) ||) or:
    sqrt(4 * x**2 + 4 * y**2 + 1)

    and a property field: mu = x**2 + y**2 + z**2

    the analytic integral over the range u=[-1, 1], v=[-1,1] found by numeric integration:

    >>> def int_f(x, y):
    ...    xsq = x**2
    ...    ysq = y**2
    ...    z = x**2 + y**2
    ...    zsq = z**2
    ...    v1 = 2 * xsq * ysq * zsq + (2 * xsq * y * z + 2 * x * ysq * z)**2
    ...    v2 = (xsq + ysq + zsq) * np.sqrt(4 * xsq + 4 * ysq + 1)
    ...    return v1 * v2
    >>> scint.dblquad(int_f, -1, 1, -1, 1, epsrel=1e-20, epsabs=1E-20)
    (51.8667132595898, 2.241102757162104e-12)
    """

    def error_eval(nx):
        ny = nx
        nz = 4  # not really important to this test, as only one "surface" is non-zero
        xlocs = np.linspace(-1, 1, nx + 1)
        ylocs = np.linspace(-1, 1, ny + 1)
        nodes_x = xlocs[:, None, None] * np.ones((1, ny + 1, nz + 1))
        nodes_y = ylocs[None, :, None] * np.ones((nx + 1, 1, nz + 1))

        nodes_z = nodes_x**2 + nodes_y**2 + np.linspace(-1, 1, nz + 1)[None, None, :]
        mesh = CurvilinearMesh((nodes_x, nodes_y, nodes_z))

        faces = (np.arange(mesh.n_faces_z) + mesh.n_faces_x + mesh.n_faces_y).reshape(
            nz + 1, nx * ny
        )
        face_inds = faces[2, :]

        prop = mesh.faces[:, 0] ** 2 + mesh.faces[:, 1] ** 2 + mesh.faces[:, 2] ** 2
        face_props = np.zeros(mesh.n_faces)
        face_props[face_inds] = prop[face_inds]

        ex = mesh.edges[:, 0]
        ey = mesh.edges[:, 1]
        w_vec = (ex * ey * (ex**2 + ey**2))[:, None] * np.c_[
            np.ones_like(ex),
            np.ones_like(ex),
            2 * ex + 2 * ey,
        ]
        w = mesh.project_edge_vector(w_vec)

        M = mesh.get_edge_inner_product_surface(model=face_props)

        discrete_val = w @ M @ w
        reference_value = 51.8667132595898
        return np.abs(discrete_val - reference_value), xlocs[1] - xlocs[0]

    assert_expected_order(error_eval, [20.0, 30.0, 40.0, 50.0])


def test_edge_line_integral_3d():
    """
    testing the line integral here of:
    scalar field: v = x**2 + y**2
    times the property: mu = (1 - x) * (x + 1) + 1
    over the path x=t, y = 2 * t, z=(t - 1) * (t + 1)
    from t=-1 to 1

    >>> t = sy.Symbol('t')
    >>> x, y, z = t, 2 * t, (t-1)*(t+1)
    >>> mu = (1 - t) * (t+1) + 1
    >>> v = x**2 + y**2
    >>> dx = sy.diff(x, t)
    >>> dy = sy.diff(y, t)
    >>> dz = sy.diff(z, t)
    >>> dl = sy.sqrt(dx **2 + dy**2 + dz**2)
    >>> integrand  = mu * v * dl
    >>> val = sy.integrate(integrand, (t, -1, 1))
    >>> float(val)
    12.490674765352512

    """

    def error_eval(nx):
        nz = ny = 4  # not really important to this test, as only one "line" is non-zero
        xlocs = np.linspace(-1, 1, nx + 1)
        nodes_x = xlocs[:, None, None] * np.ones((1, ny + 1, nz + 1))
        nodes_y = 2 * nodes_x + np.linspace(-1, 1, ny + 1)[None, :, None]
        nodes_z = (nodes_x - 1) * (nodes_x + 1) + np.linspace(-1, 1, nz + 1)[
            None, None, :
        ]

        mesh = CurvilinearMesh((nodes_x, nodes_y, nodes_z))

        # edge indices along the path... are x-edges:
        edge_inds = np.arange(mesh.n_edges_x).reshape((nz + 1, ny + 1, nx))[2, 2, :]

        prop = (1 - mesh.edges[:, 0]) * (mesh.edges[:, 0] + 1) + 1
        edge_props = np.zeros(mesh.n_edges)
        edge_props[edge_inds] = prop[edge_inds]

        ex = mesh.edges[:, 0]
        ey = mesh.edges[:, 1]
        w = ex**2 + ey**2

        M = mesh.get_edge_inner_product_line(model=edge_props)

        discrete_val = np.sum(M @ w)
        reference_value = 12.490674765352512
        print(discrete_val)
        return np.abs(discrete_val - reference_value), xlocs[1] - xlocs[0]

    assert_expected_order(error_eval, [20.0, 40.0, 80.0])


if __name__ == "__main__":
    unittest.main()
