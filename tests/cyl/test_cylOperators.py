import unittest
import numpy as np
import sympy
from sympy.abc import r, t, z

import discretize
from discretize import tests

TOL = 1e-1


# ----------------------------- Test Operators ------------------------------ #


MESHTYPES = ["uniformCylMesh"]
call2 = lambda fun, xyz: fun(xyz[:, 0], xyz[:, 2])
call3 = lambda fun, xyz: fun(xyz[:, 0], xyz[:, 1], xyz[:, 2])
cyl_row2 = lambda g, xfun, yfun: np.c_[call2(xfun, g), call2(yfun, g)]
cyl_row3 = lambda g, xfun, yfun, zfun: np.c_[
    call3(xfun, g), call3(yfun, g), call3(zfun, g)
]
cylF2 = lambda M, fx, fy: np.vstack(
    (cyl_row2(M.gridFx, fx, fy), cyl_row2(M.gridFz, fx, fy))
)
cylF3 = lambda M, fx, fy, fz: np.vstack(
    (
        cyl_row3(M.gridFx, fx, fy, fz),
        cyl_row3(M.gridFy, fx, fy, fz),
        cyl_row3(M.gridFz, fx, fy, fz),
    )
)
cylE3 = lambda M, ex, ey, ez: np.vstack(
    (
        cyl_row3(M.gridEx, ex, ey, ez),
        cyl_row3(M.gridEy, ex, ey, ez),
        cyl_row3(M.gridEz, ex, ey, ez),
    )
)


class FaceInnerProductFctsIsotropic(object):
    def fcts(self):
        j = sympy.Matrix(
            [
                r**2 * sympy.sin(t) * z,
                r * sympy.sin(t) * z,
                r * sympy.sin(t) * z**2,
            ]
        )

        # Create an isotropic sigma vector
        Sig = sympy.Matrix(
            [
                [1120 / (69 * sympy.pi) * (r * z) ** 2 * sympy.sin(t) ** 2, 0, 0],
                [0, 1120 / (69 * sympy.pi) * (r * z) ** 2 * sympy.sin(t) ** 2, 0],
                [0, 0, 1120 / (69 * sympy.pi) * (r * z) ** 2 * sympy.sin(t) ** 2],
            ]
        )

        return j, Sig

    def sol(self):
        # Do the inner product! - we are in cyl coordinates!
        j, Sig = self.fcts()
        jTSj = j.T * Sig * j
        # we are integrating in cyl coordinates
        ans = sympy.integrate(
            sympy.integrate(sympy.integrate(r * jTSj, (r, 0, 1)), (t, 0, 2 * sympy.pi)),
            (z, 0, 1),
        )[
            0
        ]  # The `[0]` is to make it a number rather than a matrix

        return ans

    def vectors(self, mesh):
        j, Sig = self.fcts()

        f_jr = sympy.lambdify((r, t, z), j[0], "numpy")
        f_jt = sympy.lambdify((r, t, z), j[1], "numpy")
        f_jz = sympy.lambdify((r, t, z), j[2], "numpy")

        f_sig = sympy.lambdify((r, t, z), Sig[0], "numpy")

        jr = f_jr(mesh.gridFx[:, 0], mesh.gridFx[:, 1], mesh.gridFx[:, 2])
        jt = f_jt(mesh.gridFy[:, 0], mesh.gridFy[:, 1], mesh.gridFy[:, 2])
        jz = f_jz(mesh.gridFz[:, 0], mesh.gridFz[:, 1], mesh.gridFz[:, 2])

        sig = f_sig(mesh.gridCC[:, 0], mesh.gridCC[:, 1], mesh.gridCC[:, 2])

        return sig, np.r_[jr, jt, jz]


class EdgeInnerProductFctsIsotropic(object):
    def fcts(self):
        h = sympy.Matrix(
            [
                r**2 * sympy.sin(t) * z,
                r * sympy.sin(t) * z,
                r * sympy.sin(t) * z**2,
            ]
        )

        # Create an isotropic sigma vector
        Sig = sympy.Matrix(
            [
                [1120 / (69 * sympy.pi) * (r * z) ** 2 * sympy.sin(t) ** 2, 0, 0],
                [0, 1120 / (69 * sympy.pi) * (r * z) ** 2 * sympy.sin(t) ** 2, 0],
                [0, 0, 1120 / (69 * sympy.pi) * (r * z) ** 2 * sympy.sin(t) ** 2],
            ]
        )

        return h, Sig

    def sol(self):
        h, Sig = self.fcts()

        hTSh = h.T * Sig * h
        ans = sympy.integrate(
            sympy.integrate(sympy.integrate(r * hTSh, (r, 0, 1)), (t, 0, 2 * sympy.pi)),
            (z, 0, 1),
        )[
            0
        ]  # The `[0]` is to make it a scalar

        return ans

    def vectors(self, mesh):
        h, Sig = self.fcts()

        f_hr = sympy.lambdify((r, t, z), h[0], "numpy")
        f_ht = sympy.lambdify((r, t, z), h[1], "numpy")
        f_hz = sympy.lambdify((r, t, z), h[2], "numpy")

        f_sig = sympy.lambdify((r, t, z), Sig[0], "numpy")

        hr = f_hr(mesh.gridEx[:, 0], mesh.gridEx[:, 1], mesh.gridEx[:, 2])
        ht = f_ht(mesh.gridEy[:, 0], mesh.gridEy[:, 1], mesh.gridEy[:, 2])
        hz = f_hz(mesh.gridEz[:, 0], mesh.gridEz[:, 1], mesh.gridEz[:, 2])

        sig = f_sig(mesh.gridCC[:, 0], mesh.gridCC[:, 1], mesh.gridCC[:, 2])

        return sig, np.r_[hr, ht, hz]


class TestCylInnerProducts_simple(unittest.TestCase):
    def setUp(self):
        n = 100.0
        self.mesh = discretize.CylindricalMesh([n, n, n])

    def test_FaceInnerProductIsotropic(self):
        # Here we will make up some j vectors that vary in space
        # j = [j_r, j_z] - to test face inner products

        fcts = FaceInnerProductFctsIsotropic()
        sig, jv = fcts.vectors(self.mesh)
        MfSig = self.mesh.get_face_inner_product(sig)
        numeric_ans = jv.T.dot(MfSig.dot(jv))

        ans = fcts.sol()

        print("------ Testing Face Inner Product-----------")
        print(
            " Analytic: {analytic}, Numeric: {numeric}, "
            "ratio (num/ana): {ratio}".format(
                analytic=ans, numeric=numeric_ans, ratio=float(numeric_ans) / ans
            )
        )
        assert np.abs(ans - numeric_ans) < TOL

    def test_EdgeInnerProduct(self):
        # Here we will make up some j vectors that vary in space
        # h = [h_t] - to test edge inner products

        fcts = EdgeInnerProductFctsIsotropic()
        sig, hv = fcts.vectors(self.mesh)
        MeSig = self.mesh.get_edge_inner_product(sig)
        numeric_ans = hv.T.dot(MeSig.dot(hv))

        ans = fcts.sol()

        print("------ Testing Edge Inner Product-----------")
        print(
            " Analytic: {analytic}, Numeric: {numeric}, "
            "ratio (num/ana): {ratio}".format(
                analytic=ans, numeric=numeric_ans, ratio=float(numeric_ans) / ans
            )
        )
        assert np.abs(ans - numeric_ans) < TOL


class TestCylFaceInnerProducts_Order(tests.OrderTest):
    meshTypes = ["uniformCylMesh"]
    meshDimension = 3

    def getError(self):
        fct = FaceInnerProductFctsIsotropic()
        sig, jv = fct.vectors(self.M)
        Msig = self.M.get_face_inner_product(sig)
        return float(fct.sol()) - jv.T.dot(Msig.dot(jv))

    def test_order(self):
        self.orderTest()


class TestCylEdgeInnerProducts_Order(tests.OrderTest):
    meshTypes = ["uniformCylMesh"]
    meshDimension = 3

    def getError(self):
        fct = EdgeInnerProductFctsIsotropic()
        sig, hv = fct.vectors(self.M)
        Msig = self.M.get_edge_inner_product(sig)
        return float(fct.sol()) - hv.T.dot(Msig.dot(hv))

    def test_order(self):
        self.orderTest()


if __name__ == "__main__":
    unittest.main()
