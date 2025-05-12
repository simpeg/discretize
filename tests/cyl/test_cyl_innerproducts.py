import discretize
from discretize import tests
import numpy as np
import sympy
from sympy.abc import r, t, z
import unittest
import scipy.sparse as sp

TOL = 1e-1
TOLD = 0.7  # tolerance on deriv checks

rng = np.random.default_rng(99)


class FaceInnerProductFctsIsotropic(object):
    """Some made up face functions to test the face inner product"""

    def fcts(self):
        j = sympy.Matrix([r**2 * z, r * z**2])

        # Create an isotropic sigma vector
        Sig = sympy.Matrix(
            [
                [420 / (sympy.pi) * (r * z) ** 2, 0],
                [0, 420 / (sympy.pi) * (r * z) ** 2],
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
        ]  # The `[0]` is to make it an int.

        return ans

    def vectors(self, mesh):
        """Get Vectors sig, sr. jx from sympy"""
        j, Sig = self.fcts()

        f_jr = sympy.lambdify((r, z), j[0], "numpy")
        f_jz = sympy.lambdify((r, z), j[1], "numpy")
        f_sigr = sympy.lambdify((r, z), Sig[0], "numpy")
        # f_sigz = sympy.lambdify((r,z), Sig[1], 'numpy')

        jr = f_jr(mesh.gridFx[:, 0], mesh.gridFx[:, 2])
        jz = f_jz(mesh.gridFz[:, 0], mesh.gridFz[:, 2])
        sigr = f_sigr(mesh.gridCC[:, 0], mesh.gridCC[:, 2])

        return sigr, np.r_[jr, jz]


class FaceInnerProductFunctionsDiagAnisotropic(FaceInnerProductFctsIsotropic):
    """
    Some made up face functions to test the diagonally anisotropic face
    inner product
    """

    def fcts(self):
        j = sympy.Matrix([r**2 * z, r * z**2])

        # Create an isotropic sigma vector
        Sig = sympy.Matrix(
            [
                [120 / (sympy.pi) * (r * z) ** 2, 0],
                [0, 420 / (sympy.pi) * (r * z) ** 2],
            ]
        )

        return j, Sig

    def vectors(self, mesh):
        """Get Vectors sig, sr. jx from sympy"""
        j, Sig = self.fcts()

        f_jr = sympy.lambdify((r, z), j[0], "numpy")
        f_jz = sympy.lambdify((r, z), j[1], "numpy")
        f_sigr = sympy.lambdify((r, z), Sig[0], "numpy")
        f_sigz = sympy.lambdify((r, z), Sig[3], "numpy")

        jr = f_jr(mesh.gridFx[:, 0], mesh.gridFx[:, 2])
        jz = f_jz(mesh.gridFz[:, 0], mesh.gridFz[:, 2])
        sigr = f_sigr(mesh.gridCC[:, 0], mesh.gridCC[:, 2])
        sigz = f_sigz(mesh.gridCC[:, 0], mesh.gridCC[:, 2])

        return np.c_[sigr, sigr, sigz], np.r_[jr, jz]


class EdgeInnerProductFctsIsotropic(object):
    """Some made up edge functions to test the edge inner product"""

    def fcts(self):
        h = sympy.Matrix([r**2 * z])

        # Create an isotropic sigma vector
        Sig = sympy.Matrix([200 / (sympy.pi) * (r * z) ** 2])

        return h, Sig

    def sol(self):
        h, Sig = self.fcts()
        # Do the inner product! - we are in cyl coordinates!
        hTSh = h.T * Sig * h
        ans = sympy.integrate(
            sympy.integrate(sympy.integrate(r * hTSh, (r, 0, 1)), (t, 0, 2 * sympy.pi)),
            (z, 0, 1),
        )[
            0
        ]  # The `[0]` is to make it an int.
        return ans

    def vectors(self, mesh):
        """Get Vectors sig, sr. jx from sympy"""
        h, Sig = self.fcts()

        f_h = sympy.lambdify((r, z), h[0], "numpy")
        f_sig = sympy.lambdify((r, z), Sig[0], "numpy")

        ht = f_h(mesh.gridEy[:, 0], mesh.gridEy[:, 2])
        sig = f_sig(mesh.gridCC[:, 0], mesh.gridCC[:, 2])

        return sig, np.r_[ht]


class EdgeInnerProductFunctionsDiagAnisotropic(EdgeInnerProductFctsIsotropic):
    """
    Some made up edge functions to test the diagonally anisotropic edge
    inner product
    """

    def vectors(self, mesh):
        h, Sig = self.fcts()

        f_h = sympy.lambdify((r, z), h[0], "numpy")
        f_sig = sympy.lambdify((r, z), Sig[0], "numpy")

        ht = f_h(mesh.gridEy[:, 0], mesh.gridEy[:, 2])
        sig = f_sig(mesh.gridCC[:, 0], mesh.gridCC[:, 2])

        return np.c_[sig, sig, sig], np.r_[ht]


class FaceInnerProductFctsFacePropertiesIsotropic(object):
    """Some made up face functions to test the face inner product"""

    def fcts(self):
        r_plane = 0.5
        z_plane = 0.5

        j_r = (r_plane**2) * z  # radial component
        j_z = r * z_plane  # vertical component

        # Create an isotropic sigma vector
        tau_r = (r_plane + z) ** 2  # r-faces
        tau_z = r + z_plane**2  # z_faces

        return j_r, j_z, tau_r, tau_z

    def sol(self):
        r_plane = 0.5

        # Do the inner product! - we are in cyl coordinates!
        j_r, j_z, tau_r, tau_z = self.fcts()

        # we are integrating in cyl coordinates
        int_r = sympy.integrate(r_plane * j_r**2 * tau_r, (z, 0, 1), (t, 0, 2 * np.pi))
        int_z = sympy.integrate(r * j_z**2 * tau_z, (r, 0, 1), (t, 0, 2 * np.pi))

        # return int_z(z_plane)
        return int_r + int_z

    def vectors(self, mesh):
        r_plane = 0.5
        z_plane = 0.5

        """Get Vectors sig, sr. jx from sympy"""
        j_r, j_z, tau_r, tau_z = self.fcts()

        fun_j_r = sympy.lambdify(z, j_r, "numpy")
        fun_j_z = sympy.lambdify(r, j_z, "numpy")
        fun_tau_r = sympy.lambdify(z, tau_r, "numpy")
        fun_tau_z = sympy.lambdify(r, tau_z, "numpy")

        eval_j_r = fun_j_r(mesh.gridFx[:, 2])
        eval_j_z = fun_j_z(mesh.gridFz[:, 0])
        eval_tau_r = 1e-12 * np.ones(mesh.nFx)
        eval_tau_z = 1e-12 * np.ones(mesh.nFz)

        k_r = np.isclose(mesh.faces_x[:, 0], r_plane)
        k_z = np.isclose(mesh.faces_z[:, 2], z_plane)

        eval_tau_r[k_r] = fun_tau_r(mesh.gridFx[k_r, 2])
        eval_tau_z[k_z] = fun_tau_z(mesh.gridFz[k_z, 0])

        return np.r_[eval_tau_r, eval_tau_z], np.r_[eval_j_r, eval_j_z]


class EdgeInnerProductFctsFacePropertiesIsotropic(object):
    """Some made up face functions to test the face inner product"""

    def fcts(self):
        r_plane = 0.5
        z_plane = 0.5

        j_t = 0.5 * r * z  # azimuthal

        # Create an isotropic sigma vector
        tau_r = (r_plane + z) ** 2  # r-faces
        tau_z = r + z_plane**2  # z_faces

        return j_t, tau_r, tau_z

    def sol(self):
        r_plane = 0.5
        z_plane = 0.5

        # Do the inner product! - we are in cyl coordinates!
        j_t, tau_r, tau_z = self.fcts()

        # we are integrating in cyl coordinates
        int_r = sympy.lambdify(
            r,
            sympy.integrate(r * j_t**2 * tau_r, (z, 0, 1), (t, 0, 2 * np.pi)),
            "numpy",
        )

        int_z = sympy.lambdify(
            z,
            sympy.integrate(r * j_t**2 * tau_z, (r, 0, 1), (t, 0, 2 * np.pi)),
            "numpy",
        )

        return int_r(r_plane) + int_z(z_plane)

    def vectors(self, mesh):
        r_plane = 0.5
        z_plane = 0.5

        """Get Vectors sig, sr. jx from sympy"""
        j_t, tau_r, tau_z = self.fcts()

        fun_j_t = sympy.lambdify((r, z), j_t, "numpy")
        fun_tau_r = sympy.lambdify(z, tau_r, "numpy")
        fun_tau_z = sympy.lambdify(r, tau_z, "numpy")

        eval_j_t = fun_j_t(mesh.gridEy[:, 0], mesh.gridEy[:, 2])
        eval_tau_r = 1e-8 * np.ones(mesh.nFx)
        eval_tau_z = 1e-8 * np.ones(mesh.nFz)

        k_r = np.isclose(mesh.faces_x[:, 0], r_plane)
        k_z = np.isclose(mesh.faces_z[:, 2], z_plane)

        eval_tau_r[k_r] = fun_tau_r(mesh.faces_x[k_r, 2])
        eval_tau_z[k_z] = fun_tau_z(mesh.faces_z[k_z, 0])

        return np.r_[eval_tau_r, eval_tau_z], eval_j_t


class TestCylInnerProducts_simple(unittest.TestCase):
    def setUp(self):
        n = 100.0
        self.mesh = discretize.CylindricalMesh([n, 1, n])

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

    def test_FaceInnerProductDiagAnisotropic(self):
        # Here we will make up some j vectors that vary in space
        # j = [j_r, j_z] - to test face inner products

        fcts = FaceInnerProductFunctionsDiagAnisotropic()
        sig, jv = fcts.vectors(self.mesh)
        MfSig = self.mesh.get_face_inner_product(sig)
        numeric_ans = jv.T.dot(MfSig.dot(jv))

        ans = fcts.sol()

        print("------ Testing Face Inner Product Anisotropic -----------")
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

    def test_EdgeInnerProductDiagAnisotropic(self):
        # Here we will make up some j vectors that vary in space
        # h = [h_t] - to test edge inner products

        fcts = EdgeInnerProductFunctionsDiagAnisotropic()

        sig, hv = fcts.vectors(self.mesh)
        MeSig = self.mesh.get_edge_inner_product(sig)
        numeric_ans = hv.T.dot(MeSig.dot(hv))

        ans = fcts.sol()

        print("------ Testing Edge Inner Product Anisotropic -----------")
        print(
            " Analytic: {analytic}, Numeric: {numeric}, "
            "ratio (num/ana): {ratio}".format(
                analytic=ans, numeric=numeric_ans, ratio=float(numeric_ans) / ans
            )
        )
        assert np.abs(ans - numeric_ans) < TOL

    def test_FaceInnerProductFacePropertiesIsotropic(self):
        # Here we will make up some j vectors that vary in space
        # j = [j_r, j_z] - to test face inner products

        fcts = FaceInnerProductFctsFacePropertiesIsotropic()
        tau, jv = fcts.vectors(self.mesh)
        Mftau = self.mesh.get_face_inner_product_surface(tau)
        numeric_ans = jv.T.dot(Mftau.dot(jv))

        ans = fcts.sol()

        print("--- Testing Face Inner Product (face properties) ---")
        print(
            " Analytic: {analytic}, Numeric: {numeric}, "
            "ratio (num/ana): {ratio}".format(
                analytic=ans, numeric=numeric_ans, ratio=float(numeric_ans) / ans
            )
        )
        assert np.abs(ans - numeric_ans) < TOL

    def test_EdgeInnerProductFacePropertiesIsotropic(self):
        # Here we will make up some j vectors that vary in space
        # j = [j_r, j_z] - to test face inner products

        fcts = EdgeInnerProductFctsFacePropertiesIsotropic()
        tau, jv = fcts.vectors(self.mesh)
        Metau = self.mesh.get_edge_inner_product_surface(tau)
        numeric_ans = jv.T.dot(Metau.dot(jv))

        ans = fcts.sol()

        print("--- Testing Edge Inner Product (face properties) ---")
        print(
            " Analytic: {analytic}, Numeric: {numeric}, "
            "ratio (num/ana): {ratio}".format(
                analytic=ans, numeric=numeric_ans, ratio=float(numeric_ans) / ans
            )
        )
        assert np.abs(ans - numeric_ans) < TOL


class TestCylFaceInnerProducts_Order(tests.OrderTest):
    meshTypes = ["uniform_symmetric_CylMesh"]
    meshDimension = 3

    def getError(self):
        fct = FaceInnerProductFctsIsotropic()
        sig, jv = fct.vectors(self.M)
        Msig = self.M.get_face_inner_product(sig)
        return float(fct.sol()) - jv.T.dot(Msig.dot(jv))

    def test_order(self):
        self.orderTest()


class TestCylEdgeInnerProducts_Order(tests.OrderTest):
    meshTypes = ["uniform_symmetric_CylMesh"]
    meshDimension = 3

    def getError(self):
        fct = EdgeInnerProductFctsIsotropic()
        sig, ht = fct.vectors(self.M)
        Msig = self.M.get_edge_inner_product(sig)
        return float(fct.sol()) - ht.T.dot(Msig.dot(ht))

    def test_order(self):
        self.orderTest()


class TestCylFaceInnerProductsDiagAnisotropic_Order(tests.OrderTest):
    meshTypes = ["uniform_symmetric_CylMesh"]
    meshDimension = 3

    def getError(self):
        fct = FaceInnerProductFunctionsDiagAnisotropic()
        sig, jv = fct.vectors(self.M)
        Msig = self.M.get_face_inner_product(sig)
        return float(fct.sol()) - jv.T.dot(Msig.dot(jv))

    def test_order(self):
        self.orderTest()


class TestCylFaceInnerProductsFaceProperties_Order(tests.OrderTest):
    meshTypes = ["uniform_symmetric_CylMesh"]
    meshDimension = 3

    def getError(self):
        fct = FaceInnerProductFctsFacePropertiesIsotropic()
        tau, jv = fct.vectors(self.M)
        Mtau = self.M.get_face_inner_product_surface(tau)
        return float(fct.sol()) - jv.T.dot(Mtau.dot(jv))

    def test_order(self):
        self.orderTest()


class TestCylEdgeInnerProductsFaceProperties_Order(tests.OrderTest):
    meshTypes = ["uniform_symmetric_CylMesh"]
    meshDimension = 3

    def getError(self):
        fct = EdgeInnerProductFctsFacePropertiesIsotropic()
        tau, jv = fct.vectors(self.M)
        Mtau = self.M.get_edge_inner_product_surface(tau)
        return float(fct.sol()) - jv.T.dot(Mtau.dot(jv))

    def test_order(self):
        self.orderTest()


class TestCylInnerProducts_Deriv(unittest.TestCase):
    def setUp(self):
        n = 2
        self.mesh = discretize.CylindricalMesh([n, 1, n])
        self.face_vec = rng.random(self.mesh.nF)
        self.edge_vec = rng.random(self.mesh.nE)
        # make up a smooth function
        self.x0 = 2 * self.mesh.gridCC[:, 0] ** 2 + self.mesh.gridCC[:, 2] ** 4

    def test_FaceInnerProductIsotropicDeriv(self):
        def fun(x):
            MfSig = self.mesh.get_face_inner_product(x)
            MfSigDeriv = self.mesh.get_face_inner_product_deriv(self.x0)
            return MfSig * self.face_vec, MfSigDeriv(self.face_vec)

        print("Testing FaceInnerProduct Isotropic")
        return self.assertTrue(
            tests.check_derivative(
                fun, self.x0, num=7, tolerance=TOLD, plotIt=False, random_seed=532
            )
        )

    def test_FaceInnerProductIsotropicDerivInvProp(self):
        def fun(x):
            MfSig = self.mesh.get_face_inner_product(x, invert_model=True)
            MfSigDeriv = self.mesh.get_face_inner_product_deriv(
                self.x0, invert_model=True
            )
            return MfSig * self.face_vec, MfSigDeriv(self.face_vec)

        print("Testing FaceInnerProduct Isotropic InvProp")
        return self.assertTrue(
            tests.check_derivative(
                fun, self.x0, num=7, tolerance=TOLD, plotIt=False, random_seed=75
            )
        )

    def test_FaceInnerProductIsotropicDerivInvMat(self):
        def fun(x):
            MfSig = self.mesh.get_face_inner_product(x, invert_matrix=True)
            MfSigDeriv = self.mesh.get_face_inner_product_deriv(
                self.x0, invert_matrix=True
            )
            return MfSig * self.face_vec, MfSigDeriv(self.face_vec)

        print("Testing FaceInnerProduct Isotropic InvMat")
        return self.assertTrue(
            tests.check_derivative(
                fun, self.x0, num=7, tolerance=TOLD, plotIt=False, random_seed=1
            )
        )

    def test_FaceInnerProductIsotropicDerivInvPropInvMat(self):
        def fun(x):
            MfSig = self.mesh.get_face_inner_product(
                x, invert_model=True, invert_matrix=True
            )
            MfSigDeriv = self.mesh.get_face_inner_product_deriv(
                self.x0, invert_model=True, invert_matrix=True
            )
            return MfSig * self.face_vec, MfSigDeriv(self.face_vec)

        print("Testing FaceInnerProduct Isotropic InvProp InvMat")
        return self.assertTrue(
            tests.check_derivative(
                fun, self.x0, num=7, tolerance=TOLD, plotIt=False, random_seed=74
            )
        )

    def test_EdgeInnerProductIsotropicDeriv(self):
        def fun(x):
            MeSig = self.mesh.get_edge_inner_product(x)
            MeSigDeriv = self.mesh.get_edge_inner_product_deriv(self.x0)
            return MeSig * self.edge_vec, MeSigDeriv(self.edge_vec)

        print("Testing EdgeInnerProduct Isotropic")
        return self.assertTrue(
            tests.check_derivative(
                fun, self.x0, num=7, tolerance=TOLD, plotIt=False, random_seed=345
            )
        )

    def test_EdgeInnerProductIsotropicDerivInvProp(self):
        def fun(x):
            MeSig = self.mesh.get_edge_inner_product(x, invert_model=True)
            MeSigDeriv = self.mesh.get_edge_inner_product_deriv(
                self.x0, invert_model=True
            )
            return MeSig * self.edge_vec, MeSigDeriv(self.edge_vec)

        print("Testing EdgeInnerProduct Isotropic InvProp")
        return self.assertTrue(
            tests.check_derivative(
                fun, self.x0, num=7, tolerance=TOLD, plotIt=False, random_seed=643
            )
        )

    def test_EdgeInnerProductIsotropicDerivInvMat(self):
        def fun(x):
            MeSig = self.mesh.get_edge_inner_product(x, invert_matrix=True)
            MeSigDeriv = self.mesh.get_edge_inner_product_deriv(
                self.x0, invert_matrix=True
            )
            return MeSig * self.edge_vec, MeSigDeriv(self.edge_vec)

        print("Testing EdgeInnerProduct Isotropic InvMat")
        return self.assertTrue(
            tests.check_derivative(
                fun, self.x0, num=7, tolerance=TOLD, plotIt=False, random_seed=363
            )
        )

    def test_EdgeInnerProductIsotropicDerivInvPropInvMat(self):
        def fun(x):
            MeSig = self.mesh.get_edge_inner_product(
                x, invert_model=True, invert_matrix=True
            )
            MeSigDeriv = self.mesh.get_edge_inner_product_deriv(
                self.x0, invert_model=True, invert_matrix=True
            )
            return MeSig * self.edge_vec, MeSigDeriv(self.edge_vec)

        print("Testing EdgeInnerProduct Isotropic InvProp InvMat")
        return self.assertTrue(
            tests.check_derivative(
                fun, self.x0, num=7, tolerance=TOLD, plotIt=False, random_seed=773
            )
        )


class TestCylInnerProductsAnisotropic_Deriv(unittest.TestCase):
    def setUp(self):
        n = 60
        self.mesh = discretize.CylindricalMesh([n, 1, n])
        self.face_vec = rng.random(self.mesh.nF)
        self.edge_vec = rng.random(self.mesh.nE)
        # make up a smooth function
        self.x0 = np.array(
            [2 * self.mesh.gridCC[:, 0] ** 2 + self.mesh.gridCC[:, 2] ** 4]
        )

    def test_FaceInnerProductAnisotropicDeriv(self):
        def fun(x):
            # fake anisotropy (tests anistropic implementation with isotropic
            # vector). First order behavior expected for fully anisotropic
            x = np.repeat(np.atleast_2d(x), 3, axis=0).T
            x0 = np.repeat(self.x0, 3, axis=0).T

            zero = sp.csr_matrix((self.mesh.nC, self.mesh.nC))
            eye = sp.eye(self.mesh.nC)
            P = sp.vstack([sp.hstack([eye, zero, eye])])

            MfSig = self.mesh.get_face_inner_product(x)
            MfSigDeriv = self.mesh.get_face_inner_product_deriv(x0)
            return MfSig * self.face_vec, MfSigDeriv(self.face_vec) * P.T

        print("Testing FaceInnerProduct Anisotropic")
        return self.assertTrue(
            tests.check_derivative(
                fun, self.x0, num=7, tolerance=TOLD, plotIt=False, random_seed=2436
            )
        )

    def test_FaceInnerProductAnisotropicDerivInvProp(self):
        def fun(x):
            x = np.repeat(np.atleast_2d(x), 3, axis=0).T
            x0 = np.repeat(self.x0, 3, axis=0).T

            zero = sp.csr_matrix((self.mesh.nC, self.mesh.nC))
            eye = sp.eye(self.mesh.nC)
            P = sp.vstack([sp.hstack([eye, zero, eye])])

            MfSig = self.mesh.get_face_inner_product(x, invert_model=True)
            MfSigDeriv = self.mesh.get_face_inner_product_deriv(x0, invert_model=True)
            return MfSig * self.face_vec, MfSigDeriv(self.face_vec) * P.T

        print("Testing FaceInnerProduct Anisotropic InvProp")
        return self.assertTrue(
            tests.check_derivative(
                fun, self.x0, num=7, tolerance=TOLD, plotIt=False, random_seed=634
            )
        )

    def test_FaceInnerProductAnisotropicDerivInvMat(self):
        def fun(x):
            x = np.repeat(np.atleast_2d(x), 3, axis=0).T
            x0 = np.repeat(self.x0, 3, axis=0).T

            zero = sp.csr_matrix((self.mesh.nC, self.mesh.nC))
            eye = sp.eye(self.mesh.nC)
            P = sp.vstack([sp.hstack([eye, zero, eye])])

            MfSig = self.mesh.get_face_inner_product(x, invert_matrix=True)
            MfSigDeriv = self.mesh.get_face_inner_product_deriv(x0, invert_matrix=True)
            return MfSig * self.face_vec, MfSigDeriv(self.face_vec) * P.T

        print("Testing FaceInnerProduct Anisotropic InvMat")
        return self.assertTrue(
            tests.check_derivative(
                fun, self.x0, num=7, tolerance=TOLD, plotIt=False, random_seed=222
            )
        )

    def test_FaceInnerProductAnisotropicDerivInvPropInvMat(self):
        def fun(x):
            x = np.repeat(np.atleast_2d(x), 3, axis=0).T
            x0 = np.repeat(self.x0, 3, axis=0).T

            zero = sp.csr_matrix((self.mesh.nC, self.mesh.nC))
            eye = sp.eye(self.mesh.nC)
            P = sp.vstack([sp.hstack([eye, zero, eye])])

            MfSig = self.mesh.get_face_inner_product(
                x, invert_model=True, invert_matrix=True
            )
            MfSigDeriv = self.mesh.get_face_inner_product_deriv(
                x0, invert_model=True, invert_matrix=True
            )
            return MfSig * self.face_vec, MfSigDeriv(self.face_vec) * P.T

        print("Testing FaceInnerProduct Anisotropic InvProp InvMat")
        return self.assertTrue(
            tests.check_derivative(
                fun, self.x0, num=7, tolerance=TOLD, plotIt=False, random_seed=654
            )
        )

    def test_EdgeInnerProductAnisotropicDeriv(self):
        def fun(x):
            x = np.repeat(np.atleast_2d(x), 3, axis=0).T
            x0 = np.repeat(self.x0, 3, axis=0).T

            zero = sp.csr_matrix((self.mesh.nC, self.mesh.nC))
            eye = sp.eye(self.mesh.nC)
            P = sp.vstack([sp.hstack([zero, eye, zero])])

            MeSig = self.mesh.get_edge_inner_product(x.reshape(self.mesh.nC, 3))
            MeSigDeriv = self.mesh.get_edge_inner_product_deriv(x0)
            return MeSig * self.edge_vec, MeSigDeriv(self.edge_vec) * P.T

        print("Testing EdgeInnerProduct Anisotropic")
        return self.assertTrue(
            tests.check_derivative(
                fun, self.x0, num=7, tolerance=TOLD, plotIt=False, random_seed=7754
            )
        )

    def test_EdgeInnerProductAnisotropicDerivInvProp(self):
        def fun(x):
            x = np.repeat(np.atleast_2d(x), 3, axis=0).T
            x0 = np.repeat(self.x0, 3, axis=0).T

            zero = sp.csr_matrix((self.mesh.nC, self.mesh.nC))
            eye = sp.eye(self.mesh.nC)
            P = sp.vstack([sp.hstack([zero, eye, zero])])

            MeSig = self.mesh.get_edge_inner_product(x, invert_model=True)
            MeSigDeriv = self.mesh.get_edge_inner_product_deriv(x0, invert_model=True)
            return MeSig * self.edge_vec, MeSigDeriv(self.edge_vec) * P.T

        print("Testing EdgeInnerProduct Anisotropic InvProp")
        return self.assertTrue(
            tests.check_derivative(
                fun, self.x0, num=7, tolerance=TOLD, plotIt=False, random_seed=1164
            )
        )

    def test_EdgeInnerProductAnisotropicDerivInvMat(self):
        def fun(x):
            x = np.repeat(np.atleast_2d(x), 3, axis=0).T
            x0 = np.repeat(self.x0, 3, axis=0).T

            zero = sp.csr_matrix((self.mesh.nC, self.mesh.nC))
            eye = sp.eye(self.mesh.nC)
            P = sp.vstack([sp.hstack([zero, eye, zero])])

            MeSig = self.mesh.get_edge_inner_product(x, invert_matrix=True)
            MeSigDeriv = self.mesh.get_edge_inner_product_deriv(x0, invert_matrix=True)
            return MeSig * self.edge_vec, MeSigDeriv(self.edge_vec) * P.T

        print("Testing EdgeInnerProduct Anisotropic InvMat")
        return self.assertTrue(
            tests.check_derivative(
                fun, self.x0, num=7, tolerance=TOLD, plotIt=False, random_seed=643
            )
        )

    def test_EdgeInnerProductAnisotropicDerivInvPropInvMat(self):
        def fun(x):
            x = np.repeat(np.atleast_2d(x), 3, axis=0).T
            x0 = np.repeat(self.x0, 3, axis=0).T

            zero = sp.csr_matrix((self.mesh.nC, self.mesh.nC))
            eye = sp.eye(self.mesh.nC)
            P = sp.vstack([sp.hstack([zero, eye, zero])])

            MeSig = self.mesh.get_edge_inner_product(
                x, invert_model=True, invert_matrix=True
            )
            MeSigDeriv = self.mesh.get_edge_inner_product_deriv(
                x0, invert_model=True, invert_matrix=True
            )
            return MeSig * self.edge_vec, MeSigDeriv(self.edge_vec) * P.T

        print("Testing EdgeInnerProduct Anisotropic InvProp InvMat")
        return self.assertTrue(
            tests.check_derivative(
                fun, self.x0, num=7, tolerance=TOLD, plotIt=False, random_seed=8654
            )
        )


class TestCylInnerProductsFaceProperties_Deriv(unittest.TestCase):
    def setUp(self):
        n = 2
        self.mesh = discretize.CylindricalMesh([n, 1, n])
        self.face_vec = rng.random(self.mesh.nF)
        self.edge_vec = rng.random(self.mesh.nE)
        # make up a smooth function
        self.x0 = np.r_[
            2 * self.mesh.gridFx[:, 0] ** 2 + self.mesh.gridFx[:, 2] ** 4,
            2 * self.mesh.gridFz[:, 0] ** 2 + self.mesh.gridFz[:, 2] ** 4,
        ]

    def test_FaceInnerProductIsotropicDeriv(self):
        def fun(x):
            MfTau = self.mesh.get_face_inner_product_surface(x)
            MfTauDeriv = self.mesh.get_face_inner_product_surface_deriv(self.x0)
            return MfTau * self.face_vec, MfTauDeriv(self.face_vec)

        print("Testing FaceInnerProduct Isotropic (Face Properties)")
        return self.assertTrue(
            tests.check_derivative(
                fun, self.x0, num=7, tolerance=TOLD, plotIt=False, random_seed=234
            )
        )

    def test_FaceInnerProductIsotropicDerivInvProp(self):
        def fun(x):
            MfTau = self.mesh.get_face_inner_product_surface(x, invert_model=True)
            MfTauDeriv = self.mesh.get_face_inner_product_surface_deriv(
                self.x0, invert_model=True
            )
            return MfTau * self.face_vec, MfTauDeriv(self.face_vec)

        print("Testing FaceInnerProduct Isotropic InvProp (Face Properties)")
        return self.assertTrue(
            tests.check_derivative(
                fun, self.x0, num=7, tolerance=TOLD, plotIt=False, random_seed=7543
            )
        )

    def test_FaceInnerProductIsotropicDerivInvMat(self):
        def fun(x):
            MfTau = self.mesh.get_face_inner_product_surface(x, invert_matrix=True)
            MfTauDeriv = self.mesh.get_face_inner_product_surface_deriv(
                self.x0, invert_matrix=True
            )
            return MfTau * self.face_vec, MfTauDeriv(self.face_vec)

        print("Testing FaceInnerProduct Isotropic InvMat (Face Properties)")
        return self.assertTrue(
            tests.check_derivative(
                fun, self.x0, num=7, tolerance=TOLD, plotIt=False, random_seed=2745725
            )
        )

    def test_EdgeInnerProductIsotropicDeriv(self):
        def fun(x):
            MeTau = self.mesh.get_edge_inner_product_surface(x)
            MeTauDeriv = self.mesh.get_edge_inner_product_surface_deriv(self.x0)
            return MeTau * self.edge_vec, MeTauDeriv(self.edge_vec)

        print("Testing EdgeInnerProduct Isotropic (Face Properties)")
        return self.assertTrue(
            tests.check_derivative(
                fun, self.x0, num=7, tolerance=TOLD, plotIt=False, random_seed=6654
            )
        )

    def test_EdgeInnerProductIsotropicDerivInvProp(self):
        def fun(x):
            MeTau = self.mesh.get_edge_inner_product_surface(x, invert_model=True)
            MeTauDeriv = self.mesh.get_edge_inner_product_surface_deriv(
                self.x0, invert_model=True
            )
            return MeTau * self.edge_vec, MeTauDeriv(self.edge_vec)

        print("Testing EdgeInnerProduct Isotropic InvProp (Face Properties)")
        return self.assertTrue(
            tests.check_derivative(
                fun, self.x0, num=7, tolerance=TOLD, plotIt=False, random_seed=4564
            )
        )

    def test_EdgeInnerProductIsotropicDerivInvMat(self):
        def fun(x):
            MeTau = self.mesh.get_edge_inner_product_surface(x, invert_matrix=True)
            MeTauDeriv = self.mesh.get_edge_inner_product_surface_deriv(
                self.x0, invert_matrix=True
            )
            return MeTau * self.edge_vec, MeTauDeriv(self.edge_vec)

        print("Testing EdgeInnerProduct Isotropic InvMat (Face Properties)")
        return self.assertTrue(
            tests.check_derivative(
                fun, self.x0, num=7, tolerance=TOLD, plotIt=False, random_seed=2355
            )
        )
