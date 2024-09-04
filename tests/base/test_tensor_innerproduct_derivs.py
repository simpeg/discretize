import numpy as np
import unittest
import discretize
from discretize import TensorMesh

rng = np.random.default_rng(542)


class TestInnerProductsDerivsTensor(unittest.TestCase):
    def doTestFace(
        self, h, rep, fast, meshType, invert_model=False, invert_matrix=False
    ):
        if meshType == "Curv":
            hRect = discretize.utils.example_curvilinear_grid(h, "rotate")
            mesh = discretize.CurvilinearMesh(hRect)
        elif meshType == "Tree":
            mesh = discretize.TreeMesh(h, levels=3)
            mesh.refine(lambda xc: 3)
            mesh.number(balance=False)
        elif meshType == "Tensor":
            mesh = discretize.TensorMesh(h)
        v = rng.random(mesh.nF)
        sig = rng.random(1) if rep == 0 else rng.random(mesh.nC * rep)

        def fun(sig):
            M = mesh.get_face_inner_product(
                sig, invert_model=invert_model, invert_matrix=invert_matrix
            )
            Md = mesh.get_face_inner_product_deriv(
                sig,
                invert_model=invert_model,
                invert_matrix=invert_matrix,
                do_fast=fast,
            )
            return M * v, Md(v)

        print(
            meshType,
            "Face",
            h,
            rep,
            fast,
            ("harmonic" if invert_model and invert_matrix else "standard"),
        )
        return discretize.tests.check_derivative(
            fun, sig, num=5, plotIt=False, random_seed=452
        )

    def doTestEdge(
        self, h, rep, fast, meshType, invert_model=False, invert_matrix=False
    ):
        if meshType == "Curv":
            hRect = discretize.utils.example_curvilinear_grid(h, "rotate")
            mesh = discretize.CurvilinearMesh(hRect)
        elif meshType == "Tree":
            mesh = discretize.TreeMesh(h, levels=3)
            mesh.refine(lambda xc: 3)
            mesh.number(balance=False)
        elif meshType == "Tensor":
            mesh = discretize.TensorMesh(h)
        v = rng.random(mesh.nE)
        sig = rng.random(1) if rep == 0 else rng.random(mesh.nC * rep)

        def fun(sig):
            M = mesh.get_edge_inner_product(
                sig, invert_model=invert_model, invert_matrix=invert_matrix
            )
            Md = mesh.get_edge_inner_product_deriv(
                sig,
                invert_model=invert_model,
                invert_matrix=invert_matrix,
                do_fast=fast,
            )
            return M * v, Md(v)

        print(
            meshType,
            "Edge",
            h,
            rep,
            fast,
            ("harmonic" if invert_model and invert_matrix else "standard"),
        )
        return discretize.tests.check_derivative(
            fun, sig, num=5, plotIt=False, random_seed=4567
        )

    def test_FaceIP_1D_float(self):
        self.assertTrue(self.doTestFace([10], 0, False, "Tensor"))

    def test_FaceIP_2D_float(self):
        self.assertTrue(self.doTestFace([10, 4], 0, False, "Tensor"))

    def test_FaceIP_3D_float(self):
        self.assertTrue(self.doTestFace([10, 4, 5], 0, False, "Tensor"))

    def test_FaceIP_1D_isotropic(self):
        self.assertTrue(self.doTestFace([10], 1, False, "Tensor"))

    def test_FaceIP_2D_isotropic(self):
        self.assertTrue(self.doTestFace([10, 4], 1, False, "Tensor"))

    def test_FaceIP_3D_isotropic(self):
        self.assertTrue(self.doTestFace([10, 4, 5], 1, False, "Tensor"))

    def test_FaceIP_2D_anisotropic(self):
        self.assertTrue(self.doTestFace([10, 4], 2, False, "Tensor"))

    def test_FaceIP_3D_anisotropic(self):
        self.assertTrue(self.doTestFace([10, 4, 5], 3, False, "Tensor"))

    def test_FaceIP_2D_tensor(self):
        self.assertTrue(self.doTestFace([10, 4], 3, False, "Tensor"))

    def test_FaceIP_3D_tensor(self):
        self.assertTrue(self.doTestFace([10, 4, 5], 6, False, "Tensor"))

    def test_FaceIP_1D_float_fast(self):
        self.assertTrue(self.doTestFace([10], 0, True, "Tensor"))

    def test_FaceIP_2D_float_fast(self):
        self.assertTrue(self.doTestFace([10, 4], 0, True, "Tensor"))

    def test_FaceIP_3D_float_fast(self):
        self.assertTrue(self.doTestFace([10, 4, 5], 0, True, "Tensor"))

    def test_FaceIP_1D_isotropic_fast(self):
        self.assertTrue(self.doTestFace([10], 1, True, "Tensor"))

    def test_FaceIP_2D_isotropic_fast(self):
        self.assertTrue(self.doTestFace([10, 4], 1, True, "Tensor"))

    def test_FaceIP_3D_isotropic_fast(self):
        self.assertTrue(self.doTestFace([10, 4, 5], 1, True, "Tensor"))

    def test_FaceIP_2D_anisotropic_fast(self):
        self.assertTrue(self.doTestFace([10, 4], 2, True, "Tensor"))

    def test_FaceIP_3D_anisotropic_fast(self):
        self.assertTrue(self.doTestFace([10, 4, 5], 3, True, "Tensor"))

    def test_EdgeIP_1D_float(self):
        self.assertTrue(self.doTestEdge([10], 0, False, "Tensor"))

    def test_EdgeIP_2D_float(self):
        self.assertTrue(self.doTestEdge([10, 4], 0, False, "Tensor"))

    def test_EdgeIP_3D_float(self):
        self.assertTrue(self.doTestEdge([10, 4, 5], 0, False, "Tensor"))

    def test_EdgeIP_1D_isotropic(self):
        self.assertTrue(self.doTestEdge([10], 1, False, "Tensor"))

    def test_EdgeIP_2D_isotropic(self):
        self.assertTrue(self.doTestEdge([10, 4], 1, False, "Tensor"))

    def test_EdgeIP_3D_isotropic(self):
        self.assertTrue(self.doTestEdge([10, 4, 5], 1, False, "Tensor"))

    def test_EdgeIP_2D_anisotropic(self):
        self.assertTrue(self.doTestEdge([10, 4], 2, False, "Tensor"))

    def test_EdgeIP_3D_anisotropic(self):
        self.assertTrue(self.doTestEdge([10, 4, 5], 3, False, "Tensor"))

    def test_EdgeIP_2D_tensor(self):
        self.assertTrue(self.doTestEdge([10, 4], 3, False, "Tensor"))

    def test_EdgeIP_3D_tensor(self):
        self.assertTrue(self.doTestEdge([10, 4, 5], 6, False, "Tensor"))

    def test_EdgeIP_1D_float_fast(self):
        self.assertTrue(self.doTestEdge([10], 0, True, "Tensor"))

    def test_EdgeIP_2D_float_fast(self):
        self.assertTrue(self.doTestEdge([10, 4], 0, True, "Tensor"))

    def test_EdgeIP_3D_float_fast(self):
        self.assertTrue(self.doTestEdge([10, 4, 5], 0, True, "Tensor"))

    def test_EdgeIP_1D_isotropic_fast(self):
        self.assertTrue(self.doTestEdge([10], 1, True, "Tensor"))

    def test_EdgeIP_2D_isotropic_fast(self):
        self.assertTrue(self.doTestEdge([10, 4], 1, True, "Tensor"))

    def test_EdgeIP_3D_isotropic_fast(self):
        self.assertTrue(self.doTestEdge([10, 4, 5], 1, True, "Tensor"))

    def test_EdgeIP_2D_anisotropic_fast(self):
        self.assertTrue(self.doTestEdge([10, 4], 2, True, "Tensor"))

    def test_EdgeIP_3D_anisotropic_fast(self):
        self.assertTrue(self.doTestEdge([10, 4, 5], 3, True, "Tensor"))

    def test_FaceIP_1D_float_fast_harmonic(self):
        self.assertTrue(
            self.doTestFace(
                [10], 0, True, "Tensor", invert_model=True, invert_matrix=True
            )
        )

    def test_FaceIP_2D_float_fast_harmonic(self):
        self.assertTrue(
            self.doTestFace(
                [10, 4], 0, True, "Tensor", invert_model=True, invert_matrix=True
            )
        )

    def test_FaceIP_3D_float_fast_harmonic(self):
        self.assertTrue(
            self.doTestFace(
                [10, 4, 5], 0, True, "Tensor", invert_model=True, invert_matrix=True
            )
        )

    def test_FaceIP_1D_isotropic_fast_harmonic(self):
        self.assertTrue(
            self.doTestFace(
                [10], 1, True, "Tensor", invert_model=True, invert_matrix=True
            )
        )

    def test_FaceIP_2D_isotropic_fast_harmonic(self):
        self.assertTrue(
            self.doTestFace(
                [10, 4], 1, True, "Tensor", invert_model=True, invert_matrix=True
            )
        )

    def test_FaceIP_3D_isotropic_fast_harmonic(self):
        self.assertTrue(
            self.doTestFace(
                [10, 4, 5], 1, True, "Tensor", invert_model=True, invert_matrix=True
            )
        )

    def test_FaceIP_2D_anisotropic_fast_harmonic(self):
        self.assertTrue(
            self.doTestFace(
                [10, 4], 2, True, "Tensor", invert_model=True, invert_matrix=True
            )
        )

    def test_FaceIP_3D_anisotropic_fast_harmonic(self):
        self.assertTrue(
            self.doTestFace(
                [10, 4, 5], 3, True, "Tensor", invert_model=True, invert_matrix=True
            )
        )

    def test_FaceIP_2D_float_Curv(self):
        self.assertTrue(self.doTestFace([10, 4], 0, False, "Curv"))

    def test_FaceIP_3D_float_Curv(self):
        self.assertTrue(self.doTestFace([10, 4, 5], 0, False, "Curv"))

    def test_FaceIP_2D_isotropic_Curv(self):
        self.assertTrue(self.doTestFace([10, 4], 1, False, "Curv"))

    def test_FaceIP_3D_isotropic_Curv(self):
        self.assertTrue(self.doTestFace([10, 4, 5], 1, False, "Curv"))

    def test_FaceIP_2D_anisotropic_Curv(self):
        self.assertTrue(self.doTestFace([10, 4], 2, False, "Curv"))

    def test_FaceIP_3D_anisotropic_Curv(self):
        self.assertTrue(self.doTestFace([10, 4, 5], 3, False, "Curv"))

    def test_FaceIP_2D_tensor_Curv(self):
        self.assertTrue(self.doTestFace([10, 4], 3, False, "Curv"))

    def test_FaceIP_3D_tensor_Curv(self):
        self.assertTrue(self.doTestFace([10, 4, 5], 6, False, "Curv"))

    def test_FaceIP_2D_float_fast_Curv(self):
        self.assertTrue(self.doTestFace([10, 4], 0, True, "Curv"))

    def test_FaceIP_3D_float_fast_Curv(self):
        self.assertTrue(self.doTestFace([10, 4, 5], 0, True, "Curv"))

    def test_FaceIP_2D_isotropic_fast_Curv(self):
        self.assertTrue(self.doTestFace([10, 4], 1, True, "Curv"))

    def test_FaceIP_3D_isotropic_fast_Curv(self):
        self.assertTrue(self.doTestFace([10, 4, 5], 1, True, "Curv"))

    def test_FaceIP_2D_anisotropic_fast_Curv(self):
        self.assertTrue(self.doTestFace([10, 4], 2, True, "Curv"))

    def test_FaceIP_3D_anisotropic_fast_Curv(self):
        self.assertTrue(self.doTestFace([10, 4, 5], 3, True, "Curv"))

    def test_EdgeIP_2D_float_Curv(self):
        self.assertTrue(self.doTestEdge([10, 4], 0, False, "Curv"))

    def test_EdgeIP_3D_float_Curv(self):
        self.assertTrue(self.doTestEdge([10, 4, 5], 0, False, "Curv"))

    def test_EdgeIP_2D_isotropic_Curv(self):
        self.assertTrue(self.doTestEdge([10, 4], 1, False, "Curv"))

    def test_EdgeIP_3D_isotropic_Curv(self):
        self.assertTrue(self.doTestEdge([10, 4, 5], 1, False, "Curv"))

    def test_EdgeIP_2D_anisotropic_Curv(self):
        self.assertTrue(self.doTestEdge([10, 4], 2, False, "Curv"))

    def test_EdgeIP_3D_anisotropic_Curv(self):
        self.assertTrue(self.doTestEdge([10, 4, 5], 3, False, "Curv"))

    def test_EdgeIP_2D_tensor_Curv(self):
        self.assertTrue(self.doTestEdge([10, 4], 3, False, "Curv"))

    def test_EdgeIP_3D_tensor_Curv(self):
        self.assertTrue(self.doTestEdge([10, 4, 5], 6, False, "Curv"))

    def test_EdgeIP_2D_float_fast_Curv(self):
        self.assertTrue(self.doTestEdge([10, 4], 0, True, "Curv"))

    def test_EdgeIP_3D_float_fast_Curv(self):
        self.assertTrue(self.doTestEdge([10, 4, 5], 0, True, "Curv"))

    def test_EdgeIP_2D_isotropic_fast_Curv(self):
        self.assertTrue(self.doTestEdge([10, 4], 1, True, "Curv"))

    def test_EdgeIP_3D_isotropic_fast_Curv(self):
        self.assertTrue(self.doTestEdge([10, 4, 5], 1, True, "Curv"))

    def test_EdgeIP_2D_anisotropic_fast_Curv(self):
        self.assertTrue(self.doTestEdge([10, 4], 2, True, "Curv"))

    def test_EdgeIP_3D_anisotropic_fast_Curv(self):
        self.assertTrue(self.doTestEdge([10, 4, 5], 3, True, "Curv"))


class TestFacePropertiesInnerProductsDerivsTensor(unittest.TestCase):
    def doTestFace(self, h, rep, meshType, invert_model=False, invert_matrix=False):
        if meshType == "Curv":
            hRect = discretize.utils.example_curvilinear_grid(h, "rotate")
            mesh = discretize.CurvilinearMesh(hRect)
        elif meshType == "Tree":
            mesh = discretize.TreeMesh(h, levels=3)
            mesh.refine(lambda xc: 3)
            mesh.number(balance=False)
        elif meshType == "Tensor":
            mesh = discretize.TensorMesh(h)
        v = rng.random(mesh.nF)
        sig = rng.random(1) if rep == 0 else rng.random(mesh.nF * rep)

        def fun(sig):
            M = mesh.get_face_inner_product_surface(
                sig, invert_model=invert_model, invert_matrix=invert_matrix
            )
            Md = mesh.get_face_inner_product_surface_deriv(
                sig, invert_model=invert_model, invert_matrix=invert_matrix
            )
            return M * v, Md(v)

        print(
            meshType,
            "Face",
            h,
            rep,
            ("harmonic" if invert_model and invert_matrix else "standard"),
        )
        return discretize.tests.check_derivative(
            fun, sig, num=5, plotIt=False, random_seed=421
        )

    def doTestEdge(self, h, rep, meshType, invert_model=False, invert_matrix=False):
        if meshType == "Curv":
            hRect = discretize.utils.example_curvilinear_grid(h, "rotate")
            mesh = discretize.CurvilinearMesh(hRect)
        elif meshType == "Tree":
            mesh = discretize.TreeMesh(h, levels=3)
            mesh.refine(lambda xc: 3)
            mesh.number(balance=False)
        elif meshType == "Tensor":
            mesh = discretize.TensorMesh(h)
        v = rng.random(mesh.nE)
        sig = rng.random(1) if rep == 0 else rng.random(mesh.nF * rep)

        def fun(sig):
            M = mesh.get_edge_inner_product_surface(
                sig, invert_model=invert_model, invert_matrix=invert_matrix
            )
            Md = mesh.get_edge_inner_product_surface_deriv(
                sig, invert_model=invert_model, invert_matrix=invert_matrix
            )
            return M * v, Md(v)

        print(
            meshType,
            "Edge",
            h,
            rep,
            ("harmonic" if invert_model and invert_matrix else "standard"),
        )
        return discretize.tests.check_derivative(
            fun, sig, num=5, plotIt=False, random_seed=31
        )

    def test_FaceIP_2D_float(self):
        self.assertTrue(self.doTestFace([10, 4], 0, "Tensor"))

    def test_FaceIP_3D_float(self):
        self.assertTrue(self.doTestFace([10, 4, 5], 0, "Tensor"))

    def test_FaceIP_2D_isotropic(self):
        self.assertTrue(self.doTestFace([10, 4], 1, "Tensor"))

    def test_FaceIP_3D_isotropic(self):
        self.assertTrue(self.doTestFace([10, 4, 5], 1, "Tensor"))

    def test_EdgeIP_2D_float(self):
        self.assertTrue(self.doTestEdge([10, 4], 0, "Tensor"))

    def test_EdgeIP_3D_float(self):
        self.assertTrue(self.doTestEdge([10, 4, 5], 0, "Tensor"))

    def test_EdgeIP_2D_isotropic(self):
        self.assertTrue(self.doTestEdge([10, 4], 1, "Tensor"))

    def test_EdgeIP_3D_isotropic(self):
        self.assertTrue(self.doTestEdge([10, 4, 5], 1, "Tensor"))

    def test_FaceIP_2D_float_invert_all(self):
        self.assertTrue(
            self.doTestFace([10, 4], 0, "Tensor", invert_model=True, invert_matrix=True)
        )

    def test_FaceIP_3D_float_invert_all(self):
        self.assertTrue(
            self.doTestFace(
                [10, 4, 5], 0, "Tensor", invert_model=True, invert_matrix=True
            )
        )

    def test_FaceIP_2D_isotropic_invert_all(self):
        self.assertTrue(
            self.doTestFace([10, 4], 1, "Tensor", invert_model=True, invert_matrix=True)
        )

    def test_FaceIP_3D_isotropic_invert_all(self):
        self.assertTrue(
            self.doTestFace(
                [10, 4, 5], 1, "Tensor", invert_model=True, invert_matrix=True
            )
        )

    def test_EdgeIP_2D_float_invert_all(self):
        self.assertTrue(
            self.doTestEdge([10, 4], 0, "Tensor", invert_model=True, invert_matrix=True)
        )

    def test_EdgeIP_3D_float_invert_all(self):
        self.assertTrue(
            self.doTestEdge(
                [10, 4, 5], 0, "Tensor", invert_model=True, invert_matrix=True
            )
        )

    def test_EdgeIP_2D_isotropic_invert_all(self):
        self.assertTrue(
            self.doTestEdge([10, 4], 1, "Tensor", invert_model=True, invert_matrix=True)
        )

    def test_EdgeIP_3D_isotropic_invert_all(self):
        self.assertTrue(
            self.doTestEdge(
                [10, 4, 5], 1, "Tensor", invert_model=True, invert_matrix=True
            )
        )


class TestEdgePropertiesInnerProductsDerivsTensor(unittest.TestCase):
    def doTestEdge(self, h, rep, meshType, invert_model=False, invert_matrix=False):
        if meshType == "Curv":
            hRect = discretize.utils.example_curvilinear_grid(h, "rotate")
            mesh = discretize.CurvilinearMesh(hRect)
        elif meshType == "Tree":
            mesh = discretize.TreeMesh(h, levels=3)
            mesh.refine(lambda xc: 3)
            mesh.number(balance=False)
        elif meshType == "Tensor":
            mesh = discretize.TensorMesh(h)
        v = rng.random(mesh.nE)
        sig = rng.random(1) if rep == 0 else rng.random(mesh.nE * rep)

        def fun(sig):
            M = mesh.get_edge_inner_product_line(
                sig,
                invert_model=invert_model,
                invert_matrix=invert_matrix,
            )
            Md = mesh.get_edge_inner_product_line_deriv(
                sig,
                invert_model=invert_model,
                invert_matrix=invert_matrix,
            )
            return M * v, Md(v)

        print(
            meshType,
            "Edge",
            h,
            rep,
            ("harmonic" if invert_model and invert_matrix else "standard"),
        )
        return discretize.tests.check_derivative(
            fun, sig, num=5, plotIt=False, random_seed=64
        )

    def test_EdgeIP_2D_float(self):
        self.assertTrue(self.doTestEdge([10, 4], 0, "Tensor"))

    def test_EdgeIP_3D_float(self):
        self.assertTrue(self.doTestEdge([10, 4, 5], 0, "Tensor"))

    def test_EdgeIP_2D_isotropic(self):
        self.assertTrue(self.doTestEdge([10, 4], 1, "Tensor"))

    def test_EdgeIP_3D_isotropic(self):
        self.assertTrue(self.doTestEdge([10, 4, 5], 1, "Tensor"))

    def test_EdgeIP_2D_float_invert_all(self):
        self.assertTrue(
            self.doTestEdge([10, 4], 0, "Tensor", invert_model=True, invert_matrix=True)
        )

    def test_EdgeIP_3D_float_invert_all(self):
        self.assertTrue(
            self.doTestEdge(
                [10, 4, 5], 0, "Tensor", invert_model=True, invert_matrix=True
            )
        )

    def test_EdgeIP_2D_isotropic_invert_all(self):
        self.assertTrue(
            self.doTestEdge([10, 4], 1, "Tensor", invert_model=True, invert_matrix=True)
        )

    def test_EdgeIP_3D_isotropic_invert_all(self):
        self.assertTrue(
            self.doTestEdge(
                [10, 4, 5], 1, "Tensor", invert_model=True, invert_matrix=True
            )
        )


class TestTensorSizeErrorRaises(unittest.TestCase):
    """Ensure exception error when model is incorrect size"""

    def setUp(self):
        self.mesh3D = TensorMesh([4, 4, 4])
        self.model = rng.random(self.mesh3D.nC)

    def test_edge_inner_product_surface_deriv(self):
        self.assertRaises(
            ValueError, self.mesh3D.get_edge_inner_product_surface_deriv, self.model
        )

    def test_face_inner_product_surface_deriv(self):
        self.assertRaises(
            ValueError, self.mesh3D.get_face_inner_product_surface_deriv, self.model
        )

    def test_edge_inner_product_line_deriv(self):
        self.assertRaises(
            ValueError, self.mesh3D.get_edge_inner_product_line_deriv, self.model
        )


class TestNone(unittest.TestCase):
    """Test None outputs"""

    def setUp(self):
        self.mesh3D = TensorMesh([4, 4, 4])

    def test_edge_inner_product_surface_deriv(self):
        self.assertIsNone(self.mesh3D.get_edge_inner_product_surface_deriv(None))

    def test_face_inner_product_surface_deriv(self):
        self.assertIsNone(self.mesh3D.get_face_inner_product_surface_deriv(None))

    def test_edge_inner_product_line_deriv(self):
        self.assertIsNone(self.mesh3D.get_edge_inner_product_line_deriv(None))
