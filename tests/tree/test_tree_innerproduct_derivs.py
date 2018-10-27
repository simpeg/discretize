import numpy as np
import unittest
import discretize

np.random.seed(50)



def doTestFace(h, rep, fast, meshType, invProp=False, invMat=False):
    if meshType == 'Curv':
        hRect = discretize.utils.exampleLrmGrid(h, 'rotate')
        mesh = discretize.CurvilinearMesh(hRect)
    elif meshType == 'Tree':
        mesh = discretize.TreeMesh(h, levels=3)
        mesh.refine(lambda xc: 3)
    elif meshType == 'Tensor':
        mesh = discretize.TensorMesh(h)
    v = np.random.rand(mesh.nF)
    sig = np.random.rand(1) if rep is 0 else np.random.rand(mesh.nC*rep)

    def fun(sig):
        M  = mesh.getFaceInnerProduct(sig, invProp=invProp, invMat=invMat)
        Md = mesh.getFaceInnerProductDeriv(sig, invProp=invProp, invMat=invMat, doFast=fast)
        return M*v, Md(v)
    print(meshType, 'Face', h, rep, fast, ('harmonic' if invProp and invMat else 'standard'))
    return discretize.Tests.checkDerivative(fun, sig, num=5, plotIt=False)



def doTestEdge(h, rep, fast, meshType, invProp=False, invMat=False):
    if meshType == 'Curv':
        hRect = discretize.utils.exampleLrmGrid(h,'rotate')
        mesh = discretize.CurvilinearMesh(hRect)
    elif meshType == 'Tree':
        mesh = discretize.TreeMesh(h, levels=3)
        mesh.refine(lambda xc: 3)
    elif meshType == 'Tensor':
        mesh = discretize.TensorMesh(h)
    v = np.random.rand(mesh.nE)
    sig = np.random.rand(1) if rep is 0 else np.random.rand(mesh.nC*rep)
    def fun(sig):
        M  = mesh.getEdgeInnerProduct(sig, invProp=invProp, invMat=invMat)
        Md = mesh.getEdgeInnerProductDeriv(sig, invProp=invProp, invMat=invMat, doFast=fast)
        return M*v, Md(v)
    print(meshType, 'Edge', h, rep, fast, ('harmonic' if invProp and invMat else 'standard'))
    return discretize.Tests.checkDerivative(fun, sig, num=5, plotIt=False)


class TestInnerProductsDerivsTensor(unittest.TestCase):

    def test_FaceIP_2D_float_Tree(self):
        self.assertTrue(doTestFace([8, 8], 0, False, 'Tree'))

    def test_FaceIP_3D_float_Tree(self):
        self.assertTrue(doTestFace([8, 8, 8], 0, False, 'Tree'))

    def test_FaceIP_2D_isotropic_Tree(self):
        self.assertTrue(doTestFace([8, 8], 1, False, 'Tree'))

    def test_FaceIP_3D_isotropic_Tree(self):
        self.assertTrue(doTestFace([8, 8, 8], 1, False, 'Tree'))

    def test_FaceIP_2D_anisotropic_Tree(self):
        self.assertTrue(doTestFace([8, 8], 2, False, 'Tree'))

    def test_FaceIP_3D_anisotropic_Tree(self):
        self.assertTrue(doTestFace([8, 8, 8], 3, False, 'Tree'))

    def test_FaceIP_2D_tensor_Tree(self):
        self.assertTrue(doTestFace([8, 8], 3, False, 'Tree'))

    def test_FaceIP_3D_tensor_Tree(self):
        self.assertTrue(doTestFace([8, 8, 8], 6, False, 'Tree'))

    def test_FaceIP_2D_float_fast_Tree(self):
        self.assertTrue(doTestFace([8, 8], 0, True, 'Tree'))

    def test_FaceIP_3D_float_fast_Tree(self):
        self.assertTrue(doTestFace([8, 8, 8], 0, True, 'Tree'))

    def test_FaceIP_2D_isotropic_fast_Tree(self):
        self.assertTrue(doTestFace([8, 8], 1, True, 'Tree'))

    def test_FaceIP_3D_isotropic_fast_Tree(self):
        self.assertTrue(doTestFace([8, 8, 8], 1, True, 'Tree'))

    def test_FaceIP_2D_anisotropic_fast_Tree(self):
        self.assertTrue(doTestFace([8, 8], 2, True, 'Tree'))

    def test_FaceIP_3D_anisotropic_fast_Tree(self):
        self.assertTrue(doTestFace([8, 8, 8], 3, True, 'Tree'))

    # def test_EdgeIP_2D_float_Tree(self):
    #     self.assertTrue(doTestEdge([8, 8], 0, False, 'Tree'))
    def test_EdgeIP_3D_float_Tree(self):
        self.assertTrue(doTestEdge([8, 8, 8], 0, False, 'Tree'))
    # def test_EdgeIP_2D_isotropic_Tree(self):
    #     self.assertTrue(doTestEdge([8, 8], 1, False, 'Tree'))

    def test_EdgeIP_3D_isotropic_Tree(self):
        self.assertTrue(doTestEdge([8, 8, 8], 1, False, 'Tree'))
    # def test_EdgeIP_2D_anisotropic_Tree(self):
    #     self.assertTrue(doTestEdge([8, 8], 2, False, 'Tree'))

    def test_EdgeIP_3D_anisotropic_Tree(self):
        self.assertTrue(doTestEdge([8, 8, 8], 3, False, 'Tree'))
    # def test_EdgeIP_2D_tensor_Tree(self):
    #     self.assertTrue(doTestEdge([8, 8], 3, False, 'Tree'))

    def test_EdgeIP_3D_tensor_Tree(self):
        self.assertTrue(doTestEdge([8, 8, 8], 6, False, 'Tree'))

    # def test_EdgeIP_2D_float_fast_Tree(self):
    #     self.assertTrue(doTestEdge([8, 8], 0, True, 'Tree'))
    def test_EdgeIP_3D_float_fast_Tree(self):
        self.assertTrue(doTestEdge([8, 8, 8], 0, True, 'Tree'))
    # def test_EdgeIP_2D_isotropic_fast_Tree(self):
    #     self.assertTrue(doTestEdge([8, 8], 1, True, 'Tree'))

    def test_EdgeIP_3D_isotropic_fast_Tree(self):
        self.assertTrue(doTestEdge([8, 8, 8], 1, True, 'Tree'))
    # def test_EdgeIP_2D_anisotropic_fast_Tree(self):
    #     self.assertTrue(doTestEdge([8, 8], 2, True, 'Tree'))

    def test_EdgeIP_3D_anisotropic_fast_Tree(self):
        self.assertTrue(doTestEdge([8, 8, 8], 3, True, 'Tree'))

if __name__ == '__main__':
    unittest.main()
