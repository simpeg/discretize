from __future__ import print_function
import unittest
import numpy as np
import sympy
from sympy.abc import r, t, z

import discretize
from discretize import Tests

np.random.seed(16)

TOL = 1e-1


# ----------------------------- Test Operators ------------------------------ #


MESHTYPES = ['uniformCylMesh', 'randomCylMesh']
call2 = lambda fun, xyz: fun(xyz[:, 0], xyz[:, 2])
call3 = lambda fun, xyz: fun(xyz[:, 0], xyz[:, 1], xyz[:, 2])
cyl_row2 = lambda g, xfun, yfun: np.c_[call2(xfun, g), call2(yfun, g)]
cyl_row3 = lambda g, xfun, yfun, zfun: np.c_[
    call3(xfun, g), call3(yfun, g), call3(zfun, g)
]
cylF2 = lambda M, fx, fy: np.vstack((
    cyl_row2(M.gridFx, fx, fy), cyl_row2(M.gridFz, fx, fy)
))
cylF3 = lambda M, fx, fy, fz: np.vstack((
    cyl_row3(M.gridFx, fx, fy, fz),
    cyl_row3(M.gridFy, fx, fy, fz),
    cyl_row3(M.gridFz, fx, fy, fz)
))
cylE3 = lambda M, ex, ey, ez: np.vstack((
    cyl_row3(M.gridEx, ex, ey, ez),
    cyl_row3(M.gridEy, ex, ey, ez),
    cyl_row3(M.gridEz, ex, ey, ez)
))


# class TestCellGradx3D(Tests.OrderTest):
#     name = "CellGradx"
#     MESHTYPES = MESHTYPES
#     meshDimension = 3
#     meshSizes = [8, 16, 32, 64]

#     def getError(self):

#         fun = lambda r, t, z: (
#             np.sin(2.*np.pi*r) + np.sin(t) + np.sin(2*np.pi*z)
#         )

#         solR = lambda r, t, z: 2.*np.pi*np.cos(2.*np.pi*r)

#         phi = call3(fun, self.M.gridCC)

#         phix_num = self.M.cellGradx * phi
#         phix_ana = call3(solR, self.M.gridFx)

#         err = np.linalg.norm(phix_num - phix_ana, np.inf)
#         return err

#     def test_order(self):
#         self.orderTest()


class TestFaceDiv3D(Tests.OrderTest):
    name = "FaceDiv"
    meshTypes = MESHTYPES
    meshDimension = 3
    meshSizes = [8, 16, 32, 64]

    def getError(self):

        funR = lambda r, t, z: np.sin(2.*np.pi*r)
        funT = lambda r, t, z: r*np.exp(-r)*np.sin(t) #* np.sin(2.*np.pi*r)
        funZ = lambda r, t, z: np.sin(2.*np.pi*z)

        sol = lambda r, t, z: (
            (2*np.pi*r*np.cos(2*np.pi*r) + np.sin(2*np.pi*r))/r +
            np.exp(-r)*np.cos(t) +
            2*np.pi*np.cos(2*np.pi*z)
        )

        Fc = cylF3(self.M, funR, funT, funZ)
        # Fc = np.c_[Fc[:, 0], np.zeros(self.M.nF), Fc[:, 1]]
        F = self.M.projectFaceVector(Fc)

        divF = self.M.faceDiv.dot(F)
        divF_ana = call3(sol, self.M.gridCC)

        err = np.linalg.norm((divF-divF_ana), np.inf)
        return err

    def test_order(self):
        self.orderTest()


class TestEdgeCurl3D(Tests.OrderTest):

    name = "edgeCurl"
    meshTypes = MESHTYPES
    meshDimension = 3
    meshSizes = [8, 16, 32, 64]

    def getError(self):

        # use the same function in r, t, z
        # need to pick functions that make sense at the axis of symmetry
        # careful that r, theta contributions make sense at axis of symmetry

        funR = lambda r, t, z: np.sin(2*np.pi*z) * np.sin(np.pi*r) * np.sin(t)
        funT = lambda r, t, z: np.cos(np.pi*z) * np.sin(np.pi*r) * np.sin(t)
        funZ = lambda r, t, z: np.sin(np.pi*r) * np.sin(t)

        derivR_t = lambda r, t, z: np.sin(2*np.pi*z) * np.sin(np.pi*r) * np.cos(t)
        derivR_z = lambda r, t, z: 2*np.pi * np.cos(2*np.pi*z) * np.sin(np.pi*r) * np.sin(t)

        derivT_r = lambda r, t, z: np.pi * np.cos(np.pi*z) * np.cos(np.pi*r) * np.sin(t)
        derivT_z = lambda r, t, z: -np.pi * np.sin(np.pi*z) * np.sin(np.pi*r) * np.sin(t)

        derivZ_r = lambda r, t, z: np.pi*np.cos(np.pi*r) * np.sin(t)
        derivZ_t = lambda r, t, z: np.sin(np.pi*r) * np.cos(t)

        sol_r = lambda r, t, z: 1./r * derivZ_t(r, t, z) - derivT_z(r, t, z)
        sol_t = lambda r, t, z: derivR_z(r, t, z) - derivZ_r(r, t, z)
        sol_z = lambda r, t, z: 1./r * ( r * derivT_r(r, t, z) + funT(r, t, z) - derivR_t(r, t, z))


        Ec = cylE3(self.M, funR, funT, funZ)
        E = self.M.projectEdgeVector(Ec)
        curlE_num = self.M.edgeCurl * E

        Fc = cylF3(self.M, sol_r, sol_t, sol_z)
        curlE_ana = self.M.projectFaceVector(Fc)

        err = np.linalg.norm((curlE_num-curlE_ana), np.inf)
        return err

    def test_order(self):
        self.orderTest()


class TestAverageSimple(unittest.TestCase):

    def setUp(self):
        n = 10
        hx = np.random.rand(n)
        hy = np.random.rand(n)
        hy = hy * 2 * np.pi / hy.sum()
        hz = np.random.rand(n)
        self.mesh = discretize.CylMesh([hx, hy, hz])

    def test_constantEdges(self):
        funR = lambda r, t, z: r
        funT = lambda r, t, z: r  # theta edges don't exist at the center of the mesh
        funZ = lambda r, t, z: z

        Ec = cylE3(self.mesh, funR, funT, funZ)
        E = self.mesh.projectEdgeVector(Ec)

        aveE = self.mesh.aveE2CCV * E

        aveE_anaR = funR(
            self.mesh.gridCC[:, 0], self.mesh.gridCC[:, 1], self.mesh.gridCC[:, 2]
        )
        aveE_anaT = funT(
            self.mesh.gridCC[:, 0], self.mesh.gridCC[:, 1], self.mesh.gridCC[:, 2]
        )
        aveE_anaZ = funZ(
            self.mesh.gridCC[:, 0], self.mesh.gridCC[:, 1], self.mesh.gridCC[:, 2]
        )

        aveE_ana = np.hstack([aveE_anaR, aveE_anaT, aveE_anaZ])

        assert np.linalg.norm(aveE - aveE_ana) < 1e-10

    def test_simplefct(self):
        funR = lambda r, t, z: r
        funT = lambda r, t, z: r
        funZ = lambda r, t, z: z

        Fc = cylF3(self.mesh, funR, funT, funZ)
        F = self.mesh.projectFaceVector(Fc)

        aveF = self.mesh.aveF2CCV * F

        aveF_anaR = funR(
            self.mesh.gridCC[:, 0], self.mesh.gridCC[:, 1], self.mesh.gridCC[:, 2]
        )
        aveF_anaT = funT(
         self.mesh.gridCC[:, 0], self.mesh.gridCC[:, 1], self.mesh.gridCC[:, 2]
        )
        aveF_anaZ = funZ(
            self.mesh.gridCC[:, 0], self.mesh.gridCC[:, 1], self.mesh.gridCC[:, 2]
        )

        aveF_ana = np.hstack([aveF_anaR, aveF_anaT, aveF_anaZ])

        assert np.linalg.norm(aveF - aveF_ana) < 1e-10


class TestAveF2CCV(Tests.OrderTest):
    name = "aveF2CCV"
    meshTypes = MESHTYPES
    meshSizes = [8, 16, 32, 64]
    meshDimension = 3
    expectedOrders = 2

    def getError(self):

        funR = lambda r, t, z: np.sin(2*np.pi*z) * np.sin(np.pi*r) * np.sin(t)
        funT = lambda r, t, z: np.sin(np.pi*z) * np.sin(np.pi*r) * np.sin(2*t)
        funZ = lambda r, t, z: np.sin(np.pi*z) * np.sin(2*np.pi*r) * np.sin(t)

        Fc = cylF3(self.M, funR, funT, funZ)
        F = self.M.projectFaceVector(Fc)

        aveF = self.M.aveF2CCV * F

        aveF_anaR = funR(
            self.M.gridCC[:, 0], self.M.gridCC[:, 1], self.M.gridCC[:, 2]
        )
        aveF_anaT = funT(
            self.M.gridCC[:, 0], self.M.gridCC[:, 1], self.M.gridCC[:, 2]
        )
        aveF_anaZ = funZ(
            self.M.gridCC[:, 0], self.M.gridCC[:, 1], self.M.gridCC[:, 2]
        )

        aveF_ana = np.hstack([aveF_anaR, aveF_anaT, aveF_anaZ])

        err = np.linalg.norm((aveF[:self.M.vnF[0]]-aveF_anaR), np.inf)
        return err

    def test_order(self):
        self.orderTest()


class TestAveF2CC(Tests.OrderTest):
    name = "aveF2CC"
    meshTypes = MESHTYPES
    meshSizes = [8, 16, 32, 64]
    meshDimension = 3
    expectedOrders = 2  # the averaging does not account for differences in theta edge lengths (inner product does though)

    def getError(self):

        fun = lambda r, t, z: np.sin(np.pi*z) * np.sin(np.pi*r) * np.sin(t)

        Fc = cylF3(self.M, fun, fun, fun)
        F = self.M.projectFaceVector(Fc)

        aveF = self.M.aveF2CC * F
        aveF_ana = fun(
            self.M.gridCC[:, 0], self.M.gridCC[:, 1], self.M.gridCC[:, 2]
        )

        err = np.linalg.norm((aveF-aveF_ana), np.inf)
        return err

    def test_order(self):
        self.orderTest()


class FaceInnerProductFctsIsotropic(object):

    def fcts(self):

        j = sympy.Matrix([
            r**2 * sympy.sin(t) * z,
            r * sympy.sin(t) * z,
            r * sympy.sin(t) * z**2,
        ])

        # Create an isotropic sigma vector
        Sig = sympy.Matrix([
            [1120/(69*sympy.pi)*(r*z)**2 * sympy.sin(t)**2, 0, 0],
            [0, 1120/(69*sympy.pi)*(r*z)**2 * sympy.sin(t)**2, 0],
            [0, 0, 1120/(69*sympy.pi)*(r*z)**2 * sympy.sin(t)**2]
        ])

        return j, Sig

    def sol(self):
        # Do the inner product! - we are in cyl coordinates!
        j, Sig = self.fcts()
        jTSj = j.T*Sig*j
        # we are integrating in cyl coordinates
        ans = sympy.integrate(
            sympy.integrate(
                sympy.integrate(r * jTSj, (r, 0, 1)),
                (t, 0, 2*sympy.pi)
            ),
            (z, 0, 1)
        )[0] # The `[0]` is to make it a number rather than a matrix

        return ans

    def vectors(self, mesh):
        j, Sig = self.fcts()

        f_jr = sympy.lambdify((r, t, z), j[0], 'numpy')
        f_jt = sympy.lambdify((r, t, z), j[1], 'numpy')
        f_jz = sympy.lambdify((r, t, z), j[2], 'numpy')

        f_sig = sympy.lambdify((r, t, z), Sig[0], 'numpy')

        jr = f_jr(mesh.gridFx[:, 0], mesh.gridFx[:, 1], mesh.gridFx[:, 2])
        jt = f_jt(mesh.gridFy[:, 0], mesh.gridFy[:, 1], mesh.gridFy[:, 2])
        jz = f_jz(mesh.gridFz[:, 0], mesh.gridFz[:, 1], mesh.gridFz[:, 2])

        sig = f_sig(mesh.gridCC[:, 0], mesh.gridCC[:, 1], mesh.gridCC[:, 2])

        return sig, np.r_[jr, jt, jz]


class TestAveE2CCV(Tests.OrderTest):
    name = "aveE2CCV"
    meshTypes = MESHTYPES
    meshSizes = [8, 16, 32, 64]
    meshDimension = 3

    def getError(self):

        funR = lambda r, t, z: np.sin(np.pi*z) * np.sin(np.pi*r) * np.sin(t)
        funT = lambda r, t, z: np.sin(np.pi*z) * np.sin(np.pi*r) * np.sin(t)
        funZ = lambda r, t, z: np.sin(np.pi*z) * np.sin(np.pi*r) * np.sin(t)

        Ec = cylE3(self.M, funR, funT, funZ)
        E = self.M.projectEdgeVector(Ec)

        aveE = self.M.aveE2CCV * E

        aveE_anaR = funR(
            self.M.gridCC[:, 0], self.M.gridCC[:, 1], self.M.gridCC[:, 2]
        )
        aveE_anaT = funT(
            self.M.gridCC[:, 0], self.M.gridCC[:, 1], self.M.gridCC[:, 2]
        )
        aveE_anaZ = funZ(
            self.M.gridCC[:, 0], self.M.gridCC[:, 1], self.M.gridCC[:, 2]
        )

        aveE_ana = np.hstack([aveE_anaR, aveE_anaT, aveE_anaZ])

        err = np.linalg.norm((aveE-aveE_ana), np.inf)
        return err

    def test_order(self):
        self.orderTest()


class TestAveE2CC(Tests.OrderTest):
    name = "aveE2CC"
    meshTypes = MESHTYPES
    meshSizes = [8, 16, 32, 64]
    meshDimension = 3

    def getError(self):

        fun = lambda r, t, z: np.sin(np.pi*z) * np.sin(np.pi*r) * np.sin(t)

        Ec = cylE3(self.M, fun, fun, fun)
        E = self.M.projectEdgeVector(Ec)

        aveE = self.M.aveE2CC * E
        aveE_ana = fun(
            self.M.gridCC[:, 0], self.M.gridCC[:, 1], self.M.gridCC[:, 2]
        )

        err = np.linalg.norm((aveE-aveE_ana), np.inf)
        return err

    def test_order(self):
        self.orderTest()


class EdgeInnerProductFctsIsotropic(object):
    def fcts(self):
        h = sympy.Matrix([
            r**2 * sympy.sin(t) * z,
            r * sympy.sin(t) * z,
            r * sympy.sin(t) * z**2,
        ])

        # Create an isotropic sigma vector
        Sig = sympy.Matrix([
            [1120/(69*sympy.pi)*(r*z)**2 * sympy.sin(t)**2, 0, 0],
            [0, 1120/(69*sympy.pi)*(r*z)**2 * sympy.sin(t)**2, 0],
            [0, 0, 1120/(69*sympy.pi)*(r*z)**2 * sympy.sin(t)**2]
        ])

        return h, Sig

    def sol(self):
        h, Sig = self.fcts()

        hTSh = h.T*Sig*h
        ans  = sympy.integrate(
            sympy.integrate(
                sympy.integrate(r * hTSh, (r, 0, 1)),
                (t, 0, 2*sympy.pi)),
            (z, 0, 1)
        )[0] # The `[0]` is to make it a scalar

        return ans

    def vectors(self, mesh):
        h, Sig = self.fcts()

        f_hr = sympy.lambdify((r, t, z), h[0], 'numpy')
        f_ht = sympy.lambdify((r, t, z), h[1], 'numpy')
        f_hz = sympy.lambdify((r, t, z), h[2], 'numpy')

        f_sig = sympy.lambdify((r, t, z), Sig[0], 'numpy')

        hr = f_hr(mesh.gridEx[:, 0], mesh.gridEx[:, 1], mesh.gridEx[:, 2])
        ht = f_ht(mesh.gridEy[:, 0], mesh.gridEy[:, 1], mesh.gridEy[:, 2])
        hz = f_hz(mesh.gridEz[:, 0], mesh.gridEz[:, 1], mesh.gridEz[:, 2])

        sig = f_sig(mesh.gridCC[:, 0], mesh.gridCC[:, 1], mesh.gridCC[:, 2])

        return sig, np.r_[hr, ht, hz]


class TestCylInnerProducts_simple(unittest.TestCase):

    def setUp(self):
        n = 100.
        self.mesh = discretize.CylMesh([n, n, n])

    def test_FaceInnerProductIsotropic(self):
        # Here we will make up some j vectors that vary in space
        # j = [j_r, j_z] - to test face inner products

        fcts = FaceInnerProductFctsIsotropic()
        sig, jv = fcts.vectors(self.mesh)
        MfSig = self.mesh.getFaceInnerProduct(sig)
        numeric_ans = jv.T.dot(MfSig.dot(jv))

        ans = fcts.sol()

        print('------ Testing Face Inner Product-----------')
        print(' Analytic: {analytic}, Numeric: {numeric}, '
              'ratio (num/ana): {ratio}'.format(
               analytic=ans, numeric=numeric_ans,
               ratio=float(numeric_ans)/ans))
        assert(np.abs(ans-numeric_ans) < TOL)

    def test_EdgeInnerProduct(self):
        # Here we will make up some j vectors that vary in space
        # h = [h_t] - to test edge inner products

        fcts = EdgeInnerProductFctsIsotropic()
        sig, hv = fcts.vectors(self.mesh)
        MeSig = self.mesh.getEdgeInnerProduct(sig)
        numeric_ans = hv.T.dot(MeSig.dot(hv))

        ans = fcts.sol()

        print('------ Testing Edge Inner Product-----------')
        print(' Analytic: {analytic}, Numeric: {numeric}, '
              'ratio (num/ana): {ratio}'.format(
               analytic=ans, numeric=numeric_ans,
               ratio=float(numeric_ans)/ans))
        assert(np.abs(ans-numeric_ans) < TOL)


class TestCylFaceInnerProducts_Order(Tests.OrderTest):

    meshTypes = ['uniformCylMesh']
    meshDimension = 3

    def getError(self):
        fct = FaceInnerProductFctsIsotropic()
        sig, jv = fct.vectors(self.M)
        Msig = self.M.getFaceInnerProduct(sig)
        return float(fct.sol()) - jv.T.dot(Msig.dot(jv))

    def test_order(self):
        self.orderTest()


class TestCylEdgeInnerProducts_Order(Tests.OrderTest):

    meshTypes = ['uniformCylMesh']
    meshDimension = 3

    def getError(self):
        fct = EdgeInnerProductFctsIsotropic()
        sig, hv = fct.vectors(self.M)
        Msig = self.M.getEdgeInnerProduct(sig)
        return float(fct.sol()) - hv.T.dot(Msig.dot(hv))

    def test_order(self):
        self.orderTest()


class MimeticProperties(unittest.TestCase):
    meshTypes = MESHTYPES
    meshDimension = 3
    meshSize = 64
    tol = 1e-11  # there is still some error due to rounding

    def test_DivCurl(self):
        for meshType in self.meshTypes:
            mesh, _ = discretize.testing.setupMesh(
                meshType, self.meshSize, self.meshDimension
            )
            v = np.random.rand(mesh.nE)
            divcurlv = mesh.faceDiv * (mesh.edgeCurl * v)
            rel_err = np.linalg.norm(divcurlv) / np.linalg.norm(v)
            passed = rel_err  < self.tol
            print(
                "Testing Div * Curl on {} : |Div Curl v| / |v| = {} "
                "... {}".format(
                    meshType, rel_err, 'FAIL' if not passed else 'ok'
                )
            )

    # # Nodal Grad has not been implemented yet
    # def test_CurlGrad(self):
    #     for meshType in self.meshTypes:
    #         mesh, _ = discretize.testing.setupMesh(
    #             meshType, self.meshSize, self.meshDimension
    #         )
    #         v = np.random.rand(mesh.nN)
    #         curlgradv = mesh.edgeCurl * (mesh.nodalGrad * v)
    #         rel_err = np.linalg.norm(curlgradv) / np.linalg.norm(v)
    #         passed = rel_err  < self.tol
    #         print(
    #             "Testing Curl * Grad on {} : |Curl Grad v| / |v|= {} "
    #             "... {}".format(
    #                 meshType, rel_err, 'FAIL' if not passed else 'ok'
    #             )
    #         )


if __name__ == '__main__':
    unittest.main()
