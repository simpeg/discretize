import numpy as np
import unittest
import discretize
from scipy.sparse.linalg import spsolve

MESHTYPES = ["uniformTensorMesh"]


def getxBCyBC_CC(mesh, alpha, beta, gamma):
    r"""
    This is a subfunction generating mixed-boundary condition:

    .. math::
        \nabla \cdot \vec{j} = -\nabla \cdot \vec{j}_s = q \\
        \rho \vec{j} = -\nabla \phi \phi \\
        \alpha \phi + \beta \frac{\partial \phi}{\partial r} = \gamma \ at \ r
        = \partial \Omega \\
        xBC = f_1(\alpha, \beta, \gamma)\\
        yBC = f(\alpha, \beta, \gamma)\\

    Computes xBC and yBC for cell-centered discretizations
    """

    if mesh.dim == 1:  # 1D
        if len(alpha) != 2 or len(beta) != 2 or len(gamma) != 2:
            raise Exception("Lenght of list, alpha should be 2")
        mesh.cell_boundary_indices

        alpha_xm, beta_xm, gamma_xm = alpha[0], beta[0], gamma[0]
        alpha_xp, beta_xp, gamma_xp = alpha[1], beta[1], gamma[1]

        # h_xm, h_xp = mesh.gridCC[fCCxm], mesh.gridCC[fCCxp]
        h_xm, h_xp = mesh.h[0][0], mesh.h[0][-1]

        a_xm = gamma_xm / (0.5 * alpha_xm - beta_xm / h_xm)
        b_xm = (0.5 * alpha_xm + beta_xm / h_xm) / (0.5 * alpha_xm - beta_xm / h_xm)
        a_xp = gamma_xp / (0.5 * alpha_xp - beta_xp / h_xp)
        b_xp = (0.5 * alpha_xp + beta_xp / h_xp) / (0.5 * alpha_xp - beta_xp / h_xp)

        xBC_xm = 0.5 * a_xm
        xBC_xp = 0.5 * a_xp / b_xp
        yBC_xm = 0.5 * (1.0 - b_xm)
        yBC_xp = 0.5 * (1.0 - 1.0 / b_xp)

        xBC = np.r_[xBC_xm, xBC_xp]
        yBC = np.r_[yBC_xm, yBC_xp]

    elif mesh.dim == 2:  # 2D
        if len(alpha) != 4 or len(beta) != 4 or len(gamma) != 4:
            raise Exception("Lenght of list, alpha should be 4")

        fxm, fxp, fym, fyp = mesh.face_boundary_indices

        alpha_xm, beta_xm, gamma_xm = alpha[0], beta[0], gamma[0]
        alpha_xp, beta_xp, gamma_xp = alpha[1], beta[1], gamma[1]
        alpha_ym, beta_ym, gamma_ym = alpha[2], beta[2], gamma[2]
        alpha_yp, beta_yp, gamma_yp = alpha[3], beta[3], gamma[3]

        # h_xm, h_xp = mesh.gridCC[fCCxm,0], mesh.gridCC[fCCxp,0]
        # h_ym, h_yp = mesh.gridCC[fCCym,1], mesh.gridCC[fCCyp,1]

        h_xm = mesh.h[0][0] * np.ones_like(alpha_xm)
        h_xp = mesh.h[0][-1] * np.ones_like(alpha_xp)
        h_ym = mesh.h[1][0] * np.ones_like(alpha_ym)
        h_yp = mesh.h[1][-1] * np.ones_like(alpha_yp)

        a_xm = gamma_xm / (0.5 * alpha_xm - beta_xm / h_xm)
        b_xm = (0.5 * alpha_xm + beta_xm / h_xm) / (0.5 * alpha_xm - beta_xm / h_xm)
        a_xp = gamma_xp / (0.5 * alpha_xp - beta_xp / h_xp)
        b_xp = (0.5 * alpha_xp + beta_xp / h_xp) / (0.5 * alpha_xp - beta_xp / h_xp)

        a_ym = gamma_ym / (0.5 * alpha_ym - beta_ym / h_ym)
        b_ym = (0.5 * alpha_ym + beta_ym / h_ym) / (0.5 * alpha_ym - beta_ym / h_ym)
        a_yp = gamma_yp / (0.5 * alpha_yp - beta_yp / h_yp)
        b_yp = (0.5 * alpha_yp + beta_yp / h_yp) / (0.5 * alpha_yp - beta_yp / h_yp)

        xBC_xm = 0.5 * a_xm
        xBC_xp = 0.5 * a_xp / b_xp
        yBC_xm = 0.5 * (1.0 - b_xm)
        yBC_xp = 0.5 * (1.0 - 1.0 / b_xp)
        xBC_ym = 0.5 * a_ym
        xBC_yp = 0.5 * a_yp / b_yp
        yBC_ym = 0.5 * (1.0 - b_ym)
        yBC_yp = 0.5 * (1.0 - 1.0 / b_yp)

        sortindsfx = np.argsort(
            np.r_[np.arange(mesh.nFx)[fxm], np.arange(mesh.nFx)[fxp]]
        )
        sortindsfy = np.argsort(
            np.r_[np.arange(mesh.nFy)[fym], np.arange(mesh.nFy)[fyp]]
        )

        xBC_x = np.r_[xBC_xm, xBC_xp][sortindsfx]
        xBC_y = np.r_[xBC_ym, xBC_yp][sortindsfy]
        yBC_x = np.r_[yBC_xm, yBC_xp][sortindsfx]
        yBC_y = np.r_[yBC_ym, yBC_yp][sortindsfy]

        xBC = np.r_[xBC_x, xBC_y]
        yBC = np.r_[yBC_x, yBC_y]

    elif mesh.dim == 3:  # 3D
        if len(alpha) != 6 or len(beta) != 6 or len(gamma) != 6:
            raise Exception("Lenght of list, alpha should be 6")
        # fCCxm,fCCxp,fCCym,fCCyp,fCCzm,fCCzp = mesh.cell_boundary_indices
        fxm, fxp, fym, fyp, fzm, fzp = mesh.face_boundary_indices

        alpha_xm, beta_xm, gamma_xm = alpha[0], beta[0], gamma[0]
        alpha_xp, beta_xp, gamma_xp = alpha[1], beta[1], gamma[1]
        alpha_ym, beta_ym, gamma_ym = alpha[2], beta[2], gamma[2]
        alpha_yp, beta_yp, gamma_yp = alpha[3], beta[3], gamma[3]
        alpha_zm, beta_zm, gamma_zm = alpha[4], beta[4], gamma[4]
        alpha_zp, beta_zp, gamma_zp = alpha[5], beta[5], gamma[5]

        # h_xm, h_xp = mesh.gridCC[fCCxm,0], mesh.gridCC[fCCxp,0]
        # h_ym, h_yp = mesh.gridCC[fCCym,1], mesh.gridCC[fCCyp,1]
        # h_zm, h_zp = mesh.gridCC[fCCzm,2], mesh.gridCC[fCCzp,2]

        h_xm = mesh.h[0][0] * np.ones_like(alpha_xm)
        h_xp = mesh.h[0][-1] * np.ones_like(alpha_xp)
        h_ym = mesh.h[1][0] * np.ones_like(alpha_ym)
        h_yp = mesh.h[1][-1] * np.ones_like(alpha_yp)
        h_zm = mesh.h[2][0] * np.ones_like(alpha_zm)
        h_zp = mesh.h[2][-1] * np.ones_like(alpha_zp)

        a_xm = gamma_xm / (0.5 * alpha_xm - beta_xm / h_xm)
        b_xm = (0.5 * alpha_xm + beta_xm / h_xm) / (0.5 * alpha_xm - beta_xm / h_xm)
        a_xp = gamma_xp / (0.5 * alpha_xp - beta_xp / h_xp)
        b_xp = (0.5 * alpha_xp + beta_xp / h_xp) / (0.5 * alpha_xp - beta_xp / h_xp)

        a_ym = gamma_ym / (0.5 * alpha_ym - beta_ym / h_ym)
        b_ym = (0.5 * alpha_ym + beta_ym / h_ym) / (0.5 * alpha_ym - beta_ym / h_ym)
        a_yp = gamma_yp / (0.5 * alpha_yp - beta_yp / h_yp)
        b_yp = (0.5 * alpha_yp + beta_yp / h_yp) / (0.5 * alpha_yp - beta_yp / h_yp)

        a_zm = gamma_zm / (0.5 * alpha_zm - beta_zm / h_zm)
        b_zm = (0.5 * alpha_zm + beta_zm / h_zm) / (0.5 * alpha_zm - beta_zm / h_zm)
        a_zp = gamma_zp / (0.5 * alpha_zp - beta_zp / h_zp)
        b_zp = (0.5 * alpha_zp + beta_zp / h_zp) / (0.5 * alpha_zp - beta_zp / h_zp)

        xBC_xm = 0.5 * a_xm
        xBC_xp = 0.5 * a_xp / b_xp
        yBC_xm = 0.5 * (1.0 - b_xm)
        yBC_xp = 0.5 * (1.0 - 1.0 / b_xp)
        xBC_ym = 0.5 * a_ym
        xBC_yp = 0.5 * a_yp / b_yp
        yBC_ym = 0.5 * (1.0 - b_ym)
        yBC_yp = 0.5 * (1.0 - 1.0 / b_yp)
        xBC_zm = 0.5 * a_zm
        xBC_zp = 0.5 * a_zp / b_zp
        yBC_zm = 0.5 * (1.0 - b_zm)
        yBC_zp = 0.5 * (1.0 - 1.0 / b_zp)

        sortindsfx = np.argsort(
            np.r_[np.arange(mesh.nFx)[fxm], np.arange(mesh.nFx)[fxp]]
        )
        sortindsfy = np.argsort(
            np.r_[np.arange(mesh.nFy)[fym], np.arange(mesh.nFy)[fyp]]
        )
        sortindsfz = np.argsort(
            np.r_[np.arange(mesh.nFz)[fzm], np.arange(mesh.nFz)[fzp]]
        )

        xBC_x = np.r_[xBC_xm, xBC_xp][sortindsfx]
        xBC_y = np.r_[xBC_ym, xBC_yp][sortindsfy]
        xBC_z = np.r_[xBC_zm, xBC_zp][sortindsfz]

        yBC_x = np.r_[yBC_xm, yBC_xp][sortindsfx]
        yBC_y = np.r_[yBC_ym, yBC_yp][sortindsfy]
        yBC_z = np.r_[yBC_zm, yBC_zp][sortindsfz]

        xBC = np.r_[xBC_x, xBC_y, xBC_z]
        yBC = np.r_[yBC_x, yBC_y, yBC_z]

    return xBC, yBC


class Test1D_InhomogeneousMixed(discretize.tests.OrderTest):
    name = "1D - Mixed"
    meshTypes = MESHTYPES
    meshDimension = 1
    expectedOrders = 2
    meshSizes = [4, 8, 16, 32]

    def getError(self):
        # Test function
        def phi_fun(x):
            return np.cos(np.pi * x)

        def j_fun(x):
            return np.pi * np.sin(np.pi * x)

        def phi_deriv(x):
            return -j_fun(x)

        def q_fun(x):
            return (np.pi**2) * np.cos(np.pi * x)

        xc_ana = phi_fun(self.M.gridCC)

        # Get boundary locations
        vecN = self.M.nodes_x

        # Setup Mixed B.C (alpha, beta, gamma)
        alpha_xm, alpha_xp = 1.0, 1.0
        beta_xm, beta_xp = 1.0, 1.0
        alpha = np.r_[alpha_xm, alpha_xp]
        beta = np.r_[beta_xm, beta_xp]
        phi_bc = phi_fun(vecN[[0, -1]])
        phi_deriv_bc = phi_deriv(vecN[[0, -1]])
        gamma = alpha * phi_bc + beta * phi_deriv_bc
        x_BC, y_BC = getxBCyBC_CC(self.M, alpha, beta, gamma)

        sigma = np.ones(self.M.nC)
        self.M.get_face_inner_product(1.0 / sigma)
        MfrhoI = self.M.get_face_inner_product(1.0 / sigma, invert_matrix=True)
        V = discretize.utils.sdiag(self.M.cell_volumes)
        Div = V * self.M.face_divergence
        P_BC, B = self.M.get_BC_projections_simple()
        q = q_fun(self.M.gridCC)
        M = B * self.M.aveCC2F
        G = Div.T - P_BC * discretize.utils.sdiag(y_BC) * M
        # Mrhoj = D.T V phi + P_BC*discretize.utils.sdiag(y_BC)*M phi - P_BC*x_BC
        rhs = V * q + Div * MfrhoI * P_BC * x_BC
        A = Div * MfrhoI * G

        if self.myTest == "xc":
            # TODO: fix the null space
            xc = spsolve(A, rhs)
            err = np.linalg.norm((xc - xc_ana), np.inf)
        else:
            NotImplementedError
        return err

    def test_order(self):
        print("==== Testing Mixed boudary conduction for CC-problem ====")
        self.name = "1D"
        self.myTest = "xc"
        self.orderTest()


class Test2D_InhomogeneousMixed(discretize.tests.OrderTest):
    name = "2D - Mixed"
    meshTypes = MESHTYPES
    meshDimension = 2
    expectedOrders = 2
    meshSizes = [4, 8, 16, 32]

    def getError(self):
        # Test function
        def phi_fun(x):
            return np.cos(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])

        def j_funX(x):
            return +np.pi * np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])

        def j_funY(x):
            return +np.pi * np.cos(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])

        def phideriv_funX(x):
            return -j_funX(x)

        def phideriv_funY(x):
            return -j_funY(x)

        def q_fun(x):
            return +2 * (np.pi**2) * phi_fun(x)

        xc_ana = phi_fun(self.M.gridCC)

        # Get boundary locations
        fxm, fxp, fym, fyp = self.M.face_boundary_indices
        gBFxm = self.M.gridFx[fxm, :]
        gBFxp = self.M.gridFx[fxp, :]
        gBFym = self.M.gridFy[fym, :]
        gBFyp = self.M.gridFy[fyp, :]

        # Setup Mixed B.C (alpha, beta, gamma)
        alpha_xm = np.ones_like(gBFxm[:, 0])
        alpha_xp = np.ones_like(gBFxp[:, 0])
        beta_xm = np.ones_like(gBFxm[:, 0])
        beta_xp = np.ones_like(gBFxp[:, 0])
        alpha_ym = np.ones_like(gBFym[:, 1])
        alpha_yp = np.ones_like(gBFyp[:, 1])
        beta_ym = np.ones_like(gBFym[:, 1])
        beta_yp = np.ones_like(gBFyp[:, 1])

        phi_bc_xm, phi_bc_xp = phi_fun(gBFxm), phi_fun(gBFxp)
        phi_bc_ym, phi_bc_yp = phi_fun(gBFym), phi_fun(gBFyp)

        phiderivX_bc_xm = phideriv_funX(gBFxm)
        phiderivX_bc_xp = phideriv_funX(gBFxp)
        phiderivY_bc_ym = phideriv_funY(gBFym)
        phiderivY_bc_yp = phideriv_funY(gBFyp)

        def gamma_fun(alpha, beta, phi, phi_deriv):
            return alpha * phi + beta * phi_deriv

        gamma_xm = gamma_fun(alpha_xm, beta_xm, phi_bc_xm, phiderivX_bc_xm)
        gamma_xp = gamma_fun(alpha_xp, beta_xp, phi_bc_xp, phiderivX_bc_xp)
        gamma_ym = gamma_fun(alpha_ym, beta_ym, phi_bc_ym, phiderivY_bc_ym)
        gamma_yp = gamma_fun(alpha_yp, beta_yp, phi_bc_yp, phiderivY_bc_yp)

        alpha = [alpha_xm, alpha_xp, alpha_ym, alpha_yp]
        beta = [beta_xm, beta_xp, beta_ym, beta_yp]
        gamma = [gamma_xm, gamma_xp, gamma_ym, gamma_yp]

        x_BC, y_BC = getxBCyBC_CC(self.M, alpha, beta, gamma)

        sigma = np.ones(self.M.nC)
        self.M.get_face_inner_product(1.0 / sigma)
        MfrhoI = self.M.get_face_inner_product(1.0 / sigma, invert_matrix=True)
        V = discretize.utils.sdiag(self.M.cell_volumes)
        Div = V * self.M.face_divergence
        P_BC, B = self.M.get_BC_projections_simple()
        q = q_fun(self.M.gridCC)
        M = B * self.M.aveCC2F
        G = Div.T - P_BC * discretize.utils.sdiag(y_BC) * M
        rhs = V * q + Div * MfrhoI * P_BC * x_BC
        A = Div * MfrhoI * G

        if self.myTest == "xc":
            xc = spsolve(A, rhs)
            err = np.linalg.norm((xc - xc_ana), np.inf)
        else:
            NotImplementedError
        return err

    def test_order(self):
        print("==== Testing Mixed boudary conduction for CC-problem ====")
        self.name = "2D"
        self.myTest = "xc"
        self.orderTest()


class Test3D_InhomogeneousMixed(discretize.tests.OrderTest):
    name = "3D - Mixed"
    meshTypes = MESHTYPES
    meshDimension = 3
    expectedOrders = 2
    meshSizes = [4, 8, 16]

    def getError(self):
        # Test function
        def phi_fun(x):
            return (
                np.cos(np.pi * x[:, 0])
                * np.cos(np.pi * x[:, 1])
                * np.cos(np.pi * x[:, 2])
            )

        def j_funX(x):
            return (
                np.pi
                * np.sin(np.pi * x[:, 0])
                * np.cos(np.pi * x[:, 1])
                * np.cos(np.pi * x[:, 2])
            )

        def j_funY(x):
            return (
                np.pi
                * np.cos(np.pi * x[:, 0])
                * np.sin(np.pi * x[:, 1])
                * np.cos(np.pi * x[:, 2])
            )

        def j_funZ(x):
            return (
                np.pi
                * np.cos(np.pi * x[:, 0])
                * np.cos(np.pi * x[:, 1])
                * np.sin(np.pi * x[:, 2])
            )

        def phideriv_funX(x):
            return -j_funX(x)

        def phideriv_funY(x):
            return -j_funY(x)

        def phideriv_funZ(x):
            return -j_funZ(x)

        def q_fun(x):
            return 3 * (np.pi**2) * phi_fun(x)

        xc_ana = phi_fun(self.M.gridCC)

        # Get boundary locations
        fxm, fxp, fym, fyp, fzm, fzp = self.M.face_boundary_indices
        gBFxm = self.M.gridFx[fxm, :]
        gBFxp = self.M.gridFx[fxp, :]
        gBFym = self.M.gridFy[fym, :]
        gBFyp = self.M.gridFy[fyp, :]
        gBFzm = self.M.gridFz[fzm, :]
        gBFzp = self.M.gridFz[fzp, :]

        # Setup Mixed B.C (alpha, beta, gamma)
        alpha_xm = np.ones_like(gBFxm[:, 0])
        alpha_xp = np.ones_like(gBFxp[:, 0])
        beta_xm = np.ones_like(gBFxm[:, 0])
        beta_xp = np.ones_like(gBFxp[:, 0])
        alpha_ym = np.ones_like(gBFym[:, 1])
        alpha_yp = np.ones_like(gBFyp[:, 1])
        beta_ym = np.ones_like(gBFym[:, 1])
        beta_yp = np.ones_like(gBFyp[:, 1])
        alpha_zm = np.ones_like(gBFzm[:, 2])
        alpha_zp = np.ones_like(gBFzp[:, 2])
        beta_zm = np.ones_like(gBFzm[:, 2])
        beta_zp = np.ones_like(gBFzp[:, 2])

        phi_bc_xm, phi_bc_xp = phi_fun(gBFxm), phi_fun(gBFxp)
        phi_bc_ym, phi_bc_yp = phi_fun(gBFym), phi_fun(gBFyp)
        phi_bc_zm, phi_bc_zp = phi_fun(gBFzm), phi_fun(gBFzp)

        phiderivX_bc_xm = phideriv_funX(gBFxm)
        phiderivX_bc_xp = phideriv_funX(gBFxp)
        phiderivY_bc_ym = phideriv_funY(gBFym)
        phiderivY_bc_yp = phideriv_funY(gBFyp)
        phiderivY_bc_zm = phideriv_funZ(gBFzm)
        phiderivY_bc_zp = phideriv_funZ(gBFzp)

        def gamma_fun(alpha, beta, phi, phi_deriv):
            return alpha * phi + beta * phi_deriv

        gamma_xm = gamma_fun(alpha_xm, beta_xm, phi_bc_xm, phiderivX_bc_xm)
        gamma_xp = gamma_fun(alpha_xp, beta_xp, phi_bc_xp, phiderivX_bc_xp)
        gamma_ym = gamma_fun(alpha_ym, beta_ym, phi_bc_ym, phiderivY_bc_ym)
        gamma_yp = gamma_fun(alpha_yp, beta_yp, phi_bc_yp, phiderivY_bc_yp)
        gamma_zm = gamma_fun(alpha_zm, beta_zm, phi_bc_zm, phiderivY_bc_zm)
        gamma_zp = gamma_fun(alpha_zp, beta_zp, phi_bc_zp, phiderivY_bc_zp)

        alpha = [alpha_xm, alpha_xp, alpha_ym, alpha_yp, alpha_zm, alpha_zp]
        beta = [beta_xm, beta_xp, beta_ym, beta_yp, beta_zm, beta_zp]
        gamma = [gamma_xm, gamma_xp, gamma_ym, gamma_yp, gamma_zm, gamma_zp]

        x_BC, y_BC = getxBCyBC_CC(self.M, alpha, beta, gamma)

        sigma = np.ones(self.M.nC)
        self.M.get_face_inner_product(1.0 / sigma)
        MfrhoI = self.M.get_face_inner_product(1.0 / sigma, invert_matrix=True)
        V = discretize.utils.sdiag(self.M.cell_volumes)
        Div = V * self.M.face_divergence
        P_BC, B = self.M.get_BC_projections_simple()
        q = q_fun(self.M.gridCC)
        M = B * self.M.aveCC2F
        G = Div.T - P_BC * discretize.utils.sdiag(y_BC) * M
        rhs = V * q + Div * MfrhoI * P_BC * x_BC
        A = Div * MfrhoI * G

        if self.myTest == "xc":
            # TODO: fix the null space
            xc = spsolve(A, rhs)
            err = np.linalg.norm((xc - xc_ana), np.inf)
        else:
            NotImplementedError
        return err

    def test_order(self):
        print("==== Testing Mixed boudary conduction for CC-problem ====")
        self.name = "3D"
        self.myTest = "xc"
        self.orderTest()


if __name__ == "__main__":
    unittest.main()
