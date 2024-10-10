import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg
import unittest
import discretize
from discretize import utils
from scipy.sparse.linalg import spsolve


class TestCC1D_InhomogeneousDirichlet(discretize.tests.OrderTest):
    name = "1D - Dirichlet"
    meshTypes = ["uniformTensorMesh"]
    meshDimension = 1
    expectedOrders = 2
    meshSizes = [4, 8, 16, 32, 64, 128]

    def getError(self):
        # Test function
        phi = lambda x: np.cos(np.pi * x)
        q_fun = lambda x: -(np.pi**2) * np.cos(np.pi * x)

        mesh = self.M

        phi_ana = phi(mesh.cell_centers)
        q_ana = q_fun(mesh.cell_centers)

        boundary_faces = mesh.boundary_faces

        phi_bc = phi(boundary_faces)

        MfI = mesh.get_face_inner_product(invert_matrix=True)
        M_bf = mesh.boundary_face_scalar_integral

        V = utils.sdiag(mesh.cell_volumes)
        G = -mesh.face_divergence.T * V
        D = mesh.face_divergence

        # Sinc the xc_bc is a known, move it to the RHS!
        A = V @ D @ MfI @ G
        rhs = V @ q_ana - V @ D @ MfI @ M_bf @ phi_bc

        phi_test = spsolve(A, rhs)
        err = np.linalg.norm((phi_test - phi_ana)) / np.sqrt(mesh.n_cells)

        return err

    def test_orderX(self):
        self.name = "1D - InhomogeneousDirichlet_Inverse"
        self.myTest = "xc"
        self.orderTest()


class TestCC2D_InhomogeneousDirichlet(discretize.tests.OrderTest):
    name = "2D - Dirichlet"
    meshTypes = ["uniformTensorMesh", "uniformTree", "rotateCurv"]
    meshDimension = 2
    expectedOrders = [2, 2, 1]
    meshSizes = [4, 8, 16, 32, 64]

    def getError(self):
        # Test function
        phi = lambda x: np.cos(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
        q_fun = lambda x: -2 * (np.pi**2) * phi(x)

        mesh = self.M
        phi_ana = phi(mesh.cell_centers)
        q_ana = q_fun(mesh.cell_centers)

        phi_bc = phi(mesh.boundary_faces)

        MfI = mesh.get_face_inner_product(invert_matrix=True)
        M_bf = mesh.boundary_face_scalar_integral

        V = utils.sdiag(mesh.cell_volumes)
        G = -mesh.face_divergence.T * V
        D = mesh.face_divergence

        # Sinc the xc_bc is a known, move it to the RHS!
        A = V @ D @ MfI @ G
        rhs = V @ q_ana - V @ D @ MfI @ M_bf @ phi_bc

        phi_test = spsolve(A, rhs)
        if self._meshType == "rotateCurv":
            err = np.linalg.norm(mesh.cell_volumes * (phi_test - phi_ana))
        else:
            err = np.linalg.norm(phi_test - phi_ana) / np.sqrt(mesh.n_cells)

        return err

    def test_orderX(self):
        self.name = "2D - InhomogeneousDirichlet_Inverse"
        self.myTest = "xc"
        self.orderTest()


class TestCC1D_InhomogeneousNeumann(discretize.tests.OrderTest):
    name = "1D - Neumann"
    meshTypes = ["uniformTensorMesh"]
    meshDimension = 1
    expectedOrders = 2
    meshSizes = [4, 8, 16, 32, 64, 128]

    def getError(self):
        # Test function
        phi = lambda x: np.sin(np.pi * x)
        j_fun = lambda x: np.pi * np.cos(np.pi * x)
        q_fun = lambda x: -(np.pi**2) * np.sin(np.pi * x)

        mesh = self.M
        xc_ana = phi(mesh.cell_centers)
        q_ana = q_fun(mesh.cell_centers)
        j_ana = j_fun(mesh.faces_x)

        phi_bc = phi(mesh.boundary_faces)
        j_bc = j_fun(mesh.boundary_faces)

        MfI = mesh.get_face_inner_product(invert_matrix=True)

        V = utils.sdiag(mesh.cell_volumes)
        G = mesh.face_divergence.T * V
        D = mesh.face_divergence

        # construct matrix with robin operator
        alpha = 0.0
        beta = 1.0
        gamma = alpha * phi_bc + beta * j_bc * mesh.boundary_face_outward_normals
        B_bc, b_bc = mesh.cell_gradient_weak_form_robin(
            alpha=alpha, beta=beta, gamma=gamma
        )

        j = MfI @ ((-G + B_bc) @ xc_ana + b_bc)

        # Since the xc_bc is a known, move it to the RHS!
        A = V @ D @ MfI @ (-G + B_bc)
        rhs = V @ q_ana - V @ D @ MfI @ b_bc

        xc_disc, info = sp.linalg.minres(A, rhs, rtol=1e-6)

        if self.myTest == "j":
            err = np.linalg.norm((j - j_ana), np.inf)
        elif self.myTest == "xcJ":
            # TODO: fix the null space
            xc, info = linalg.minres(A, rhs, rtol=1e-6)
            j = MfI @ ((-G + B_bc) @ xc + b_bc)
            err = np.linalg.norm((j - j_ana)) / np.sqrt(mesh.n_edges)
            if info > 0:
                print("Solve does not work well")
                print("ACCURACY", np.linalg.norm(utils.mkvc(A * xc) - rhs))
        return err

    def test_orderJ(self):
        self.name = "1D - InhomogeneousNeumann_Forward j"
        self.myTest = "j"
        self.orderTest()

    def test_orderXJ(self):
        self.name = "1D - InhomogeneousNeumann_Inverse J"
        self.myTest = "xcJ"
        self.orderTest()


class TestCC2D_InhomogeneousNeumann(discretize.tests.OrderTest):
    name = "2D - Neumann"
    meshTypes = ["uniformTensorMesh", "uniformTree", "rotateCurv"]
    meshDimension = 2
    expectedOrders = [2, 2, 1]
    meshSizes = [4, 8, 16, 32]
    # meshSizes = [4]

    def getError(self):
        # Test function
        phi = lambda x: np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
        j_funX = lambda x: np.pi * np.cos(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
        j_funY = lambda x: np.pi * np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
        q_fun = lambda x: -2 * (np.pi**2) * phi(x)

        mesh = self.M
        phi_ana = phi(mesh.cell_centers)
        q_ana = q_fun(mesh.cell_centers)

        phi_bc = phi(mesh.boundary_faces)
        jx_bc = j_funX(mesh.boundary_faces)
        jy_bc = j_funY(mesh.boundary_faces)
        j_bc = np.c_[jx_bc, jy_bc]

        j_bc_dot_n = np.sum(j_bc * mesh.boundary_face_outward_normals, axis=-1)

        MfI = mesh.get_face_inner_product(invert_matrix=True)

        V = utils.sdiag(mesh.cell_volumes)
        G = mesh.face_divergence.T * V
        D = mesh.face_divergence

        # construct matrix with robin operator
        alpha = 0.0
        beta = 1.0
        gamma = alpha * phi_bc + beta * j_bc_dot_n

        B_bc, b_bc = mesh.cell_gradient_weak_form_robin(
            alpha=alpha, beta=beta, gamma=gamma
        )

        A = V @ D @ MfI @ (-G + B_bc)
        rhs = V @ q_ana - V @ D @ MfI @ b_bc

        phi_test, info = linalg.minres(A, rhs, rtol=1e-6)
        phi_test -= phi_test.mean()
        phi_ana -= phi_ana.mean()

        # err = np.linalg.norm(phi_test - phi_ana)/np.sqrt(mesh.n_cells)

        if self._meshType == "rotateCurv":
            err = np.linalg.norm(mesh.cell_volumes * (phi_test - phi_ana))
        else:
            err = np.linalg.norm((phi_test - phi_ana), np.inf)

        return err

    def test_orderX(self):
        self.name = "2D - InhomogeneousNeumann_Inverse"
        self.myTest = "xc"
        self.orderTest()


class TestCC1D_InhomogeneousMixed(discretize.tests.OrderTest):
    name = "1D - Mixed"
    meshTypes = ["uniformTensorMesh"]
    meshDimension = 1
    expectedOrders = 2
    meshSizes = [4, 8, 16, 32, 64, 128]

    def getError(self):
        # Test function
        phi = lambda x: np.cos(0.5 * np.pi * x)
        j_fun = lambda x: -0.5 * np.pi * np.sin(0.5 * np.pi * x)
        q_fun = lambda x: -0.25 * (np.pi**2) * np.cos(0.5 * np.pi * x)

        mesh = self.M
        xc_ana = phi(mesh.cell_centers)
        q_ana = q_fun(mesh.cell_centers)
        j_ana = j_fun(mesh.faces_x)

        phi_bc = phi(mesh.boundary_faces)
        j_bc = j_fun(mesh.boundary_faces)

        MfI = mesh.get_face_inner_product(invert_matrix=True)

        V = utils.sdiag(mesh.cell_volumes)
        G = mesh.face_divergence.T * V
        D = mesh.face_divergence

        # construct matrix with robin operator
        alpha = np.r_[1.0, 0.0]
        beta = np.r_[0.0, 1.0]
        gamma = alpha * phi_bc + beta * j_bc * mesh.boundary_face_outward_normals
        B_bc, b_bc = mesh.cell_gradient_weak_form_robin(
            alpha=alpha, beta=beta, gamma=gamma
        )

        A = V @ D @ MfI @ (-G + B_bc)
        rhs = V @ q_ana - V @ D @ MfI @ b_bc

        if self.myTest == "xc":
            xc = spsolve(A, rhs)
            err = np.linalg.norm(xc - xc_ana) / np.sqrt(mesh.n_cells)
        elif self.myTest == "xcJ":
            xc = spsolve(A, rhs)
            j = MfI @ ((-G + B_bc) @ xc + b_bc)
            err = np.linalg.norm(j - j_ana, np.inf)
        return err

    def test_orderX(self):
        self.name = "1D - InhomogeneousMixed_Inverse"
        self.myTest = "xc"
        self.orderTest()

    def test_orderXJ(self):
        self.name = "1D - InhomogeneousMixed_Inverse J"
        self.myTest = "xcJ"
        self.orderTest()


class TestCC2D_InhomogeneousMixed(discretize.tests.OrderTest):
    name = "2D - Mixed"
    meshTypes = ["uniformTensorMesh", "uniformTree", "rotateCurv"]
    meshDimension = 2
    expectedOrders = [2, 2, 1]
    meshSizes = [2, 4, 8, 16]
    # meshSizes = [4]

    def getError(self):
        # Test function
        phi = lambda x: np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
        j_funX = lambda x: np.pi * np.cos(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
        j_funY = lambda x: np.pi * np.sin(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
        q_fun = lambda x: -2 * (np.pi**2) * phi(x)

        mesh = self.M
        phi_ana = phi(mesh.cell_centers)
        q_ana = q_fun(mesh.cell_centers)

        phi_bc = phi(mesh.boundary_faces)
        jx_bc = j_funX(mesh.boundary_faces)
        jy_bc = j_funY(mesh.boundary_faces)
        j_bc = np.c_[jx_bc, jy_bc]

        j_bc_dot_n = np.sum(j_bc * mesh.boundary_face_outward_normals, axis=-1)

        MfI = mesh.get_face_inner_product(invert_matrix=True)

        V = utils.sdiag(mesh.cell_volumes)
        G = mesh.face_divergence.T * V
        D = mesh.face_divergence

        # construct matrix with robin operator
        # get indices of x0 boundary and y0 boundary
        n_bounary_faces = len(j_bc_dot_n)
        dirichlet_locs = np.any(mesh.boundary_faces == 0.0, axis=1)

        alpha = np.zeros(n_bounary_faces)
        alpha[dirichlet_locs] = 1.0

        beta = np.zeros(n_bounary_faces)
        beta[~dirichlet_locs] = 1.0

        gamma = alpha * phi_bc + beta * j_bc_dot_n

        B_bc, b_bc = mesh.cell_gradient_weak_form_robin(
            alpha=alpha, beta=beta, gamma=gamma
        )

        A = V @ D @ MfI @ (-G + B_bc)
        rhs = V @ q_ana - V @ D @ MfI @ b_bc

        phi_test = spsolve(A, rhs)

        if self._meshType == "rotateCurv":
            err = np.linalg.norm(mesh.cell_volumes * (phi_test - phi_ana))
        else:
            err = np.linalg.norm((phi_test - phi_ana)) / np.sqrt(mesh.n_cells)

        return err

    def test_orderX(self):
        self.name = "2D - InhomogeneousMixed_Inverse"
        self.myTest = "xc"
        self.orderTest()


class TestCC3D_InhomogeneousMixed(discretize.tests.OrderTest):
    name = "3D - Mixed"
    meshTypes = ["uniformTensorMesh", "uniformTree", "rotateCurv"]
    meshDimension = 3
    expectedOrders = [2, 2, 2]
    meshSizes = [2, 4, 8, 16]

    def getError(self):
        # Test function
        phi = (
            lambda x: np.sin(np.pi * x[:, 0])
            * np.sin(np.pi * x[:, 1])
            * np.sin(np.pi * x[:, 2])
        )

        j_funX = (
            lambda x: np.pi
            * np.cos(np.pi * x[:, 0])
            * np.sin(np.pi * x[:, 1])
            * np.sin(np.pi * x[:, 2])
        )
        j_funY = (
            lambda x: np.pi
            * np.sin(np.pi * x[:, 0])
            * np.cos(np.pi * x[:, 1])
            * np.sin(np.pi * x[:, 2])
        )
        j_funZ = (
            lambda x: np.pi
            * np.sin(np.pi * x[:, 0])
            * np.sin(np.pi * x[:, 1])
            * np.cos(np.pi * x[:, 2])
        )

        q_fun = lambda x: -3 * (np.pi**2) * phi(x)

        mesh = self.M
        phi_ana = phi(mesh.cell_centers)
        q_ana = q_fun(mesh.cell_centers)

        phi_bc = phi(mesh.boundary_faces)
        jx_bc = j_funX(mesh.boundary_faces)
        jy_bc = j_funY(mesh.boundary_faces)
        jz_bc = j_funZ(mesh.boundary_faces)
        j_bc = np.c_[jx_bc, jy_bc, jz_bc]

        j_bc_dot_n = np.sum(j_bc * mesh.boundary_face_outward_normals, axis=-1)

        MfI = mesh.get_face_inner_product(invert_matrix=True)

        V = utils.sdiag(mesh.cell_volumes)
        G = mesh.face_divergence.T * V
        D = mesh.face_divergence

        # construct matrix with robin operator
        # get indices of x0 boundary, y0, and z0 boundary
        n_bounary_faces = len(j_bc_dot_n)
        dirichlet_locs = np.any(mesh.boundary_faces == 0.0, axis=1)

        alpha = np.zeros(n_bounary_faces)
        alpha[dirichlet_locs] = 1.0

        beta = np.zeros(n_bounary_faces)
        beta[~dirichlet_locs] = 1.0

        gamma = alpha * phi_bc + beta * j_bc_dot_n

        B_bc, b_bc = mesh.cell_gradient_weak_form_robin(
            alpha=alpha, beta=beta, gamma=gamma
        )

        A = V @ D @ MfI @ (-G + B_bc)
        rhs = V @ q_ana - V @ D @ MfI @ b_bc

        phi_test = spsolve(A, rhs)

        if self._meshType == "rotateCurv":
            err = np.linalg.norm(mesh.cell_volumes * (phi_test - phi_ana))
        else:
            err = np.linalg.norm(phi_test - phi_ana) / np.sqrt(mesh.n_cells)
        return err

    def test_orderX(self):
        self.name = "3D - InhomogeneousMixed_Inverse"
        self.myTest = "xc"
        self.orderTest()


class TestN1D_boundaries(discretize.tests.OrderTest):
    name = "1D - Boundaries"
    meshTypes = ["uniformTensorMesh"]
    meshDimension = 1
    expectedOrders = 2
    meshSizes = [2, 4, 8, 16, 32, 64, 128]
    # meshSizes = [4]

    def getError(self):
        # Test function
        phi = lambda x: np.sin(np.pi * x)
        j_fun = lambda x: np.pi * np.cos(np.pi * x)
        q_fun = lambda x: -1 * (np.pi**2) * phi(x)

        mesh = self.M
        mesh.origin = [
            -0.25,
        ]

        phi_ana = phi(mesh.nodes)
        q_ana = q_fun(mesh.nodes)

        phi_bc = phi(mesh.boundary_nodes)
        j_bc = j_fun(mesh.boundary_nodes)

        # construct matrix with robin operator
        beta = 1.0
        if self.boundary_type == "Robin":
            alpha = 1.0
        elif self.boundary_type == "Mixed":
            alpha = np.r_[1.0, 0.0]
        else:
            alpha = 0.0

        gamma = alpha * phi_bc + beta * j_bc * mesh.boundary_face_outward_normals

        Me = mesh.get_edge_inner_product()
        Mn = sp.diags(mesh.average_node_to_cell.T @ mesh.cell_volumes)
        G = mesh.nodal_gradient

        B_bc, b_bc = mesh.edge_divergence_weak_form_robin(alpha, beta, gamma)

        A = -G.T @ Me @ G + B_bc
        rhs = Mn @ q_ana - b_bc

        if self.boundary_type == "Nuemann":
            # put a single dirichlet node on the boundary
            P_b = sp.eye(mesh.n_nodes, format="csc")[:, 0]
            P_f = sp.eye(mesh.n_nodes, format="csc")[:, 1:]
            # P_f.T @ A @ (P_f @ x + P_b @ ana) = P_f.T @ rhs
            rhs = P_f.T @ rhs - (P_f.T @ A @ P_b) @ phi_ana[[0]]
            A = P_f.T @ A @ P_f

        phi_test = spsolve(A, rhs)

        if self.boundary_type == "Nuemann":
            phi_test = P_f @ phi_test + P_b @ phi_ana[[0]]

        err = np.linalg.norm(phi_test - phi_ana, np.inf)

        return err

    def test_orderNuemannX(self):
        self.name = "1D - NuemannBoundary_Inverse"
        self.boundary_type = "Nuemann"
        self.orderTest()

    def test_orderRobinX(self):
        self.name = "1D - RobinBoundary_Inverse"
        self.boundary_type = "Robin"
        self.orderTest()

    def test_orderMixed(self):
        self.name = "1D - MixedBoundary_Inverse"
        self.boundary_type = "Mixed"
        self.orderTest()


class TestN2D_boundaries(discretize.tests.OrderTest):
    name = "2D - Boundaries"
    meshTypes = ["uniformTensorMesh", "uniformTree", "rotateCurv"]
    meshDimension = 2
    expectedOrders = 2
    tolerance = [0.8, 0.8, 0.6]
    meshSizes = [8, 16, 32, 64]
    # meshSizes = [4]

    def getError(self):
        # Test function
        phi = lambda x: np.sin(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
        j_funX = lambda x: np.pi * np.cos(np.pi * x[:, 0]) * np.sin(np.pi * x[:, 1])
        j_funY = lambda x: np.pi * np.cos(np.pi * x[:, 1]) * np.sin(np.pi * x[:, 0])
        q_fun = lambda x: -2 * (np.pi**2) * phi(x)

        mesh = self.M
        if self._meshType == "rotateCurv":
            nodes_x, nodes_y = mesh.node_list
            nodes_x -= 0.25
            nodes_y -= 0.25
            mesh = discretize.CurvilinearMesh([nodes_x, nodes_y])
        else:
            mesh.origin = np.r_[-0.25, -0.25]

        phi_ana = phi(mesh.nodes)
        q_ana = q_fun(mesh.nodes)

        if self.boundary_type == "Nuemann":
            # Nuemann with J defined at boundary nodes
            jx_bc = j_funX(mesh.boundary_nodes)
            jy_bc = j_funY(mesh.boundary_nodes)
            j_bc = np.c_[jx_bc, jy_bc]

            M_bn = mesh.boundary_node_vector_integral

            B_bc = sp.csr_matrix((mesh.n_nodes, mesh.n_nodes))
            b_bc = M_bn @ (j_bc.reshape(-1, order="F"))
        else:
            phi_bc = phi(mesh.boundary_faces)
            jx_bc = j_funX(mesh.boundary_faces)
            jy_bc = j_funY(mesh.boundary_faces)
            j_bc = np.c_[jx_bc, jy_bc]

            j_bc_dot_n = np.sum(j_bc * mesh.boundary_face_outward_normals, axis=-1)

            # construct matrix with robin operator
            if self.boundary_type == "Robin":
                alpha = 1.0
            else:
                # get indices of x0 boundary and y0 boundary
                n_boundary_faces = len(j_bc_dot_n)
                robin_locs = np.any(mesh.boundary_faces == -0.25, axis=1)

                alpha = np.zeros(n_boundary_faces)
                alpha[robin_locs] = 1.0

            beta = 1.0
            gamma = alpha * phi_bc + beta * j_bc_dot_n

            B_bc, b_bc = mesh.edge_divergence_weak_form_robin(alpha, beta, gamma)

        Me = mesh.get_edge_inner_product()
        Mn = sp.diags(mesh.average_node_to_cell.T @ mesh.cell_volumes)
        G = mesh.nodal_gradient

        A = -G.T @ Me @ G + B_bc
        rhs = Mn @ q_ana - b_bc

        if self.boundary_type == "Nuemann":
            # put a single dirichlet node on the boundary
            P_b = sp.eye(mesh.n_nodes, format="csc")[:, 0]
            P_f = sp.eye(mesh.n_nodes, format="csc")[:, 1:]
            # P_f.T @ A @ (P_f @ x + P_b @ ana) = P_f.T @ rhs
            rhs = P_f.T @ rhs - (P_f.T @ A @ P_b) @ phi_ana[[0]]
            A = P_f.T @ A @ P_f

        phi_test = spsolve(A, rhs)

        if self.boundary_type == "Nuemann":
            phi_test = P_f @ phi_test + P_b @ phi_ana[[0]]

        err = np.linalg.norm(phi_test - phi_ana, np.inf)
        return err

    def test_orderNuemannX(self):
        self.name = "2D - NuemannBoundary_Inverse"
        self.boundary_type = "Nuemann"
        self.orderTest()

    def test_orderRobinX(self):
        self.name = "2D - RobinBoundary_Inverse"
        self.boundary_type = "Robin"
        self.orderTest()

    def test_orderMixed(self):
        self.name = "2D - MixedBoundary_Inverse"
        self.boundary_type = "Mixed"
        self.orderTest()


class TestN3D_boundaries(discretize.tests.OrderTest):
    name = "3D - Boundaries"
    meshTypes = ["uniformTensorMesh", "uniformTree", "rotateCurv"]
    meshDimension = 3
    expectedOrders = 2
    tolerance = 0.6
    meshSizes = [2, 4, 8, 16]
    # meshSizes = [4]

    def getError(self):
        # Test function
        phi = (
            lambda x: np.sin(np.pi * x[:, 0])
            * np.sin(np.pi * x[:, 1])
            * np.sin(np.pi * x[:, 2])
        )

        j_funX = (
            lambda x: np.pi
            * np.cos(np.pi * x[:, 0])
            * np.sin(np.pi * x[:, 1])
            * np.sin(np.pi * x[:, 2])
        )
        j_funY = (
            lambda x: np.pi
            * np.sin(np.pi * x[:, 0])
            * np.cos(np.pi * x[:, 1])
            * np.sin(np.pi * x[:, 2])
        )
        j_funZ = (
            lambda x: np.pi
            * np.sin(np.pi * x[:, 0])
            * np.sin(np.pi * x[:, 1])
            * np.cos(np.pi * x[:, 2])
        )

        q_fun = lambda x: -3 * (np.pi**2) * phi(x)

        mesh = self.M
        if self._meshType == "rotateCurv":
            nodes_x, nodes_y, nodes_z = mesh.node_list
            nodes_x -= 0.25
            nodes_y -= 0.25
            nodes_z -= 0.25
            mesh = discretize.CurvilinearMesh([nodes_x, nodes_y, nodes_z])
        else:
            mesh.origin = np.r_[-0.25, -0.25, -0.25]

        phi_ana = phi(mesh.nodes)
        q_ana = q_fun(mesh.nodes)

        if self.boundary_type == "Nuemann":
            # Nuemann with J defined at boundary nodes
            jx_bc = j_funX(mesh.boundary_nodes)
            jy_bc = j_funY(mesh.boundary_nodes)
            jz_bc = j_funZ(mesh.boundary_nodes)
            j_bc = np.c_[jx_bc, jy_bc, jz_bc]

            M_bn = mesh.boundary_node_vector_integral

            B_bc = sp.csr_matrix((mesh.n_nodes, mesh.n_nodes))
            b_bc = M_bn @ (j_bc.reshape(-1, order="F"))
        else:
            phi_bc = phi(mesh.boundary_faces)
            jx_bc = j_funX(mesh.boundary_faces)
            jy_bc = j_funY(mesh.boundary_faces)
            jz_bc = j_funZ(mesh.boundary_faces)
            j_bc = np.c_[jx_bc, jy_bc, jz_bc]

            j_bc_dot_n = np.sum(j_bc * mesh.boundary_face_outward_normals, axis=-1)

            # construct matrix with robin operator
            if self.boundary_type == "Robin":
                alpha = 1.0
            else:
                # get indices of x0, y0 and z0 boundaries
                n_boundary_faces = len(j_bc_dot_n)
                robin_locs = np.any(mesh.boundary_faces == -0.25, axis=1)

                alpha = np.zeros(n_boundary_faces)
                alpha[robin_locs] = 1.0

            beta = 1.0
            gamma = alpha * phi_bc + beta * j_bc_dot_n

            B_bc, b_bc = mesh.edge_divergence_weak_form_robin(alpha, beta, gamma)

        Me = mesh.get_edge_inner_product()
        Mn = sp.diags(mesh.average_node_to_cell.T @ mesh.cell_volumes)
        G = mesh.nodal_gradient

        A = -G.T @ Me @ G + B_bc
        rhs = Mn @ q_ana - b_bc

        if self.boundary_type == "Nuemann":
            P_b = sp.eye(mesh.n_nodes, format="csc")[:, 0]
            P_f = sp.eye(mesh.n_nodes, format="csc")[:, 1:]
            # P_f.T @ A @ (P_f @ x + P_b @ ana) = P_f.T @ rhs
            rhs = P_f.T @ rhs - (P_f.T @ A @ P_b) @ phi_ana[[0]]
            A = P_f.T @ A @ P_f

        phi_test = spsolve(A, rhs)

        if self.boundary_type == "Nuemann":
            phi_test = P_f @ phi_test + P_b @ phi_ana[[0]]
        err = np.linalg.norm(phi_test - phi_ana) / np.sqrt(mesh.n_nodes)
        return err

    def test_orderNuemannX(self):
        self.name = "3D - NuemannBoundary_Inverse"
        self.boundary_type = "Nuemann"
        self.orderTest()

    def test_orderRobinX(self):
        self.name = "3D - RobinBoundary_Inverse"
        self.boundary_type = "Robin"
        self.orderTest()

    def test_orderMixed(self):
        self.name = "3D - MixedBoundary_Inverse"
        self.boundary_type = "Mixed"
        self.orderTest()


if __name__ == "__main__":
    unittest.main()
