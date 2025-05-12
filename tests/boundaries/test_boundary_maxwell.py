import numpy as np
import discretize
from discretize import utils
from scipy.sparse.linalg import spsolve


class TestFz2D_InhomogeneousDirichlet(discretize.tests.OrderTest):
    name = "2D - Dirichlet"
    meshTypes = ["uniformTensorMesh", "uniformTree", "rotateCurv"]
    meshDimension = 2
    expectedOrders = [2, 2, 1]
    meshSizes = [4, 8, 16, 32, 64]

    def getError(self):
        # Test function
        # PDE: Curl(Curl Ez) + Ez = q
        # faces_z are cell_centers on 2D mesh
        ez_fun = lambda x: np.cos(np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1])
        q_fun = lambda x: (1 + 2 * np.pi**2) * ez_fun(x)

        mesh = self.M
        ez_ana = ez_fun(mesh.cell_centers)
        q_ana = q_fun(mesh.cell_centers)

        ez_bc = ez_fun(mesh.boundary_edges)

        MeI = mesh.get_edge_inner_product(invert_matrix=True)
        M_be = mesh.boundary_edge_vector_integral

        V = utils.sdiag(mesh.cell_volumes)
        C = mesh.edge_curl

        A = V @ C @ MeI @ C.T @ V + V
        rhs = V @ q_ana + V @ C @ MeI @ M_be @ ez_bc

        ez_test = spsolve(A, rhs)
        if self._meshType == "rotateCurv":
            err = np.linalg.norm(mesh.cell_volumes * (ez_test - ez_ana))
        else:
            err = np.linalg.norm(ez_test - ez_ana) / np.sqrt(mesh.n_cells)

        return err

    def test_orderX(self):
        self.name = "2D - InhomogeneousDirichlet_Inverse"
        self.myTest = "xc"
        self.orderTest()


class TestE3D_Inhomogeneous(discretize.tests.OrderTest):
    name = "3D - Maxwell"
    meshTypes = ["uniformTensorMesh", "uniformTree", "rotateCurv"]
    meshDimension = 3
    expectedOrders = [2, 2, 1]
    meshSizes = [4, 8, 16]

    def getError(self):
        # Test function
        # PDE: Curl(Curl E) + E = q
        def e_fun(x):
            x = np.cos(x)
            cx, cy, cz = x[:, 0], x[:, 1], x[:, 2]
            return np.c_[cy * cz, cx * cz, cx * cy]

        def h_fun(x):
            cx, cy, cz = np.cos(x[:, 0]), np.cos(x[:, 1]), np.cos(x[:, 2])
            sx, sy, sz = np.sin(x[:, 0]), np.sin(x[:, 1]), np.sin(x[:, 2])
            return np.c_[(-sy + sz) * cx, (sx - sz) * cy, (-sx + sy) * cz]

        def q_fun(x):
            return 3 * e_fun(x)

        mesh = self.M
        C = mesh.edge_curl

        if "Face" in self.myTest:
            e_ana = mesh.project_face_vector(e_fun(mesh.faces))
            q_ana = mesh.project_face_vector(q_fun(mesh.faces))

            e_bc = e_fun(mesh.boundary_edges).reshape(-1, order="F")

            MeI = mesh.get_edge_inner_product(invert_matrix=True)
            M_be = mesh.boundary_edge_vector_integral

            Mf = mesh.get_face_inner_product()

            A = Mf @ C @ MeI @ C.T @ Mf + Mf
            rhs = Mf @ q_ana + Mf @ C @ MeI @ M_be @ e_bc
        elif "Edge" in self.myTest:
            e_ana = mesh.project_edge_vector(e_fun(mesh.edges))
            q_ana = mesh.project_edge_vector(q_fun(mesh.edges))

            h_bc = h_fun(mesh.boundary_edges).reshape(-1, order="F")

            Mf = mesh.get_face_inner_product()
            Me = mesh.get_edge_inner_product()
            M_be = mesh.boundary_edge_vector_integral

            A = C.T @ Mf @ C + Me
            rhs = Me @ q_ana + M_be * h_bc

        e_test = spsolve(A, rhs)

        diff = e_test - e_ana
        if "Face" in self.myTest:
            if "Curv" in self._meshType or "Tree" in self._meshType:
                err = np.linalg.norm(Mf * diff)
            else:
                err = np.linalg.norm(e_test - e_ana) / np.sqrt(mesh.n_faces)
        elif "Edge" in self.myTest:
            if "Curv" in self._meshType or "Tree" in self._meshType:
                err = np.linalg.norm(Me * diff)
            else:
                err = np.linalg.norm(e_test - e_ana) / np.sqrt(mesh.n_edges)
        return err

    def test_orderFace(self):
        self.name = "3D - InhomogeneousDirichlet_Inverse"
        self.myTest = "Face - Dirichlet"
        self.orderTest()

    def test_orderNuemann(self):
        self.name = "3D - InhomogeneousDirichlet_Inverse"
        self.myTest = "Edge - Nuemann"
        self.orderTest()
