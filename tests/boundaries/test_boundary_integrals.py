import numpy as np
import scipy.sparse as sp
import discretize


def u(*args):
    if len(args) == 2:
        x, y = args
        return x**3 + y**2
    x, y, z = args
    return x**3 + y**2 + z**4


def v(*args):
    if len(args) == 2:
        x, y = args
        return np.c_[x**2, 3*y**3]
    x, y, z = args
    return np.c_[2*x**2, 3*y**3, -4*z**2]


def w(*args):
    if len(args) == 2:
        x, y = args
        return np.c_[(y-2)**2, (x+2)**2]
    x, y, z = args
    return np.c_[(y-2)**2 + z**2, (x+2)**2 - (z-4)**2, y**2-x**2]

# mesh will be on [0, 1] square

# int_V grad_u dot v dV = -4/15
# int_V u div_v dV = 27/20
# int_v curl_w dot v dV = 17/6

class Test3DBoundaryIntegral(discretize.tests.OrderTest):
    ame = "3D"
    meshTypes = [
        "uniformTensorMesh",
        "randomTensorMesh",
        "uniformTree",
        "uniformCurv",
        "rotateCurv",
        "sphereCurv"
        ]
    meshDimension = 3
    expectedOrders = [2, 1, 2, 2, 2, 0]
    meshSizes = [4, 8, 16, 32]

    def getError(self):
        mesh = self.M
        if self.myTest == "cell_grad":
            # Functions:
            u_cc = u(*mesh.cell_centers.T)
            faces = np.r_[mesh.faces_x, mesh.faces_y, mesh.faces_z]
            v_f = mesh.project_face_vector(v(*faces.T))
            u_bf = u(*mesh.boundary_faces.T)

            D = mesh.face_divergence
            M_c = sp.diags(mesh.cell_volumes)
            M_bf = mesh.boundary_face_scalar_integral

            discrete_val = -(v_f.T @ D.T) @ M_c @ u_cc + v_f.T @ (M_bf @ u_bf)
            if "sphere" not in self._meshType:
                true_val = -4/15
            else:
                true_val = 48*np.pi/35
        elif self.myTest == "edge_div":
            u_n = u(*mesh.nodes.T)
            edges = np.r_[mesh.edges_x, mesh.edges_y, mesh.edges_z]
            v_e = mesh.project_edge_vector(v(*edges.T))
            v_bn = v(*mesh.boundary_nodes.T).reshape(-1, order='F')

            M_e = mesh.get_edge_inner_product()
            G = mesh.nodal_gradient
            M_bn = mesh.boundary_node_vector_integral

            discrete_val = -(u_n.T @ G.T) @ M_e @ v_e + u_n.T @ (M_bn @ v_bn)
            if "sphere" not in self._meshType:
                true_val = 27/20
            else:
                true_val = 8*np.pi/5
        elif self.myTest == "face_curl":
            faces = np.r_[mesh.faces_x, mesh.faces_y, mesh.faces_z]
            edges = np.r_[mesh.edges_x, mesh.edges_y, mesh.edges_z]
            w_f = mesh.project_face_vector(w(*faces.T))
            v_e = mesh.project_edge_vector(v(*edges.T))
            w_be = w(*mesh.boundary_edges.T).reshape(-1, order='F')

            M_f = mesh.get_face_inner_product()
            Curl = mesh.edge_curl
            M_be = mesh.boundary_edge_vector_integral

            discrete_val = (v_e.T @ Curl.T) @ M_f @ w_f - v_e.T @ (M_be @ w_be)
            if "sphere" not in self._meshType:
                true_val = -79/6
            else:
                true_val = -64*np.pi/5

        return np.abs(discrete_val - true_val)

    def test_orderWeakCellGradIntegral(self):
        self.name = "3D - weak cell gradient integral w/boundary"
        self.myTest = "cell_grad"
        self.orderTest()

    def test_orderWeakEdgeDivIntegral(self):
        self.name = "3D - weak edge divergence integral w/boundary"
        self.myTest = "edge_div"
        self.orderTest()

    def test_orderWeakFaceCurlIntegral(self):
        self.name = "3D - weak face curl integral w/boundary"
        self.myTest = "face_curl"
        self.orderTest()
